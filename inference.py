import os
import yaml
import math
from argparse import ArgumentParser
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from tqdm import tqdm
import numpy as np
import cv2  # pyright: ignore[reportMissingImports]
import time

try:
    from thop import profile as thop_profile  # pyright: ignore[reportMissingImports]
except ImportError:
    thop_profile = None

# Ensure these imports match your project structure
from data.EventRainEFFT2D import EventRainEFFT2D
from model_interface import ModelInterface 
# from model_interface_dual import ModelInterfaceDual # Not used for single head
from data_interface import DataInterface
from configs.config_schema import load_config_with_schema, AppConfig


def _count_parameters_millions(model: torch.nn.Module) -> float:
    """
    Counts trainable model parameters in millions.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def _estimate_gflops(model: torch.nn.Module, sample_input: torch.Tensor):
    """
    Estimates GFLOPs for one forward pass (batch=1).
    Tries THOP first, then falls back to torch.profiler FLOP stats.
    Returns None if profiling fails.
    """
    if thop_profile is not None:
        try:
            with torch.no_grad():
                macs, _ = thop_profile(model, inputs=(sample_input[:1],), verbose=False)
            # Common conversion: FLOPs ~= 2 * MACs.
            return (2.0 * macs) / 1e9
        except Exception:
            pass

    # Fallback path when THOP is unavailable.
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if sample_input.device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        with torch.no_grad():
            with torch.profiler.profile(
                activities=activities,
                with_flops=True,
                record_shapes=False,
            ) as prof:
                _ = model(sample_input[:1])

        total_flops = 0.0
        for event in prof.key_averages():
            total_flops += float(getattr(event, "flops", 0.0) or 0.0)

        if total_flops > 0.0:
            return total_flops / 1e9
    except Exception:
        pass

    return None


def _print_inference_metric_table(
    title: str,
    gflops: float,
    params_m: float,
    total_infer_time_s: float,
    num_samples: int,
):
    """
    Prints a compact metrics table for model complexity and runtime.
    """
    avg_time_per_sample = total_infer_time_s / max(num_samples, 1)
    gflops_text = f"{gflops:.3f}" if gflops is not None else "N/A"

    print("\n" + "=" * 60)
    print(f"{title} Metrics")
    print("=" * 60)
    print(f"{'Metric':<30} | {'Value':>24}")
    print("-" * 60)
    print(f"{'GFLOPs (batch=1)':<30} | {gflops_text:>24}")
    print(f"{'Parameters (Millions)':<30} | {params_m:>24.3f}")
    print(f"{'Inference Time (s, total)':<30} | {total_infer_time_s:>24.4f}")
    print(f"{'Inference Time (s/sample)':<30} | {avg_time_per_sample:>24.6f}")
    print("=" * 60 + "\n")


def calculate_components(pred_map, gt_map):
    """
    Computes Intersection and Total for basic SR calculation.
    """
    intersection = torch.sum(gt_map * pred_map)
    total = torch.sum(gt_map)
    return intersection.item(), total.item()

def calculate_nr_single_head(derained_bin, rain_gt):
    """
    Computes Noise Removal (NR) for a single head model.
    NR = (Total Rain Pixels - Rain Pixels remaining in Derained) / Total Rain Pixels
    """
    # Rain pixels that exist in input but were NOT removed (present in derained prediction)
    remaining_rain = torch.sum(rain_gt * derained_bin)
    
    total_rain = torch.sum(rain_gt)
    
    # Successfully removed rain = Total - Remaining
    removed_rain = total_rain - remaining_rain
    
    return removed_rain.item(), total_rain.item()


def calculate_segmentation_metrics(pred_bin, gt_bin):
    """
    Computes IoU, Dice, and PSNR for a binary batch.
    """
    # True Positives, False Positives, False Negatives
    tp = torch.sum(pred_bin * gt_bin).item()
    fp = torch.sum(pred_bin * (1 - gt_bin)).item()
    fn = torch.sum((1 - pred_bin) * gt_bin).item()
    
    # 1. IoU (Intersection over Union)
    union = tp + fp + fn
    iou = tp / (union + 1e-8)
    
    # 2. Dice (F1 Score)
    # Dice = 2*TP / (2*TP + FP + FN)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    
    # 3. PSNR (Peak Signal-to-Noise Ratio)
    # MSE between binary maps. Max value is 1.0.
    mse = torch.mean((pred_bin - gt_bin) ** 2).item()
    if mse == 0:
        psnr = 100.0 # Perfect reconstruction
    else:
        psnr = 10 * math.log10(1.0 / mse)
        
    return iou, dice, psnr


def _tensor_to_u8_gray(sample: torch.Tensor) -> np.ndarray:
    """
    Converts a single sample tensor (C,H,W) into uint8 grayscale image.
    """
    if sample.ndim == 3:
        if sample.shape[0] == 1:
            image = sample[0]
        else:
            image = sample.mean(dim=0)
    else:
        image = sample

    image = torch.clamp(image.detach().float().cpu(), 0.0, 1.0)
    return (image.numpy() * 255.0).astype(np.uint8)


def _build_labeled_2x3_frame(
    top_row: tuple,
    bottom_row: tuple,
    top_label: str,
    bottom_label: str,
) -> np.ndarray:
    """
    Builds one white-background visualization frame:
      rows: rain intensity (2 rows), cols: raw / derained / GT (3 cols)
    """
    row_gap = 6
    col_gap = 6
    top_margin = 72
    left_margin = 165
    right_margin = 24
    bottom_margin = 70

    # Each element in row tuples is a uint8 grayscale frame.
    h, w = top_row[0].shape
    canvas_h = top_margin + (2 * h) + row_gap + bottom_margin
    canvas_w = left_margin + (3 * w) + (2 * col_gap) + right_margin
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    col_names = ["Raw image", "Derained image", "Ground Truth"]
    row_names = [top_label, bottom_label]
    rows = [top_row, bottom_row]

    # Column labels.
    for col in range(3):
        x = left_margin + col * (w + col_gap)
        cv2.putText(
            canvas,
            col_names[col],
            (x + 8, canvas_h - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (90, 90, 90),
            2,
            lineType=cv2.LINE_AA,
        )

    # Row labels and image tiles.
    for row in range(2):
        y = top_margin + row * (h + row_gap)
        cv2.putText(
            canvas,
            row_names[row],
            (28, y + (h // 2) + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (90, 90, 90),
            2,
            lineType=cv2.LINE_AA,
        )

        for col in range(3):
            x = left_margin + col * (w + col_gap)
            tile_gray = rows[row][col]
            tile_bgr = cv2.cvtColor(tile_gray, cv2.COLOR_GRAY2BGR)
            canvas[y:y + h, x:x + w] = tile_bgr

    return canvas


def _write_comparison_video(
    per_type_frames: dict,
    rain_types: list,
    output_path: str,
    fps: int,
):
    """
    Writes a 2x3 comparison video for two rain intensities.
    """
    first_type, second_type = rain_types
    n_frames = min(len(per_type_frames[first_type]), len(per_type_frames[second_type]))

    if n_frames == 0:
        print(
            f"[Video] Skipped writing because no paired frames were found for "
            f"{first_type} and {second_type}."
        )
        return

    first_frame = _build_labeled_2x3_frame(
        per_type_frames[first_type][0],
        per_type_frames[second_type][0],
        first_type.replace("mm", " mm"),
        second_type.replace("mm", " mm"),
    )
    height, width = first_frame.shape[:2]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for: {output_path}")

    try:
        for i in range(n_frames):
            frame = _build_labeled_2x3_frame(
                per_type_frames[first_type][i],
                per_type_frames[second_type][i],
                first_type.replace("mm", " mm"),
                second_type.replace("mm", " mm"),
            )
            writer.write(frame)
    finally:
        writer.release()

    print(f"[Video] Saved {n_frames} frames at {fps} FPS -> {output_path}")


def _build_labeled_1x2_frame(raw_img: np.ndarray, derained_img: np.ndarray) -> np.ndarray:
    """
    Builds one white-background frame with columns: raw input, derained output.
    """
    row_gap = 6
    col_gap = 6
    top_margin = 24
    left_margin = 24
    right_margin = 24
    bottom_margin = 70

    h, w = raw_img.shape
    canvas_h = top_margin + h + row_gap + bottom_margin
    canvas_w = left_margin + (2 * w) + col_gap + right_margin
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    labels = ["Raw image", "Derained image"]
    for col in range(2):
        x = left_margin + col * (w + col_gap)
        cv2.putText(
            canvas,
            labels[col],
            (x + 8, canvas_h - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (90, 90, 90),
            2,
            lineType=cv2.LINE_AA,
        )

    canvas[top_margin:top_margin + h, left_margin:left_margin + w] = cv2.cvtColor(
        raw_img, cv2.COLOR_GRAY2BGR
    )
    x2 = left_margin + w + col_gap
    canvas[top_margin:top_margin + h, x2:x2 + w] = cv2.cvtColor(
        derained_img, cv2.COLOR_GRAY2BGR
    )
    return canvas


def _write_realworld_video(frames: list, output_path: str, fps: int):
    """
    Writes a 1x2 real-world comparison video (raw, derained).
    """
    if len(frames) == 0:
        print(f"[Video] Skipped writing empty sequence: {output_path}")
        return

    first = _build_labeled_1x2_frame(frames[0][0], frames[0][1])
    h, w = first.shape[:2]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for: {output_path}")

    try:
        for raw_u8, derained_u8 in frames:
            writer.write(_build_labeled_1x2_frame(raw_u8, derained_u8))
    finally:
        writer.release()

    print(f"[Video] Saved {len(frames)} frames at {fps} FPS -> {output_path}")


def run_realworld_inference(model, cfg, device, args):
    """
    Runs prediction on merge-only real-world data and writes per-intensity videos.
    """
    print(f"\n[RealWorld] Running inference on {args.realworld_root}")
    dataset_kwargs = asdict(cfg.DATA.dataset)
    dataset_kwargs.pop("class_name", None)
    dataset_kwargs.pop("file_name", None)
    dataset_kwargs["root"] = args.realworld_root

    realworld_set = EventRainEFFT2D(
        **dataset_kwargs,
        purpose=args.realworld_purpose,
        require_raw=False,
    )

    dl_cfg = cfg.DATA.dataloader
    test_batch = dl_cfg.test_batch_size or dl_cfg.batch_size
    realworld_loader = DataLoader(
        dataset=realworld_set,
        batch_size=test_batch,
        num_workers=dl_cfg.num_workers,
        shuffle=False,
        persistent_workers=dl_cfg.persistent_workers,
        pin_memory=dl_cfg.pin_memory,
        multiprocessing_context=dl_cfg.multiprocessing_context,
        drop_last=False,
    )

    per_type_frames = {}
    params_m = _count_parameters_millions(model)
    gflops = None
    total_infer_time_s = 0.0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(realworld_loader, desc="Real-world inference"):
            rainy = batch["merge"].to(device)
            rain_type_list = batch["rain_type"]

            if gflops is None:
                gflops = _estimate_gflops(model, rainy)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            derained = model(rainy)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_infer_time_s += (time.perf_counter() - start_time)
            total_samples += rainy.size(0)

            if isinstance(derained, (tuple, list)):
                derained = derained[0]

            if args.realworld_binarize:
                derained_vis = (derained > args.realworld_threshold).float()
            else:
                derained_vis = torch.clamp(derained, 0.0, 1.0)

            for k in range(derained_vis.size(0)):
                r_type = rain_type_list[k] if k < len(rain_type_list) else "unknown"
                if r_type not in per_type_frames:
                    per_type_frames[r_type] = []
                per_type_frames[r_type].append(
                    (_tensor_to_u8_gray(rainy[k]), _tensor_to_u8_gray(derained_vis[k]))
                )

    _print_inference_metric_table(
        title="[RealWorld] Inference",
        gflops=gflops,
        params_m=params_m,
        total_infer_time_s=total_infer_time_s,
        num_samples=total_samples,
    )

    if args.no_save:
        print("[RealWorld] --no-save set; skipping video writing.")
        return

    for r_type, frames in per_type_frames.items():
        output_path = os.path.join(
            args.realworld_output_dir,
            f"realworld_{r_type}_raw_derained_{args.video_fps}fps.mp4",
        )
        _write_realworld_video(frames=frames, output_path=output_path, fps=args.video_fps)


def tune_thresholds_on_train(model, train_loader, device, num_steps=19):
    """
    Phase 1: Iterates through the training set to find the threshold that maximizes 
    F1-Score (Dice) for the Derained Signal.
    """
    print(f"\n[Phase 1] Tuning threshold on Training Set ({len(train_loader.dataset)} samples)...")
    
    # Sweep range
    taus = torch.linspace(0.05, 0.95, steps=num_steps, device=device)
    tau_values = [round(t.item(), 3) for t in taus]
    
    # Storage for accumulated F1 scores
    # structure: tau -> list of scores
    global_f1_d_storage = {t: [] for t in tau_values}

    model.eval()
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Calibrating"):
            gt = batch['raw'].to(device)
            rainy = batch['merge'].to(device)
            
            # Forward Pass (Single Output)
            derained = model(rainy)
            # Handle tuple return if model output varies (e.g. derained, loss)
            if isinstance(derained, (tuple, list)):
                derained = derained[0]

            # Pre-calculate Ground Truths
            gt_binary = (gt > 0.1).float()

            # Sweep all thresholds for this batch
            for t in tau_values:
                # Tuning Tau_Derained to maximize Dice (F1)
                derained_bin = (derained > t).float()
                _, dice_d, _ = calculate_segmentation_metrics(derained_bin, gt_binary)
                global_f1_d_storage[t].append(dice_d)

    # --- Find Optimal Threshold ---
    print("Calculating optimal statistics (Maximizing F1/Dice)...")

    best_tau_d = 0.5
    best_train_f1_d = -1.0
    for t in tau_values:
        avg_f1 = np.mean(global_f1_d_storage[t])
        if avg_f1 > best_train_f1_d:
            best_train_f1_d = avg_f1
            best_tau_d = t

    print(f"  > Selected Tau_Derained: {best_tau_d} (Train F1: {best_train_f1_d:.4f})")
    
    return best_tau_d


def evaluate_test_set(model, test_loader, tau_d, device, args):
    """
    Phase 2: Runs inference on Test Set using fixed threshold found in Phase 1.
    Calculates standard (SR, NR, DA) and advanced (IoU, Dice, PSNR) metrics.
    Builds a single 2x3 comparison video (2 rain intensities x 3 columns).
    """
    print(f"\n[Phase 2] Evaluating Test Set using Tau_D={tau_d}...")

    # Initialize storage for all metrics
    metrics = {
        '50mm':    {'sr': [], 'nr': [], 'da': [], 'iou': [], 'dice': [], 'psnr': []},
        '100mm':   {'sr': [], 'nr': [], 'da': [], 'iou': [], 'dice': [], 'psnr': []},
        '150mm':   {'sr': [], 'nr': [], 'da': [], 'iou': [], 'dice': [], 'psnr': []},
        'unknown': {'sr': [], 'nr': [], 'da': [], 'iou': [], 'dice': [], 'psnr': []}
    }
    
    base_path = '/fs/nexus-scratch/tuxunlu/git/Event-Deraining/inference'
    target_rain_types = [x.strip() for x in args.video_rain_types.split(",") if x.strip()]
    if len(target_rain_types) != 2:
        raise ValueError("--video-rain-types must contain exactly two comma-separated values.")
    per_type_frames = {target_rain_types[0]: [], target_rain_types[1]: []}
    params_m = _count_parameters_millions(model)
    gflops = None
    total_infer_time_s = 0.0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            gt = batch['raw'].to(device)
            rainy = batch['merge'].to(device)
            rain_type_list = batch['rain_type']
            
            if gflops is None:
                gflops = _estimate_gflops(model, rainy)

            # Forward Pass (Single Output)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            derained = model(rainy)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_infer_time_s += (time.perf_counter() - start_time)
            total_samples += rainy.size(0)

            if isinstance(derained, (tuple, list)):
                derained = derained[0]
            
            # --- Metrics Calculation (Using Fixed Threshold) ---
            gt_binary = (gt > 0.1).float()
            rainy_binary = (rainy > 0.1).float()
            
            # Apply Threshold
            derained_bin = (derained > tau_d).float()
            
            # --- 1. Standard Metrics: SR, NR, DA ---
            
            # SR (Signal Retention): How much of GT is preserved?
            pb, tb = calculate_components(derained_bin, gt_binary)
            sr = pb / (tb + 1e-8)
            
            # NR (Noise Removal): How much Rain is ABSENT in derained?
            # rain_gt is pixels that are in Input but NOT in GT
            rain_gt = rainy_binary * (1 - gt_binary)
            pr, tr = calculate_nr_single_head(derained_bin, rain_gt)
            nr = pr / (tr + 1e-8)
            
            # DA
            da = 0.5 * (sr + nr)
            
            # --- 2. Advanced Metrics: IoU, Dice, PSNR ---
            # Evaluated on the Signal (Derained) vs Ground Truth
            iou, dice, psnr = calculate_segmentation_metrics(derained_bin, gt_binary)
            
            # --- Store Metrics (Per Batch Aggregation) ---
            # Handle batch size > 1 if necessary, though current loop logic implies aggregation
            # We append per-batch averages here.
            
            r_type = rain_type_list[0] if len(rain_type_list) > 0 else 'unknown'
            if r_type not in metrics:
                r_type = 'unknown'
            
            metrics[r_type]['sr'].append(sr)
            metrics[r_type]['nr'].append(nr)
            metrics[r_type]['da'].append(da)
            metrics[r_type]['iou'].append(iou)
            metrics[r_type]['dice'].append(dice)
            metrics[r_type]['psnr'].append(psnr)

            # --- Collect Frames for Video ---
            if not args.no_save:
                batch_size = derained_bin.size(0)
                for k in range(batch_size):
                    this_r_type = rain_type_list[k] if k < len(rain_type_list) else 'unknown'
                    if this_r_type not in per_type_frames:
                        continue

                    frame_triplet = (
                        _tensor_to_u8_gray(rainy[k]),
                        _tensor_to_u8_gray(derained_bin[k]),
                        _tensor_to_u8_gray(gt[k]),
                    )
                    per_type_frames[this_r_type].append(frame_triplet)

    # --- Final Reporting ---
    print("\n" + "=" * 60)
    print(f"Final Test Results (Fixed Tau_D={tau_d})")
    print("=" * 60)
    for r_type, vals in metrics.items():
        if len(vals['da']) > 0:
            print(f"Results for [{r_type}] over {len(vals['da'])} samples:")
            # Original Metrics
            print(f"  Signal Retention (SR):   {np.mean(vals['sr']):.4f}")
            print(f"  Noise Removal (NR):      {np.mean(vals['nr']):.4f}")
            print(f"  Denoising Accuracy (DA): {np.mean(vals['da']):.4f}")
            print("-" * 30)
            # New Metrics
            print(f"  Intersection over Union: {np.mean(vals['iou']):.4f}")
            print(f"  Dice Score (F1):         {np.mean(vals['dice']):.4f}")
            print(f"  PSNR (dB):               {np.mean(vals['psnr']):.2f}")
            print("-" * 60)
    print("\n")
    _print_inference_metric_table(
        title="[Phase 2] Test Inference",
        gflops=gflops,
        params_m=params_m,
        total_infer_time_s=total_infer_time_s,
        num_samples=total_samples,
    )

    if not args.no_save:
        output_path = os.path.join(
            base_path,
            f"comparison_{target_rain_types[0]}_{target_rain_types[1]}_{args.video_fps}fps.mp4",
        )
        _write_comparison_video(
            per_type_frames=per_type_frames,
            rain_types=target_rain_types,
            output_path=output_path,
            fps=args.video_fps,
        )


def main():
    parser = ArgumentParser()
    parser.add_argument('--config_path', default='config/config.yaml', help='YAML config file')
    parser.add_argument('--test_checkpoint', required=True, help='Path to .pth or .ckpt checkpoint')
    parser.add_argument('--no-save', action='store_true', help='If set, do not save output video')
    parser.add_argument('--video-fps', type=int, default=10, help='Output comparison video FPS')
    parser.add_argument(
        '--video-rain-types',
        type=str,
        default='50mm,150mm',
        help='Exactly two rain intensity labels used as top,bottom rows (e.g. 50mm,150mm)',
    )
    parser.add_argument(
        '--realworld-root',
        type=str,
        default=None,
        help='Root of real-world dataset (merge_data only is supported).',
    )
    parser.add_argument(
        '--realworld-purpose',
        type=str,
        default='test',
        help='Purpose subfolder under merge_data for real-world root (if present).',
    )
    parser.add_argument(
        '--realworld-output-dir',
        type=str,
        default='/fs/nexus-scratch/tuxunlu/git/Event-Deraining/inference/realworld',
        help='Output directory for real-world inference videos.',
    )
    parser.add_argument(
        '--realworld-binarize',
        action='store_true',
        help='If set, binarize real-world output with --realworld-threshold.',
    )
    parser.add_argument(
        '--realworld-threshold',
        type=float,
        default=0.5,
        help='Threshold used only when --realworld-binarize is enabled.',
    )
    args = parser.parse_args()

    cfg, _ = load_config_with_schema(args.config_path)

    # Load Model (Single Interface)
    model_interface_kwargs = {
        "model_cfg": cfg.MODEL,
        "optimizer_cfg": cfg.OPTIMIZER,
        "scheduler_cfg": cfg.SCHEDULER,
        "training_cfg": cfg.TRAINING,
        "data_cfg": cfg.DATA,
    }
    
    # Use standard ModelInterface for single-head models
    model_module = ModelInterface.load_from_checkpoint(
            args.test_checkpoint,
            strict=False,
            **model_interface_kwargs,
        )
    
    # Load Data
    data_module = DataInterface(**{"data_cfg": cfg.DATA})

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_module.to(device)

    if args.realworld_root is not None:
        run_realworld_inference(model_module, cfg, device, args)
        return

    # 1. Tune on Training Set
    train_loader = data_module.train_dataloader()
    tau_d = tune_thresholds_on_train(model_module, train_loader, device)

    # 2. Evaluate on Test Set
    test_loader = data_module.test_dataloader()
    evaluate_test_set(model_module, test_loader, tau_d, device, args)


if __name__ == "__main__":
    main()