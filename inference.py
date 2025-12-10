import os
import yaml
import math
from argparse import ArgumentParser

import torch
import torchvision
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from tqdm import tqdm
import numpy as np

# Ensure these imports match your project structure
from data.EventRainEFFT2D import EventRainEFFT2D
from model_interface import ModelInterface 
# from model_interface_dual import ModelInterfaceDual # Not used for single head
from data_interface import DataInterface
from configs.config_schema import load_config_with_schema, AppConfig


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
    Saves images one by one.
    """
    print(f"\n[Phase 2] Evaluating Test Set using Tau_D={tau_d}...")

    # Initialize storage for all metrics
    metrics = {
        '50mm':    {'sr': [], 'nr': [], 'da': [], 'iou': [], 'dice': [], 'psnr': []},
        '100mm':   {'sr': [], 'nr': [], 'da': [], 'iou': [], 'dice': [], 'psnr': []},
        '150mm':   {'sr': [], 'nr': [], 'da': [], 'iou': [], 'dice': [], 'psnr': []},
        'unknown': {'sr': [], 'nr': [], 'da': [], 'iou': [], 'dice': [], 'psnr': []}
    }
    
    base_path = '/fs/nexus-scratch/tuxunlu/git/event-based-deraining/inference'

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            gt = batch['raw'].to(device)
            rainy = batch['merge'].to(device)
            rain_type_list = batch['rain_type']
            
            # Forward Pass (Single Output)
            derained = model(rainy)
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

            # --- Save Images (One by One) ---
            if not args.no_save:
                # Iterate over the batch dimension to save individual files
                batch_size = derained_bin.size(0)
                
                for k in range(batch_size):
                    # Robustly get rain type for this specific sample
                    this_r_type = rain_type_list[k] if k < len(rain_type_list) else 'unknown'
                    
                    out_dirs = {
                        'derained': os.path.join(base_path, this_r_type, 'derained'),
                        'gt': os.path.join(base_path, this_r_type, 'gt'),
                        'rainy': os.path.join(base_path, this_r_type, 'rainy'),
                        # 'pred_rain': No pred_rain to save
                    }
                    for d in out_dirs.values():
                        os.makedirs(d, exist_ok=True)

                    # Unique filename using batch index AND sample index within batch
                    fname_suffix = f"batch{idx}_sample{k}.png"

                    torchvision.utils.save_image(
                        derained_bin[k],
                        os.path.join(out_dirs['derained'], f"derained_{this_r_type}_{fname_suffix}"),
                        normalize=True, scale_each=True
                    )
                    torchvision.utils.save_image(
                        gt[k],
                        os.path.join(out_dirs['gt'], f"gt_{this_r_type}_{fname_suffix}"),
                        normalize=True, scale_each=True
                    )
                    torchvision.utils.save_image(
                        rainy[k],
                        os.path.join(out_dirs['rainy'], f"rainy_{this_r_type}_{fname_suffix}"),
                        normalize=True, scale_each=True
                    )

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


def main():
    parser = ArgumentParser()
    parser.add_argument('--config_path', default='config/config.yaml', help='YAML config file')
    parser.add_argument('--test_checkpoint', required=True, help='Path to .pth or .ckpt checkpoint')
    parser.add_argument('--no-save', action='store_true', help='If set, do not save output images')
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

    # 1. Tune on Training Set
    train_loader = data_module.train_dataloader()
    tau_d = tune_thresholds_on_train(model_module, train_loader, device)

    # 2. Evaluate on Test Set
    test_loader = data_module.test_dataloader()
    evaluate_test_set(model_module, test_loader, tau_d, device, args)


if __name__ == "__main__":
    main()