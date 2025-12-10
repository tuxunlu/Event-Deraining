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

from data.EventRainEFFT2D import EventRainEFFT2D
from model_interface import ModelInterface
from model_interface_dual import ModelInterfaceDual
from data_interface import DataInterface
from configs.config_schema import load_config_with_schema, AppConfig


def calculate_components(pred_map, gt_map):
    """
    Computes Intersection and Total for basic SR/NR calculation.
    """
    intersection = torch.sum(gt_map * pred_map)
    total = torch.sum(gt_map)
    return intersection.item(), total.item()


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
    mse = torch.mean((pred_bin - gt_bin) ** 2).item()
    if mse == 0:
        psnr = 100.0 # Perfect reconstruction
    else:
        psnr = 10 * math.log10(1.0 / mse)
        
    return iou, dice, psnr


def tune_thresholds_on_train(model, train_loader, device, num_steps=19):
    """
    Phase 1: Iterates through the training set to find the thresholds that maximize 
    F1-Score (Dice) for both the Signal (Derained) and Noise (Rain) heads.
    """
    print(f"\n[Phase 1] Tuning thresholds on Training Set ({len(train_loader.dataset)} samples)...")
    
    # Sweep range
    taus = torch.linspace(0.05, 0.95, steps=num_steps, device=device)
    tau_values = [round(t.item(), 3) for t in taus]
    
    # Storage structure: tau -> list of scores
    dice_derained_storage = {t: [] for t in tau_values}
    dice_rain_storage = {t: [] for t in tau_values}
    
    # (Optional) Store IoU just for logging, though we optimize on Dice
    iou_derained_storage = {t: [] for t in tau_values}

    model.eval()
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Calibrating"):
            gt = batch['raw'].to(device)
            rainy = batch['merge'].to(device)
            
            # Forward Pass
            derained, pred_rain, _ = model(rainy)

            # Ground Truths
            gt_binary = (gt > 0.1).float()
            rainy_binary = (rainy > 0.1).float()
            rain_gt = rainy_binary * (1 - gt_binary) # Pure rain noise

            # Sweep
            for t in tau_values:
                # 1. Derained Head vs Signal GT
                derained_bin = (derained > t).float()
                iou_d, dice_d, _ = calculate_segmentation_metrics(derained_bin, gt_binary)
                dice_derained_storage[t].append(dice_d)
                iou_derained_storage[t].append(iou_d)

                # 2. Rain Head vs Rain GT
                pred_rain_bin = (pred_rain > t).float()
                _, dice_r, _ = calculate_segmentation_metrics(pred_rain_bin, rain_gt)
                dice_rain_storage[t].append(dice_r)

    # --- Find Optimal Thresholds ---
    print("Calculating optimal statistics (Maximizing F1/Dice)...")

    # Best Tau for Derained (Signal)
    best_tau_d = 0.5
    best_dice_d = -1.0
    for t in tau_values:
        avg_dice = np.mean(dice_derained_storage[t])
        if avg_dice > best_dice_d:
            best_dice_d = avg_dice
            best_tau_d = t
            
    # Best Tau for Rain (Noise)
    best_tau_r = 0.5
    best_dice_r = -1.0
    for t in tau_values:
        avg_dice = np.mean(dice_rain_storage[t])
        if avg_dice > best_dice_r:
            best_dice_r = avg_dice
            best_tau_r = t

    # Retrieve corresponding IoU for logging
    best_iou_d = np.mean(iou_derained_storage[best_tau_d])

    print(f"  > Selected Tau_Derained: {best_tau_d} (Train F1: {best_dice_d:.4f}, IoU: {best_iou_d:.4f})")
    print(f"  > Selected Tau_Rain:     {best_tau_r} (Train F1: {best_dice_r:.4f})")
    
    return best_tau_d, best_tau_r


def evaluate_test_set(model, test_loader, tau_d, tau_r, device, args):
    """
    Phase 2: Runs inference on Test Set using fixed thresholds found in Phase 1.
    Saves images individually (one file per sample).
    """
    print(f"\n[Phase 2] Evaluating Test Set using Tau_D={tau_d} and Tau_R={tau_r}...")

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
            
            derained, pred_rain, _ = model(rainy)
            
            # --- Apply Thresholds ---
            gt_binary = (gt > 0.1).float()
            derained_bin = (derained > tau_d).float()
            
            rainy_binary = (rainy > 0.1).float()
            rain_gt = rainy_binary * (1 - gt_binary)
            pred_rain_bin = (pred_rain > tau_r).float()
            
            # --- Metrics ---
            
            # 1. Standard (SR, NR, DA)
            pb, tb = calculate_components(derained_bin, gt_binary)
            sr = pb / (tb + 1e-8)
            
            pr, tr = calculate_components(pred_rain_bin, rain_gt)
            nr = pr / (tr + 1e-8)
            da = 0.5 * (sr + nr)
            
            # 2. Advanced (IoU, Dice, PSNR)
            iou, dice, psnr = calculate_segmentation_metrics(derained_bin, gt_binary)
            
            # --- Store ---
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
                batch_size = derained_bin.size(0)
                
                for k in range(batch_size):
                    this_r_type = rain_type_list[k] if k < len(rain_type_list) else 'unknown'
                    
                    out_dirs = {
                        'derained': os.path.join(base_path, this_r_type, 'derained'),
                        'gt': os.path.join(base_path, this_r_type, 'gt'),
                        'rainy': os.path.join(base_path, this_r_type, 'rainy'),
                        'pred_rain': os.path.join(base_path, this_r_type, 'pred_rain'),
                    }
                    for d in out_dirs.values():
                        os.makedirs(d, exist_ok=True)

                    fname_suffix = f"batch{idx}_sample{k}.png"

                    torchvision.utils.save_image(
                        derained_bin[k],
                        os.path.join(out_dirs['derained'], f"derained_{this_r_type}_{fname_suffix}"),
                        normalize=True, scale_each=True
                    )
                    torchvision.utils.save_image(
                        pred_rain_bin[k],
                        os.path.join(out_dirs['pred_rain'], f"pred_rain_{this_r_type}_{fname_suffix}"),
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
    print(f"Final Test Results (Fixed Tau_D={tau_d}, Tau_R={tau_r})")
    print("=" * 60)
    
    for r_type, vals in metrics.items():
        if len(vals['da']) > 0:
            print(f"Results for [{r_type}] over {len(vals['da'])} samples:")
            print(f"  Signal Retention (SR):   {np.mean(vals['sr']):.4f}")
            print(f"  Noise Removal (NR):      {np.mean(vals['nr']):.4f}")
            print(f"  Denoising Accuracy (DA): {np.mean(vals['da']):.4f}")
            print("-" * 30)
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

    model_interface_kwargs = {
        "model_cfg": cfg.MODEL,
        "optimizer_cfg": cfg.OPTIMIZER,
        "scheduler_cfg": cfg.SCHEDULER,
        "training_cfg": cfg.TRAINING,
        "data_cfg": cfg.DATA,
    }
    
    model_module = ModelInterfaceDual.load_from_checkpoint(
            args.test_checkpoint,
            strict=False,
            **model_interface_kwargs,
        )
    
    data_module = DataInterface(**{"data_cfg": cfg.DATA})

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_module.to(device)

    # 1. Tune on Training Set
    train_loader = data_module.train_dataloader()
    tau_d, tau_r = tune_thresholds_on_train(model_module, train_loader, device)

    # 2. Evaluate on Test Set
    test_loader = data_module.test_dataloader()
    evaluate_test_set(model_module, test_loader, tau_d, tau_r, device, args)


if __name__ == "__main__":
    main()