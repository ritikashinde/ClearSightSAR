# novel implementation/src/utils/metrics.py

import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips
import os
import sys
import numpy as np 

# --- Initialization Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_RANGE = 2.0  # Data is normalized to [-1, 1], so max_val - min_val = 2.0

# --- LPIPS Initialization (Perceptual Distance) ---
try:
    LPIPS_METRIC = lpips.LPIPS(net='alex', version='0.1').to(DEVICE)
    LPIPS_METRIC.eval() 
except Exception as e:
    print(f"LPIPS initialization failed on {DEVICE}. Continuing without LPIPS.")
    LPIPS_METRIC = None

# --- Standard Metrics Initialization (CRITICAL FIX: data_range=2.0) ---
PSNR_METRIC = PeakSignalNoiseRatio(data_range=DATA_RANGE).to(DEVICE)
SSIM_METRIC = StructuralSimilarityIndexMeasure(data_range=DATA_RANGE).to(DEVICE)


def calculate_metrics(pred, target):
    """
    Calculates PSNR, SSIM, and LPIPS given two image tensors in the range [-1, 1].
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    pred = pred.to(DEVICE)
    target = target.to(DEVICE)
    
    # Calculate Metrics (using temporary metric instances for easy calculation)
    # We use a temporary instance here to avoid cross-batch state corruption
    psnr_temp = PeakSignalNoiseRatio(data_range=DATA_RANGE).to(DEVICE)
    ssim_temp = StructuralSimilarityIndexMeasure(data_range=DATA_RANGE).to(DEVICE)
    
    psnr = psnr_temp(pred, target).item()
    ssim = ssim_temp(pred, target).item()
    
    # LPIPS 
    if LPIPS_METRIC:
        lpips_score = LPIPS_METRIC(pred, target).mean().item()
    else:
        lpips_score = np.nan 

    return {
        'PSNR': psnr,
        'SSIM': ssim,
        'LPIPS': lpips_score
    }