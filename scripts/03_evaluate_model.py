# novel implementation/scripts/03_evaluate_model.py

import torch
import os
import sys
import yaml
from tqdm import tqdm
from diffusers import DDPMScheduler
import numpy as np

# Add the project root to the path for correct imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.data.dataset import get_dataloader 
from src.utils.metrics import calculate_metrics 
from src.models.vae_encoder import VAEEncoder
from src.models.unet_model import ConditionalUNet


# --- CONFIGURATION ---
UNET_WEIGHTS_PATH = 'data/models/diffusion_unet_final.pth'
# NEW
ENCODER_WEIGHTS_PATH = 'data/models/vae_autoencoder_final.pth'
OUTPUT_DIR = 'outputs/evaluation_results.txt'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open(os.path.join(PROJECT_ROOT, 'configs', 'model.yaml'), 'r') as f:
    MODEL_CONFIG = yaml.safe_load(f)
with open(os.path.join(PROJECT_ROOT, 'configs', 'train.yaml'), 'r') as f:
    TRAIN_CONFIG = yaml.safe_load(f)['DIFFUSION_TRAIN']
# ---------------------

def sample_and_evaluate():
    print(f"Starting Evaluation on {DEVICE}...")

    # 1. Load Trained Models
    encoder = VAEEncoder(**MODEL_CONFIG['VAE_ENCODER']).to(DEVICE)
    try:
        encoder.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, ENCODER_WEIGHTS_PATH), map_location=DEVICE))
    except FileNotFoundError:
        print(f"ERROR: VAE Encoder weights not found. Cannot proceed with evaluation.")
        return
    encoder.eval()

    unet = ConditionalUNet(**MODEL_CONFIG['UNET_MODEL']).to(DEVICE)
    try:
        unet.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, UNET_WEIGHTS_PATH), map_location=DEVICE))
    except FileNotFoundError:
        print(f"ERROR: Diffusion UNet weights not found. Did training complete?")
        return
    unet.eval()

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # FIX 2: Set inference timesteps (50 is standard for fast, high-quality generation)
    num_inference_steps = 50
    scheduler.set_timesteps(num_inference_steps)
    
    # 2. Data Loading (Use a test split)
    test_loader = get_dataloader(batch_size=TRAIN_CONFIG['batch_size'], split='train[80%:]', streaming=False) 

    all_metrics = {'PSNR': [], 'SSIM': [], 'LPIPS': []}
    
    # 3. Inference and Evaluation Loop
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating Model")
        
        for batch in pbar:
            # Inputs
            gt_img = batch['gt_image'].to(DEVICE)
            cloudy_img = batch['cloudy_image'].to(DEVICE)
            sar_img = batch['sar_image'].to(DEVICE)

            # a. VAE Encoding (Condition)
            cloudy_latent, _ = encoder(cloudy_img)
            sar_latent, _ = encoder(sar_img)
            cond_latent = torch.cat([cloudy_latent, sar_latent], dim=1) 
            
            # FIX 1: Get the correct latent shape using the gt_img, then make noise
            gt_latent, _ = encoder(gt_img)
            latent = torch.randn_like(gt_latent).to(DEVICE) 
            
            # Run the full denoising loop
            for t in scheduler.timesteps:
                noise_pred = unet(latent, t, cond_latent)
                latent = scheduler.step(noise_pred, t, latent).prev_sample
                
            # FIX 3: Decode the tiny latent back into a full 3x256x256 image
            # Note: Assuming your VAEEncoder class has a 'decode' method. 
            # If your decoder is a separate class, you will need to load it here!
            # NEW
            recon_img = encoder.decode(latent) 
            
            # c. Calculate Metrics
            metrics = calculate_metrics(recon_img, gt_img)
            
            all_metrics['PSNR'].append(metrics['PSNR'])
            all_metrics['SSIM'].append(metrics['SSIM'])
            all_metrics['LPIPS'].append(metrics['LPIPS'])

            pbar.set_postfix(PSNR=f"{metrics['PSNR']:.2f}", LPIPS=f"{metrics['LPIPS']:.4f}")

    # 4. Final Aggregation
    final_results = {key: np.mean(values) for key, values in all_metrics.items()}
    
    # 5. Output Results Directly to Console 
    print("\n--- Project Implementation Complete ---")
    print("--- Final Evaluation Results (Novel DMDiff) ---\n")
    print(f"PSNR (Higher is better): {final_results['PSNR']:.4f}")
    print(f"SSIM (Closer to 1 is better): {final_results['SSIM']:.4f}")
    print(f"LPIPS (Lower is better): {final_results['LPIPS']:.4f}")
    print("\nACTION: Copy the output above and manually paste it into the file: outputs/evaluation_results.txt")


if __name__ == '__main__':
    sample_and_evaluate()