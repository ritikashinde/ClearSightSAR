# novel implementation/scripts/04_generate_image.py

import torch
import os
import sys
import yaml
from diffusers import DDPMScheduler
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Add the parent directory to the path for correct imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import get_dataloader 
from src.models.vae_encoder import VAEEncoder
from src.models.unet_model import ConditionalUNet

# --- CONFIGURATION ---
UNET_WEIGHTS_PATH = 'data/models/diffusion_unet_final.pth'
# FIX 1: Use the fully trained VAE (Encoder + Decoder)
ENCODER_WEIGHTS_PATH = 'data/models/vae_autoencoder_final.pth'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 

with open(os.path.join(PROJECT_ROOT, 'configs', 'model.yaml'), 'r') as f:
    MODEL_CONFIG = yaml.safe_load(f)
# ------------------------------------------

def display_generated_image():
    print(f"Starting Image Generation and Visualization on {DEVICE}...")

    # 1. Load Trained Models 
    encoder = VAEEncoder(**MODEL_CONFIG['VAE_ENCODER']).to(DEVICE)
    unet = ConditionalUNet(**MODEL_CONFIG['UNET_MODEL']).to(DEVICE)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # FIX 2: Set inference timesteps for fast generation
    scheduler.set_timesteps(50)
    
    encoder_path_full = os.path.join(PROJECT_ROOT, ENCODER_WEIGHTS_PATH)
    unet_path_full = os.path.join(PROJECT_ROOT, UNET_WEIGHTS_PATH)
    
    try:
        encoder.load_state_dict(torch.load(encoder_path_full, map_location=DEVICE))
        unet.load_state_dict(torch.load(unet_path_full, map_location=DEVICE))
        print(f"Model weights loaded successfully onto {DEVICE}.")
    except FileNotFoundError:
        print("FATAL ERROR: Weights not found. Ensure files are in data/models/ folder.")
        return

    encoder.eval()
    unet.eval()
    
    # 2. Load Single Test Image
    test_loader = get_dataloader(batch_size=1, split='train[85%:86%]', streaming=False)
    try:
        batch = next(iter(test_loader))
    except StopIteration:
        print("Error: Could not load the specified test sample.")
        return

    gt_img = batch['gt_image'].to(DEVICE)
    cloudy_img = batch['cloudy_image'].to(DEVICE)
    sar_img = batch['sar_image'].to(DEVICE)

    # 3. VAE Encoding (Condition)
    with torch.no_grad():
        cloudy_latent, _ = encoder(cloudy_img)
        sar_latent, _ = encoder(sar_img)
        cond_latent = torch.cat([cloudy_latent, sar_latent], dim=1) 
        
        # FIX 3: Get correct latent shape
        gt_latent, _ = encoder(gt_img)
        
    # 4. Latent Sampling (Reverse Diffusion)
    latent = torch.randn_like(gt_latent).to(DEVICE) 
    
    for t in tqdm(scheduler.timesteps, desc="Denoising Image"):
        with torch.no_grad():
            noise_pred = unet(latent, t, cond_latent)
            latent = scheduler.step(noise_pred, t, latent).prev_sample
            
    # FIX 4: Decode the tiny latent back into a full 256x256 image
    with torch.no_grad():
        recon_img_tensor = encoder.decode(latent)
    
    # 5. Decode and Visualize
    def tensor_to_image(tensor):
        """Converts [-1, 1] tensor to [0, 1] NumPy array for plotting (on CPU)."""
        return ((tensor.cpu().squeeze(0).permute(1, 2, 0) + 1) / 2).clamp(0, 1).numpy()

    recon_img_array = tensor_to_image(recon_img_tensor)
    cloudy_img_array = tensor_to_image(cloudy_img)
    gt_img_array = tensor_to_image(gt_img)
    sar_img_array = tensor_to_image(sar_img)

    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(cloudy_img_array)
    axes[0].set_title("1. Cloudy Optical Input")
    axes[0].axis('off')

    axes[1].imshow(sar_img_array, cmap='gray') # SAR is grayscale
    axes[1].set_title("2. SAR Structural Input")
    axes[1].axis('off')

    axes[2].imshow(recon_img_array)
    axes[2].set_title("3. Generated Cloud-Free Output")
    axes[2].axis('off')

    axes[3].imshow(gt_img_array)
    axes[3].set_title("4. Ground Truth (Target)")
    axes[3].axis('off')

    plt.suptitle("Cloud Removal Results (Novel DMDiff Implementation)", fontsize=16, fontweight='bold')
    
    # Save the plot so you have a physical copy!
    save_path = os.path.join(PROJECT_ROOT, 'outputs', 'final_generation_result.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\nImage saved successfully to: {save_path}")
    
    plt.show()

if __name__ == '__main__':
    display_generated_image()