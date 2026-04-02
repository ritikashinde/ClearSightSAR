# novel implementation/scripts/02_train_novel_dm.py

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import os
import sys
import yaml
from diffusers import DDPMScheduler
import matplotlib.pyplot as plt
import numpy as np

# --- PATH & IMPORT SETUP ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.data.dataset import get_dataloader 
from src.utils.losses import PerceptualLoss 
from src.models.vae_encoder import VAEEncoder
from src.models.unet_model import ConditionalUNet

# --- CONFIGURATION LOADING ---
with open(os.path.join(PROJECT_ROOT, 'configs', 'train.yaml'), 'r') as f:
    TRAIN_CONFIG = yaml.safe_load(f)['DIFFUSION_TRAIN']
with open(os.path.join(PROJECT_ROOT, 'configs', 'model.yaml'), 'r') as f:
    MODEL_CONFIG = yaml.safe_load(f)

# --- EXECUTION CONFIGURATION ---
ENCODER_WEIGHT_PATH = 'data/models/ssl_encoder_weights.pth'
UNET_SAVE_FILENAME = 'diffusion_unet_final.pth' 
UNET_SAVE_DIR = 'data/models' 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------

# --- UTILITIES: EARLY STOPPING & PLOTTING ---
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss, model, path):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), path)
            print(f"   -> Validation loss improved! Saving best weights to {path}")
        else:
            self.counter += 1
            print(f"   -> EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

def plot_losses(train_losses, val_losses, save_path="outputs/dm_loss_curve.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Diffusion Model: Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"\nLoss curve saved successfully to: {save_path}")
# --------------------------------------------

def train_novel_diffusion_model():
    print(f"Starting STABLE Diffusion Training on {DEVICE}...")

    # 1. Initialize Components
    encoder = VAEEncoder(**MODEL_CONFIG['VAE_ENCODER']).to(DEVICE)
    encoder_path_full = os.path.join(PROJECT_ROOT, ENCODER_WEIGHT_PATH)
    
    try:
        encoder.load_state_dict(torch.load(encoder_path_full, map_location=DEVICE))
        print("Successfully loaded SSL pre-trained encoder weights.")
    except FileNotFoundError:
        print("ERROR: Pre-trained weights not found. You must run 01_ssl_pretrain.py first.")
        return
        
    encoder.eval() 
    
    unet = ConditionalUNet(**MODEL_CONFIG['UNET_MODEL']).to(DEVICE) 
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    save_dir_full = os.path.join(PROJECT_ROOT, UNET_SAVE_DIR)
    os.makedirs(save_dir_full, exist_ok=True)
    unet_save_path_full = os.path.join(save_dir_full, UNET_SAVE_FILENAME)

    if os.path.exists(unet_save_path_full):
        try:
            unet.load_state_dict(torch.load(unet_save_path_full, map_location=DEVICE))
            print(f"RESUME SUCCESS: Loaded existing UNet checkpoint.")
        except Exception as e:
            print(f"Warning: Failed to load UNet checkpoint ({e}). Starting fresh.")
            
    loss_simple = nn.MSELoss() 
    optimizer = AdamW(unet.parameters(), lr=TRAIN_CONFIG['learning_rate'])
    
    # Data Loading 
    train_loader = get_dataloader(TRAIN_CONFIG['batch_size'], split='train[:70%]', streaming=False) 
    val_loader = get_dataloader(TRAIN_CONFIG['batch_size'], split='train[70%:80%]', streaming=False) 

    # --- TRACKERS ---
    train_history = []
    val_history = []
    early_stopping = EarlyStopping(patience=5)

    # 2. Training Loop 
    for epoch in range(TRAIN_CONFIG['epochs']): 
        unet.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Novel DM Epoch {epoch+1}/{TRAIN_CONFIG['epochs']}")
        
        for batch in pbar:
            gt_img = batch['gt_image'].to(DEVICE)
            cloudy_img = batch['cloudy_image'].to(DEVICE)
            sar_img = batch['sar_image'].to(DEVICE)
            
            optimizer.zero_grad()

            with torch.no_grad():
                gt_latent, _ = encoder(gt_img)
                cloudy_latent, _ = encoder(cloudy_img)
                sar_latent, _ = encoder(sar_img)

                cond_latent = torch.cat([cloudy_latent, sar_latent], dim=1) 
            
            noise = torch.randn_like(gt_latent)
            # FIX: Dynamic batch size for final batch mismatch
            actual_batch_size = gt_latent.shape[0]
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (actual_batch_size,), device=DEVICE).long()
            
            noisy_latent = scheduler.add_noise(gt_latent, noise, timesteps)
            noise_pred = unet(noisy_latent, timesteps, cond_latent)
            loss = loss_simple(noise_pred, noise)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(mse_loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Avg Train Loss: {avg_train_loss:.4f}")

        # Validation
        unet.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                gt_img = batch['gt_image'].to(DEVICE)
                cloudy_img = batch['cloudy_image'].to(DEVICE)
                sar_img = batch['sar_image'].to(DEVICE)
                
                gt_latent, _ = encoder(gt_img)
                cloudy_latent, _ = encoder(cloudy_img)
                sar_latent, _ = encoder(sar_img)
                cond_latent = torch.cat([cloudy_latent, sar_latent], dim=1)
                
                noise = torch.randn_like(gt_latent)
                # FIX: Dynamic batch size for validation mismatch
                actual_batch_size = gt_latent.shape[0]
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (actual_batch_size,), device=DEVICE).long()
                
                # REPAIRED: Actual validation math
                noisy_latent = scheduler.add_noise(gt_latent, noise, timesteps)
                noise_pred = unet(noisy_latent, timesteps, cond_latent)
                loss = loss_simple(noise_pred, noise)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Update Trackers & Check Early Stopping
        train_history.append(avg_train_loss)
        val_history.append(avg_val_loss)
        
        early_stopping(avg_val_loss, unet, path=unet_save_path_full)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch+1}! Model is starting to overfit.")
            break

    # 3. Plot the final training curves
    plot_path = os.path.join(PROJECT_ROOT, 'outputs', 'dm_loss_curve.png')
    plot_losses(train_history, val_history, save_path=plot_path)

if __name__ == '__main__':
    train_novel_diffusion_model()