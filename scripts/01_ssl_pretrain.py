# novel implementation/scripts/01_ssl_pretrain.py (FINAL AMP OPTIMIZED)

import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
import sys
import yaml

# --- PATH & IMPORT SETUP ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.data.dataset import get_dataloader 
from src.utils.losses import InfoNCELoss 
from src.models.vae_encoder import VAEEncoder
from src.models.projection_head import ProjectionHead

# --- CONFIGURATION LOADING ---
with open(os.path.join(PROJECT_ROOT, 'configs', 'train.yaml'), 'r') as f:
    TRAIN_CONFIG = yaml.safe_load(f)['SSL_PRETRAIN']
with open(os.path.join(PROJECT_ROOT, 'configs', 'model.yaml'), 'r') as f:
    MODEL_CONFIG = yaml.safe_load(f)

# --- EXECUTION CONFIGURATION ---
MODEL_SAVE_PATH = 'data/models/ssl_encoder_weights.pth'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------

def ssl_pretrain_encoder():
    print(f"Starting AMP-Accelerated SSL Pre-training on {DEVICE}...")

    # 1. Initialize Components
    encoder = VAEEncoder(**MODEL_CONFIG['VAE_ENCODER']).to(DEVICE)
    proj_config = MODEL_CONFIG['PROJECTION_HEAD']
    projection_head = ProjectionHead(**proj_config).to(DEVICE)
    loss_fn = InfoNCELoss(temperature=TRAIN_CONFIG['temp_nce']).to(DEVICE)
    
    optimizer = optim.AdamW(list(encoder.parameters()) + list(projection_head.parameters()), 
                            lr=TRAIN_CONFIG['learning_rate'])

    # Initialize the GradScaler for AMP
    scaler = GradScaler(enabled=(DEVICE == 'cuda'), device=DEVICE)

    # 2. Data Loading (70% Train, 10% Val)
    train_loader = get_dataloader(TRAIN_CONFIG['batch_size'], split='train[:70%]', streaming=False) 
    val_loader = get_dataloader(TRAIN_CONFIG['batch_size'], split='train[70%:80%]', streaming=False) 

    # 3. Training Loop
    encoder.train()
    projection_head.train()
    
    for epoch in range(TRAIN_CONFIG['epochs']):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"SSL Epoch {epoch+1}/{TRAIN_CONFIG['epochs']}")
        
        for batch in pbar:
            sar_img = batch['sar_image'].to(DEVICE)
            gt_img = batch['gt_image'].to(DEVICE)
            
            optimizer.zero_grad()

            # Autocast context manager for mixed precision
            with autocast(device_type=DEVICE, enabled=(DEVICE == 'cuda')):
                mu_sar, _ = encoder(sar_img)
                mu_gt, _ = encoder(gt_img)

                embedding_sar = projection_head(mu_sar)
                embedding_gt = projection_head(mu_gt)

                loss = loss_fn(embedding_sar, embedding_gt)
            
            # Scaled Backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Avg SSL Train Loss: {avg_loss:.4f}")

        # Validation (Now with a visual progress bar!)
        encoder.eval()
        projection_head.eval()
        val_loss = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{TRAIN_CONFIG['epochs']}")
            for batch in val_pbar:
                sar_img = batch['sar_image'].to(DEVICE)
                gt_img = batch['gt_image'].to(DEVICE)
                
                # Autocast for validation to speed it up
                with autocast(device_type=DEVICE, enabled=(DEVICE == 'cuda')):
                    mu_sar, _ = encoder(sar_img)
                    mu_gt, _ = encoder(gt_img)
                    
                    embedding_sar = projection_head(mu_sar)
                    embedding_gt = projection_head(mu_gt)
                    
                    loss = loss_fn(embedding_sar, embedding_gt)
                    
                val_loss += loss.item()
                val_pbar.set_postfix(val_loss=f"{loss.item():.4f}")
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}\n")

        encoder.train()
        projection_head.train()

    # 4. Save Final Encoder Weights
    save_dir = os.path.join(PROJECT_ROOT, 'data', 'models')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(encoder.state_dict(), os.path.join(save_dir, 'ssl_encoder_weights.pth'))
    print(f"\nFinished SSL training. Encoder weights saved to {os.path.join(save_dir, 'ssl_encoder_weights.pth')}")

if __name__ == '__main__':
    ssl_pretrain_encoder()