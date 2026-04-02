# novel implementation/src/utils/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights 
import os 
import sys 

# ----------------------------------------
# 1. InfoNCE Loss (Novelty 1: SSL Pre-training)
# ----------------------------------------

class InfoNCELoss(nn.Module):
    """
    Implements InfoNCE Loss for contrastive learning between aligned SAR and Optical embeddings.
    Maximizes similarity of positive pairs (SAR[i] vs GT[i]) over negative pairs (GT[j] where j != i).
    """
    def __init__(self, temperature=0.07, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, query, positive_key):
        """
        Args:
            query (torch.Tensor): [B, D] (e.g., SAR embeddings)
            positive_key (torch.Tensor): [B, D] (e.g., GT Optical embeddings)
        """
        # 1. Normalize embeddings for Cosine Similarity
        query = F.normalize(query, dim=1)
        positive_key = F.normalize(positive_key, dim=1)

        # 2. Compute similarity matrix [B, B]
        # Diagonal elements are positive pairs; off-diagonals are negative pairs
        logits = query @ positive_key.T
        
        # 3. Apply temperature scaling
        logits = logits / self.temperature

        # 4. Create labels for CrossEntropyLoss
        # Target for row 'i' is column 'i' (the positive pair)
        labels = torch.arange(len(query), device=query.device) 

        # 5. Compute Cross-Entropy loss (the core InfoNCE loss)
        loss = F.cross_entropy(logits, labels, reduction=self.reduction)
        return loss

# ----------------------------------------
# 2. Perceptual Loss (Novelty 3: Image Quality)
# ----------------------------------------

class PerceptualLoss(nn.Module):
    """
    VGG-based Perceptual Loss (Feature Reconstruction Loss).
    Uses a standard VGG16 feature extractor.
    """
    def __init__(self, layer_idx=9): # Targets the relu2_2 features (standard choice)
        super().__init__()
        
        # Determine device
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Load VGG16 pre-trained on ImageNet using standard torchvision weights
        try:
            # We explicitly load weights to ensure torchvision does the work, bypassing external defaults
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(DEVICE)
        except Exception as e:
            print("FATAL ERROR: VGG16 download failed. This might be related to network/proxy issues.")
            print("Please ensure you can access standard torchvision model weights.")
            raise e
            
        # Freeze VGG parameters (fixed feature extractor)
        for param in vgg.parameters():
            param.requires_grad = False
            
        # Select layers up to the desired feature map (layer_idx=9 for relu2_2)
        self.vgg_features = nn.Sequential(*list(vgg.children())[:layer_idx])
        self.criterion = nn.L1Loss() # L1 loss on the feature maps

    def forward(self, pred_image, target_image):
        """Calculates L1 distance between VGG feature maps."""
        
        # VGG requires input to be scaled appropriately (usually [0, 1])
        # Rescale model output from [-1, 1] -> [0, 1]
        pred_image_scaled = (pred_image + 1) / 2
        target_image_scaled = (target_image + 1) / 2
        
        # Extract features
        pred_features = self.vgg_features(pred_image_scaled)
        target_features = self.vgg_features(target_image_scaled)
        
        # Calculate L1 distance (feature reconstruction loss)
        loss = self.criterion(pred_features, target_features)
        return loss