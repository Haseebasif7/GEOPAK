"""
GEOPAK-V3 Model Architecture
Dual-Encoder with CLIP + Scene Encoder and Gated Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
import timm
import torchvision.models as models
from torchvision import transforms


class ProjectionLayer(nn.Module):
    """Projection layer to normalize encoder outputs to compatible space"""
    def __init__(self, in_dim, out_dim=512):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = F.gelu(x)
        return x


class GatedFusion(nn.Module):
    """Gated fusion to combine CLIP and Scene encoder features"""
    def __init__(self, dim=512):
        super().__init__()
        self.gate = nn.Linear(dim * 2, 1)  # Takes concatenated features
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.gate.bias, -0.4)
        
    def forward(self, e_clip, e_scene):
        """
        Args:
            e_clip: CLIP features [batch_size, dim]
            e_scene: Scene encoder features [batch_size, dim]
        Returns:
            e_fused: Fused features [batch_size, dim]
        """
        # Concatenate features
        concat = torch.cat([e_clip, e_scene], dim=1)  # [batch_size, dim*2]
        
        # Compute gate weight (alpha)
        alpha = torch.sigmoid(self.gate(concat))  # [batch_size, 1]
        
        # Gated fusion: α · E_scene + (1 − α) · E_clip
        e_fused = alpha * e_scene + (1 - alpha) * e_clip
        
        return e_fused


class GeopakModel(nn.Module):
    """
    GEOPAK-V3 Model: Dual-Encoder Architecture
    
    - Encoder A: CLIP (ViT-B/16, CLIP-pretrained) - Primary, Robust
    - Encoder B: Scene Encoder (Places365-pretrained) - Specialist
    - Gated Fusion to combine features
    """
    
    def __init__(
        self,
        fusion_dim=512,
        freeze_clip=True,
        freeze_scene=True,
        scene_model_path=None
    ):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        
        # ============================================================
        # Encoder A: CLIP (ViT-B/16)
        # ============================================================
        print("Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", use_fast=True)
        
        clip_dim = self.clip_model.config.vision_config.hidden_size  # 768 for ViT-B/16
        
        # Freeze CLIP if requested
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
            print("CLIP encoder: FROZEN")
        else:
            print("CLIP encoder: TRAINABLE")
        
        # ============================================================
        # Encoder B: Scene Encoder - ResNet-50 (ImageNet-pretrained
        # ============================================================
        print("Loading Scene Encoder: ResNet-50...")
        
        if scene_model_path is not None:
            scene_encoder = models.resnet50(num_classes=365)
            checkpoint = torch.load(scene_model_path, map_location='cpu')
        else:
            scene_encoder = models.resnet50(num_classes=365)
            # Path from model/province/ to project root
            checkpoint = torch.load("../../resnet50_places365.pth.tar", map_location='cpu')

        state_dict = {k.replace('module.',''): v for k, v in checkpoint['state_dict'].items()}
        scene_encoder.load_state_dict(state_dict)
        scene_encoder = torch.nn.Sequential(*list(scene_encoder.children())[:-1])
        self.scene_encoder = scene_encoder  # Assign to self
        scene_dim = 2048
        
        if freeze_scene:
            for param in self.scene_encoder.parameters():
                param.requires_grad = False
            print("Scene encoder (ResNet-50): FROZEN")
        else:
            print("Scene encoder (ResNet-50): TRAINABLE")
        
        # ============================================================
        # Projection Layers (CRITICAL: Normalize to compatible space)
        # ============================================================
        self.clip_projection = ProjectionLayer(clip_dim, fusion_dim)
        self.scene_projection = ProjectionLayer(scene_dim, fusion_dim)
        
        # ============================================================
        # Gated Fusion
        # ============================================================
        self.fusion = GatedFusion(dim=fusion_dim)
        
        print(f"Model initialized:")
        print(f"  CLIP dim: {clip_dim} → {fusion_dim}")
        print(f"  Scene dim: {scene_dim} → {fusion_dim}")
        print(f"  Fusion dim: {fusion_dim}")
    
    def forward(self, images):
        """
        Forward pass
        
        Args:
            images: Tensor [batch_size, 3, 224, 224] (unnormalized, range [0,1])
        
        Returns:
            e_fused: Fused features [batch_size, fusion_dim]
        """
        batch_size = images.shape[0]
        
        # CLIP normalization: mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(images.device)
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(images.device)
        clip_images = (images - clip_mean) / clip_std

        
        clip_outputs = self.clip_model.vision_model(pixel_values=clip_images)
        clip_features = clip_outputs.last_hidden_state[:, 0, :]  # CLS token
        e_clip = self.clip_projection(clip_features)
        
        # ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
        resnet_images = (images - imagenet_mean) / imagenet_std

        scene_features = self.scene_encoder(resnet_images)
        # Pooling is already included in the Sequential model (AdaptiveAvgPool2d)
        scene_features = scene_features.view(batch_size, -1)
        e_scene = self.scene_projection(scene_features)
        
        # ============================================================
        # Gated Fusion
        # ============================================================
        e_fused = self.fusion(e_clip, e_scene)
        
        return e_fused
    