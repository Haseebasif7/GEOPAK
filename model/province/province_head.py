from .encoder import GeopakModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProvinceHead(nn.Module):

    def __init__(self, fusion_dim=512, hidden_dim=256, freeze_clip=True, freeze_scene=True, scene_model_path=None, temperature=1.0):
        super().__init__()
        self.encoder = GeopakModel(fusion_dim=fusion_dim, freeze_clip=freeze_clip, freeze_scene=freeze_scene, scene_model_path=scene_model_path)
        
        # Ensure fusion gate is trainable even when encoders are frozen
        # This is critical for Phase 0: "Fusion gate learns when scene helps"
        if freeze_clip and freeze_scene:
            # Make fusion gate trainable
            for param in self.encoder.fusion.parameters():
                param.requires_grad = True
            # Make projection layers trainable (they adapt features)
            for param in self.encoder.clip_projection.parameters():
                param.requires_grad = True
            for param in self.encoder.scene_projection.parameters():
                param.requires_grad = True
        
        # Province classification head
        self.linear = nn.Linear(fusion_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.linear_out = nn.Linear(hidden_dim, 7)  # 7 provinces
        
        nn.init.zeros_(self.linear_out.bias)  # Zero bias = no class preference
        nn.init.normal_(self.linear_out.weight, mean=0.0, std=0.01)  # Small weights
        
        self.temperature = 1.2

    def forward(self, images):
        """
        Forward pass
        
        Args:
            images: Tensor [batch_size, 3, 224, 224]
        
        Returns:
            logits: Tensor [batch_size, 7] - Province logits scaled by temperature (no softmax, use CrossEntropyLoss)
        """
        # Get fused features (fusion gate learns during training)
        e_fused = self.encoder(images)  # [batch_size, fusion_dim]
        
        # Province classification
        x = self.linear(e_fused)
        x = self.norm(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear_out(x)  # [batch_size, 7]
        
        x = x / self.temperature
        
        return x

    