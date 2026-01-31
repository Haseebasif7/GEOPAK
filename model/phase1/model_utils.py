"""
Utility functions for Phase 1 model initialization and loading
"""

import torch
from pathlib import Path
import pandas as pd
from typing import Optional, Dict

from .geopak_phase1 import GeopakPhase1Model
from .utils import compute_province_weights, load_cell_metadata


def create_phase1_model(
    cell_metadata_path: Path,
    fusion_dim: int = 512,
    hidden_dim: int = 256,
    cell_embed_dim: int = 96,
    province_embed_dim: int = 32,
    freeze_clip: bool = True,
    freeze_scene: bool = True,
    scene_model_path: Optional[Path] = None,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
    phase0_checkpoint_path: Optional[Path] = None
) -> GeopakPhase1Model:
    """
    Create and initialize Phase 1 model
    
    Args:
        cell_metadata_path: Path to cell_metadata.csv
        fusion_dim: Fusion dimension (default: 512)
        hidden_dim: Hidden dimension for heads (default: 256)
        cell_embed_dim: Cell embedding dimension (default: 96)
        province_embed_dim: Province embedding dimension (default: 32)
        freeze_clip: Whether to freeze CLIP encoder (default: True for Phase 1)
        freeze_scene: Whether to freeze Scene encoder (default: True for Phase 1)
        scene_model_path: Path to Places365 ResNet-50 checkpoint (optional)
        temperature: Temperature for province head (default: 1.0)
        device: Device to move model to (optional)
        phase0_checkpoint_path: Path to Phase 0 checkpoint (province_best.pt) to load weights from
    
    Returns:
        Initialized GeopakPhase1Model with Phase 0 weights loaded if provided
    """
    model = GeopakPhase1Model(
        cell_metadata_path=cell_metadata_path,
        fusion_dim=fusion_dim,
        hidden_dim=hidden_dim,
        cell_embed_dim=cell_embed_dim,
        province_embed_dim=province_embed_dim,
        freeze_clip=freeze_clip,
        freeze_scene=freeze_scene,
        scene_model_path=str(scene_model_path) if scene_model_path else None,
        temperature=temperature
    )
    
    if device is not None:
        model = model.to(device)
    
    # Load Phase 0 weights if checkpoint provided
    if phase0_checkpoint_path is not None:
        phase0_checkpoint_path = Path(phase0_checkpoint_path)
        if phase0_checkpoint_path.exists():
            print(f"📂 Loading Phase 0 weights from: {phase0_checkpoint_path}")
            checkpoint = torch.load(phase0_checkpoint_path, map_location=device)
            phase0_state = checkpoint['model_state_dict']
            
            # Load encoder weights (CLIP, Scene, fusion, projections)
            # Load encoder weights (CLIP, Scene, fusion, projections)
            encoder_state = {}
            for key, value in phase0_state.items():
                if key.startswith('encoder.'):
                    # Remove 'encoder.' prefix to match Phase 1 encoder structure
                    new_key = key[len('encoder.'):]
                    encoder_state[new_key] = value
            
            # Load encoder state (strict=False to allow for any missing keys)
            missing_keys, unexpected_keys = model.encoder.load_state_dict(encoder_state, strict=False)
            if missing_keys:
                print(f"   ⚠️  Encoder missing keys (expected): {len(missing_keys)}")
            if unexpected_keys:
                print(f"   ⚠️  Encoder unexpected keys: {len(unexpected_keys)}")
            print("   ✅ Loaded encoder weights (CLIP, Scene, fusion, projections)")
            
            # Load province head weights
            # Load province head weights
            # We can load directly into province_head submodule (keys match without prefix)
            province_head_state = {}
            province_head_keys = ['linear.weight', 'linear.bias', 'norm.weight', 'norm.bias', 
                                 'linear_out.weight', 'linear_out.bias']
            for key in province_head_keys:
                if key in phase0_state:
                    province_head_state[key] = phase0_state[key]
            
            # Load directly into province_head submodule (keys match without prefix)
            missing_keys, unexpected_keys = model.province_head.load_state_dict(province_head_state, strict=False)
            if missing_keys:
                print(f"   ⚠️  Province head missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"   ⚠️  Province head unexpected keys: {unexpected_keys}")
            print("   ✅ Loaded province head weights")
            
            # Get temperature from Phase 0 checkpoint if available
            phase0_temp = checkpoint.get('temperature', temperature)
            if phase0_temp != temperature:
                print(f"   ℹ️  Phase 0 temperature: {phase0_temp}, using Phase 1 temperature: {temperature}")
        else:
            print(f"⚠️  Phase 0 checkpoint not found: {phase0_checkpoint_path}, starting from scratch")
    
    return model


def load_phase1_checkpoint(
    checkpoint_path: Path,
    cell_metadata_path: Path,
    device: Optional[torch.device] = None,
    **model_kwargs
) -> tuple[GeopakPhase1Model, Dict]:
    """
    Load Phase 1 model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        cell_metadata_path: Path to cell_metadata.csv
        device: Device to load model to
        **model_kwargs: Additional arguments for model creation
    
    Returns:
        (model, checkpoint_info) tuple
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with same architecture
    model = create_phase1_model(
        cell_metadata_path=cell_metadata_path,
        device=device,
        **model_kwargs
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', None),
        'best_val_error': checkpoint.get('best_val_error', None),
        'loss_weights': checkpoint.get('loss_weights', None),
        'optimizer_state': checkpoint.get('optimizer_state_dict', None),
    }
    
    return model, checkpoint_info


def get_trainable_parameters(model: GeopakPhase1Model) -> Dict[str, int]:
    """
    """Get count of trainable parameters by component"""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    components = {
        'encoder_fusion': sum(p.numel() for p in model.encoder.fusion.parameters() if p.requires_grad),
        'encoder_projections': (
            sum(p.numel() for p in model.encoder.clip_projection.parameters() if p.requires_grad) +
            sum(p.numel() for p in model.encoder.scene_projection.parameters() if p.requires_grad)
        ),
        'province_head': sum(p.numel() for p in model.province_head.parameters() if p.requires_grad),
        'geocell_heads': sum(p.numel() for head in model.geocell_heads.values() 
                            for p in head.parameters() if p.requires_grad),
        'embeddings': (
            sum(p.numel() for p in model.cell_embedding.parameters() if p.requires_grad) +
            sum(p.numel() for p in model.province_embedding.parameters() if p.requires_grad)
        ),
        'offset_head': sum(p.numel() for p in model.offset_head.parameters() if p.requires_grad),
        'aux_head': sum(p.numel() for p in model.aux_head.parameters() if p.requires_grad),
    }
    
    components['total'] = total
    
    return components
