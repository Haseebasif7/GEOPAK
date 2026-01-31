"""
Utility functions for Phase 2 model initialization and training setup
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from model.phase1.geopak_phase1 import GeopakPhase1Model


def load_phase1_checkpoint(
    checkpoint_path: Path,
    cell_metadata_path: Path,
    scene_model_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Tuple[GeopakPhase1Model, Dict]:
    """
    Load Phase 1 checkpoint for Phase 2 training
    
    Args:
        checkpoint_path: Path to phase1_best.pt
        cell_metadata_path: Path to cell_metadata.csv
        scene_model_path: Path to Places365 ResNet-50 checkpoint
        device: Device to load model to
    
    Returns:
        (model, checkpoint_info) tuple
    """
    print(f"📂 Loading Phase 1 checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model with same architecture as Phase 1
    # Encoders will be frozen initially, we'll unfreeze them after loading
    model = GeopakPhase1Model(
        cell_metadata_path=str(cell_metadata_path),
        freeze_clip=True,  # Will unfreeze top layers later
        freeze_scene=True,  # Will unfreeze top layers later
        scene_model_path=str(scene_model_path) if scene_model_path else None,
        temperature=checkpoint.get('temperature', 1.0)
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if device is not None:
        model = model.to(device)
    
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', None),
        'best_val_error': checkpoint.get('best_val_error', None),
        'loss_weights': checkpoint.get('loss_weights', None),
        'temperature': checkpoint.get('temperature', 1.0),
    }
    
    epoch_str = str(checkpoint_info['epoch']) if checkpoint_info['epoch'] is not None else 'unknown'
    val_error_str = f"{checkpoint_info['best_val_error']:.4f} km" if checkpoint_info['best_val_error'] is not None else 'unknown'
    print(f"   ✅ Loaded Phase 1 checkpoint (epoch {epoch_str}, val error: {val_error_str})")
    
    return model, checkpoint_info


def unfreeze_clip_top_layers(model: GeopakPhase1Model, top_percent: float = 0.3) -> int:
    """
    """Unfreeze top layers of CLIP encoder (ViT-B/16)"""
    
    Args:
        model: GeopakPhase1Model instance
        top_percent: Percentage of top layers to unfreeze (default: 0.3 for 30%)
    
    Returns:
        Number of unfrozen parameters
    """
    clip_model = model.encoder.clip_model
    
    # HuggingFace CLIP: vision_model.encoder.layers (not visual.transformer.resblocks)
    # ViT-B/16 has 12 transformer blocks in vision encoder
    vision_model = clip_model.vision_model
    total_blocks = len(vision_model.encoder.layers)
    num_blocks_to_unfreeze = max(1, int(total_blocks * top_percent))
    start_block = total_blocks - num_blocks_to_unfreeze
    
    print(f"\n🔓 Unfreezing CLIP encoder:")
    print(f"   Total blocks: {total_blocks}")
    print(f"   Unfreezing top {top_percent*100:.0f}%: blocks {start_block}-{total_blocks-1}")
    
    unfrozen_params = 0
    
    # Unfreeze last N transformer blocks
    for i in range(start_block, total_blocks):
        for param in vision_model.encoder.layers[i].parameters():
            param.requires_grad = True
            unfrozen_params += param.numel()
    
    # Also unfreeze the final layer norm and projection if they exist
    if hasattr(vision_model, 'post_layernorm'):
        for param in vision_model.post_layernorm.parameters():
            param.requires_grad = True
            unfrozen_params += param.numel()
    
    # Visual projection (if exists)
    if hasattr(vision_model, 'visual_projection') and vision_model.visual_projection is not None:
        vision_model.visual_projection.requires_grad = True
        unfrozen_params += vision_model.visual_projection.numel()
    
    print(f"   ✅ Unfrozen {unfrozen_params:,} CLIP parameters")
    
    return unfrozen_params


def unfreeze_scene_top_layers(model: GeopakPhase1Model, top_percent: float = 0.2) -> int:
    """
    """Unfreeze top layers of Scene encoder (ResNet-50 Places365)"""
    
    Args:
        model: GeopakPhase1Model instance
        top_percent: Percentage indicator (default: 0.2 for 20%, but we only unfreeze layer4)
    
    Returns:
        Number of unfrozen parameters
    """
    scene_encoder = model.encoder.scene_encoder
    
    print(f"\n🔓 Unfreezing Scene encoder:")
    print(f"   Strategy: Unfreeze layer4 only (top ~20%)")
    
    unfrozen_params = 0
    
    # Scene encoder is a Sequential wrapper around ResNet-50 layers
    # Need to access the layers through the Sequential container
    # The Sequential contains: [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool]
    # We want to unfreeze layer4 (index -2, since avgpool is last)
    
    # Try direct attribute access first (if not wrapped)
    if hasattr(scene_encoder, 'layer4'):
        for param in scene_encoder.layer4.parameters():
            param.requires_grad = True
            unfrozen_params += param.numel()
        print(f"   ✅ Unfrozen layer4 (direct access)")
    else:
        # Access through Sequential container
        # Find layer4 in the Sequential children
        for name, module in scene_encoder.named_children():
            # In Sequential, children are numbered 0, 1, 2, etc.
            # We need to find which one is layer4
            pass
        
        # Alternative: iterate through all children and find layer4 by type/structure
        children_list = list(scene_encoder.children())
        # ResNet structure: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool
        # layer4 should be at index -2 (second to last)
        if len(children_list) >= 2:
            layer4 = children_list[-2]  # Assuming avgpool is last
            for param in layer4.parameters():
                param.requires_grad = True
                unfrozen_params += param.numel()
            print(f"   ✅ Unfrozen layer4 (via Sequential, index -2)")
    
    print(f"   ✅ Unfrozen {unfrozen_params:,} Scene parameters")
    
    return unfrozen_params


def create_phase2_optimizer(
    model: GeopakPhase1Model,
    lr_heads: float = 5e-4,
    lr_embeddings: float = 5e-4,
    lr_fusion: float = 5e-4,
    lr_clip: float = 1e-5,
    lr_scene: float = 5e-6,
    weight_decay: float = 1e-5,
) -> optim.AdamW:
    """
    Create optimizer with different learning rates for different parameter groups
    
    Parameter Groups:
    1. Province + cell + offset heads: 5e-4
    2. Cell & province embeddings: 5e-4
    3. Fusion gate: 5e-4
    4. CLIP (unfrozen layers): 1e-5
    5. Scene (unfrozen layers): 5e-6
    
    Args:
        model: GeopakPhase1Model instance
        lr_heads: Learning rate for prediction heads
        lr_embeddings: Learning rate for embeddings
        lr_fusion: Learning rate for fusion gate
        lr_clip: Learning rate for unfrozen CLIP layers
        lr_scene: Learning rate for unfrozen Scene layers
        weight_decay: Weight decay for all groups
    
    Returns:
        AdamW optimizer with parameter groups
    """
    # Collect parameters for each group
    param_groups = []
    
    # Group 1: Prediction heads (province, geocell, offset, aux)
    head_params = []
    head_params.extend(model.province_head.parameters())
    for head in model.geocell_heads.values():
        head_params.extend(head.parameters())
    head_params.extend(model.offset_head.parameters())
    head_params.extend(model.aux_head.parameters())
    
    param_groups.append({
        'params': head_params,
        'lr': lr_heads,
        'name': 'heads'
    })
    
    # Group 2: Embeddings
    embedding_params = []
    embedding_params.extend(model.cell_embedding.parameters())
    embedding_params.extend(model.province_embedding.parameters())
    
    param_groups.append({
        'params': embedding_params,
        'lr': lr_embeddings,
        'name': 'embeddings'
    })
    
    # Group 3: Fusion gate and projections
    fusion_params = []
    fusion_params.extend(model.encoder.fusion.parameters())
    fusion_params.extend(model.encoder.clip_projection.parameters())
    fusion_params.extend(model.encoder.scene_projection.parameters())
    
    param_groups.append({
        'params': fusion_params,
        'lr': lr_fusion,
        'name': 'fusion'
    })
    
    # Group 4: CLIP unfrozen layers
    clip_params = [p for p in model.encoder.clip_model.parameters() if p.requires_grad]
    
    if clip_params:
        param_groups.append({
            'params': clip_params,
            'lr': lr_clip,
            'name': 'clip'
        })
    
    # Group 5: Scene unfrozen layers
    scene_params = [p for p in model.encoder.scene_encoder.parameters() if p.requires_grad]
    
    if scene_params:
        param_groups.append({
            'params': scene_params,
            'lr': lr_scene,
            'name': 'scene'
        })
    
    # Create optimizer
    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    
    # Print parameter group info
    print(f"\n⚙️  Optimizer parameter groups:")
    for i, group in enumerate(optimizer.param_groups):
        num_params = sum(p.numel() for p in group['params'])
        print(f"   {i+1}. {group['name']:12s}: {num_params:10,} params, lr={group['lr']:.2e}")
    
    return optimizer


def get_trainable_parameters(model: GeopakPhase1Model) -> Dict[str, int]:
    """
    Get count of trainable parameters by component
    
    Returns:
        Dict with parameter counts
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    components = {
        'clip_encoder': sum(p.numel() for p in model.encoder.clip_model.parameters() if p.requires_grad),
        'scene_encoder': sum(p.numel() for p in model.encoder.scene_encoder.parameters() if p.requires_grad),
        'fusion': sum(p.numel() for p in model.encoder.fusion.parameters() if p.requires_grad),
        'projections': (
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


    print(f"   {'TOTAL':<20s} {params['total']:>15,}")
