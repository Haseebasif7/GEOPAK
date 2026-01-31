"""
"""Training script for Phase 2 - Partial Vision Adaptation"""
import sys
import argparse
from pathlib import Path
import json

# Add project root to Python path
# Handle both local execution and Modal execution
modal_project_root = Path("/root/geopak")
if modal_project_root.exists():
    if str(modal_project_root) not in sys.path:
        sys.path.insert(0, str(modal_project_root))

# Determine project root for local execution
local_project_root = Path(__file__).parent.parent.parent

# Set project_root variable (used later in code)
if modal_project_root.exists() and (modal_project_root / "model").exists():
    # Running on Modal - code is at /root/geopak
    project_root = modal_project_root
else:
    # Running locally
    project_root = local_project_root
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# Modal imports (optional - only needed when using Modal)
try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    modal = None

# Import Phase 1 components (reuse everything except optimizer setup)
from pipeline.data_streaming.geo_dataset import GeopakDataset
from pipeline.data_streaming.transforms import get_train_transforms, get_val_test_transforms
from pipeline.data_streaming.balanced_sampler import ProvinceBalancedBatchSampler, DEFAULT_BATCH_SPLIT
from model.phase1.geopak_phase1 import GeopakPhase1Model
from model.phase1.losses import Phase1TotalLoss
from model.phase1.utils import haversine_km
from model.province.train_province import calculate_class_weights

# Import Phase 1 training functions (reuse train_epoch, validate, etc.)
from model.phase1.train_phase1 import (
    collate_fn,
    load_province_mapping,
    get_device,
    train_epoch,
    validate,
    print_province_metrics,
    save_checkpoint
)

# Import Phase 2 utilities
from model.phase2.model_utils import (
    load_phase1_checkpoint,
    unfreeze_clip_top_layers,
    unfreeze_scene_top_layers,
    create_phase2_optimizer,
    print_trainable_parameters
)


def train_main(
    phase1_checkpoint_path=None,
    train_csv_path=None,
    val_csv_path=None,
    checkpoint_dir_path=None,
    cell_metadata_path=None,
    batch_size=64,
    num_epochs=35,
    lr_heads=5e-4,
    lr_embeddings=5e-4,
    lr_fusion=5e-4,
    lr_clip=1e-5,
    lr_scene=5e-6,
    weight_decay=1e-5,
    beta=0.999,
    use_lr_schedule=True,
    schedule_type="cosine",  # "cosine" or "step"
    num_workers=4,
    pin_memory=False,
):
    """
    Main training function for Phase 2
    
    Args:
        phase1_checkpoint_path: Path to phase1_best.pt
        train_csv_path: Path to train.csv
        val_csv_path: Path to test.csv
        checkpoint_dir_path: Path to checkpoint directory
        cell_metadata_path: Path to cell_metadata.csv
        batch_size: Batch size (64, same as Phase 1)
        num_epochs: Number of training epochs (30-40 for Phase 2)
        lr_heads: Learning rate for prediction heads (5e-4)
        lr_embeddings: Learning rate for embeddings (5e-4)
        lr_fusion: Learning rate for fusion gate (5e-4)
        lr_clip: Learning rate for unfrozen CLIP layers (1e-5)
        lr_scene: Learning rate for unfrozen Scene layers (5e-6)
        weight_decay: Weight decay (1e-5)
        beta: Effective-number weighting parameter
        use_lr_schedule: Whether to use learning rate scheduling
        schedule_type: "cosine" or "step" (drop ×0.3 at epoch 20)
        num_workers: Number of data loader workers
        pin_memory: Enable pin_memory for faster GPU transfer
    """
    # Configuration
    if phase1_checkpoint_path is None:
        phase1_checkpoint = project_root / 'checkpoints' / 'phase1' / 'phase1_best.pt'
    else:
        phase1_checkpoint = Path(phase1_checkpoint_path)
    
    if not phase1_checkpoint.exists():
        raise FileNotFoundError(f"Phase 1 checkpoint not found: {phase1_checkpoint}")
    
    if train_csv_path is None:
        train_csv = project_root / 'train.csv'
    else:
        train_csv = Path(train_csv_path)
    
    if val_csv_path is None:
        val_csv = project_root / 'test.csv'
    else:
        val_csv = Path(val_csv_path)
    
    if checkpoint_dir_path is None:
        checkpoint_dir = project_root / 'checkpoints' / 'phase2'
    else:
        checkpoint_dir = Path(checkpoint_dir_path)
    
    if cell_metadata_path is None:
        cell_metadata = project_root / 'pipeline' / 'geocells' / 'cell_metadata.csv'
    else:
        cell_metadata = Path(cell_metadata_path)
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Open log file for writing
    log_file_path = checkpoint_dir / 'log.txt'
    log_file = open(log_file_path, 'w')
    
    # Open CSV file for structured loss logging
    csv_log_path = checkpoint_dir / 'losses.csv'
    csv_log_file = open(csv_log_path, 'w')
    csv_log_file.write("epoch,split,loss_total,loss_province,loss_geocell,loss_offset,loss_aux,median_error_km,mean_error_km,p90_error_km,")
    csv_log_file.write("sindh_acc,punjab_acc,kpk_acc,ict_acc,gb_acc,balochistan_acc,ajk_acc\n")
    csv_log_file.flush()
    
    log_file.write("GEOPAK PHASE 2 TRAINING - Partial Vision Adaptation\n")
    log_file.flush()
    
    # Device
    device = get_device()
    device_msg = f"\n🖥️  Device: {device}\n"
    print(device_msg)
    log_file.write(device_msg)
    
    # Find scene model path
    scene_model_path = None
    possible_paths = [
        project_root / "resnet50_places365.pth.tar",
        project_root.parent / "resnet50_places365.pth.tar",
        Path("/data/resnet50_places365.pth.tar"),  # Modal volume
        Path("resnet50_places365.pth.tar"),
    ]
    
    for path in possible_paths:
        if path.exists():
            scene_model_path = path
            print(f"   Found scene model: {scene_model_path}")
            log_file.write(f"   Found scene model: {scene_model_path}\n")
            break
            
    if scene_model_path is None:
        print("⚠️  Scene model not found! ResNet-50 initialization might fail.")
        log_file.write("⚠️  Scene model not found! ResNet-50 initialization might fail.\n")
    
    # Load Phase 1 checkpoint
    print(f"\n📂 Loading Phase 1 checkpoint...")
    log_file.write(f"\n📂 Loading Phase 1 checkpoint...\n")
    model, checkpoint_info = load_phase1_checkpoint(
        checkpoint_path=phase1_checkpoint,
        cell_metadata_path=cell_metadata,
        scene_model_path=scene_model_path,
        device=device
    )
    
    epoch_str = str(checkpoint_info['epoch']) if checkpoint_info['epoch'] is not None else 'unknown'
    val_error_str = f"{checkpoint_info['best_val_error']:.4f} km" if checkpoint_info['best_val_error'] is not None else 'unknown'
    
    phase1_msg = f"   Phase 1 epoch: {epoch_str}\n"
    phase1_msg += f"   Phase 1 val error: {val_error_str}\n"
    phase1_msg += f"   Temperature: {checkpoint_info['temperature']}\n"
    print(phase1_msg)
    log_file.write(phase1_msg)
    log_file.flush()
    
    # Unfreeze encoder layers
    print(f"\n🔓 Unfreezing encoder layers...")
    log_file.write(f"\n🔓 Unfreezing encoder layers...\n")
    
    clip_params = unfreeze_clip_top_layers(model, top_percent=0.3)
    scene_params = unfreeze_scene_top_layers(model, top_percent=0.2)
    
    unfreeze_msg = f"   Total unfrozen encoder params: {clip_params + scene_params:,}\n"
    print(unfreeze_msg)
    log_file.write(unfreeze_msg)
    log_file.flush()
    
    # Print trainable parameters
    print_trainable_parameters(model)
    
    # Load datasets
    print(f"\n📦 Loading datasets...")
    log_file.write(f"\n📦 Loading datasets...\n")
    train_transform = get_train_transforms()
    val_transform = get_val_test_transforms()
    
    train_dataset = GeopakDataset(
        csv_path=str(train_csv),
        transform=train_transform
    )
 
    val_dataset = GeopakDataset(
        csv_path=str(val_csv),
        transform=val_transform
    )
    
    dataset_msg = f"   Train: {len(train_dataset):,} samples\n   Val:   {len(val_dataset):,} samples\n"
    print(dataset_msg)
    log_file.write(dataset_msg)
    
    # Calculate province weights (same as Phase 1)
    print(f"\n⚖️  Calculating province weights...")
    log_file.write(f"\n⚖️  Calculating province weights...\n")
    weight_tensor, province_counts = calculate_class_weights(train_dataset, beta=beta)
    
    province_mapping = load_province_mapping()
    province_weights = {}
    for province, province_id in province_mapping.items():
        province_weights[province] = weight_tensor[province_id].item()
    
    log_file.write("\nProvince Weights:\n")
    for province, province_id in sorted(province_mapping.items(), key=lambda x: x[1]):
        weight = province_weights[province]
        count = province_counts.get(province, 0)
        log_file.write(f"   {province:<25} Count: {count:>6,}, Weight: {weight:>5.2f}\n")
    log_file.flush()
    
    # Create loss function (same as Phase 1)
    print(f"\n📊 Creating loss function...")
    log_file.write(f"\n📊 Creating loss function...\n")
    loss_fn = Phase1TotalLoss(
        province_weights=province_weights,
        cell_metadata=model.cell_metadata_df,
        cell_neighbors=model.cell_neighbors,
        device=device,
        use_offset_rampup=True  # Keep same ramp-up strategy
    )
    
    # Create data loaders (same as Phase 1)
    print(f"\n🔄 Creating data loaders...")
    log_file.write(f"\n🔄 Creating data loaders...\n")
    balanced_sampler = ProvinceBalancedBatchSampler(
        dataset=train_dataset,
        province_batch_split=DEFAULT_BATCH_SPLIT,  # Same sampler as Phase 1
        batches_per_epoch=1300,
        shuffle=True,
        seed=42
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=balanced_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    # Create optimizer with parameter groups
    print(f"\n⚙️  Creating optimizer with parameter groups...")
    log_file.write(f"\n⚙️  Creating optimizer with parameter groups...\n")
    optimizer = create_phase2_optimizer(
        model=model,
        lr_heads=lr_heads,
        lr_embeddings=lr_embeddings,
        lr_fusion=lr_fusion,
        lr_clip=lr_clip,
        lr_scene=lr_scene,
        weight_decay=weight_decay
    )
    
    # Create learning rate scheduler (optional)
    scheduler = None
    if use_lr_schedule:
        if schedule_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs, eta_min=0
            )
            schedule_msg = f"   Using Cosine Annealing LR schedule\n"
        elif schedule_type == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=20, gamma=0.3
            )
            schedule_msg = f"   Using Step LR schedule (drop ×0.3 at epoch 20)\n"
        else:
            schedule_msg = f"   No LR schedule\n"
        
        print(schedule_msg)
        log_file.write(schedule_msg)
        log_file.flush()
    
    # Training loop
    train_start_msg = f"\nStarting Phase 2 training...\n"
    train_start_msg += f"   Epochs: {num_epochs}\n"
    train_start_msg += f"   Learning rates:\n"
    train_start_msg += f"     - Heads: {lr_heads}\n"
    train_start_msg += f"     - Embeddings: {lr_embeddings}\n"
    train_start_msg += f"     - Fusion: {lr_fusion}\n"
    train_start_msg += f"     - CLIP: {lr_clip}\n"
    train_start_msg += f"     - Scene: {lr_scene}\n"
    train_start_msg += f"   Weight decay: {weight_decay}\n"
    print(train_start_msg)
    log_file.write(train_start_msg)
    log_file.flush()
    
    best_val_error = checkpoint_info['best_val_error'] if checkpoint_info['best_val_error'] is not None else float('inf')
    epochs_without_improvement = 0
    
    # Red flag monitoring
    rare_provinces = ["Gilgit-Baltistan", "Balochistan", "Azad Kashmir"]
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_losses, train_prov_acc, train_med_err, train_mean_err, train_p90_err = train_epoch(
            model, train_loader, loss_fn, optimizer, device, epoch
        )
        
        # Validate
        val_losses, val_prov_acc, val_med_err, val_mean_err, val_p90_err = validate(
            model, val_loader, loss_fn, device, epoch
        )
        
        # Step scheduler if using one
        if scheduler is not None:
            scheduler.step()
            current_lrs = [group['lr'] for group in optimizer.param_groups]
            lr_msg = f"   Current LRs: {current_lrs}\n"
            print(lr_msg)
            log_file.write(lr_msg)
        
        # Print metrics
        epoch_summary = f"\nEpoch {epoch}/{num_epochs}:\n"
        epoch_summary += f"   Train Loss: {train_losses['total']:.4f} (prov: {train_losses['province']:.4f}, "
        epoch_summary += f"cell: {train_losses['geocell']:.4f}, off: {train_losses['offset']:.4f}, "
        epoch_summary += f"aux: {train_losses['aux']:.4f})\n"
        epoch_summary += f"   Val Loss:   {val_losses['total']:.4f} (prov: {val_losses['province']:.4f}, "
        epoch_summary += f"cell: {val_losses['geocell']:.4f}, off: {val_losses['offset']:.4f}, "
        epoch_summary += f"aux: {val_losses['aux']:.4f})\n"
        epoch_summary += f"   Train Error: median={train_med_err:.2f}km, mean={train_mean_err:.2f}km, p90={train_p90_err:.2f}km\n"
        epoch_summary += f"   Val Error:   median={val_med_err:.2f}km, mean={val_mean_err:.2f}km, p90={val_p90_err:.2f}km\n"
        
        print(epoch_summary)
        log_file.write(epoch_summary)
        log_file.flush()
        
        print_province_metrics(train_prov_acc, "Train", log_file=log_file)
        print_province_metrics(val_prov_acc, "Val", log_file=log_file)
        
        # Red flag monitoring
        red_flags = []
        
        # Check for rare province collapse
        for prov in rare_provinces:
            if val_prov_acc.get(prov, 0) < 30.0:  # Arbitrary threshold
                red_flags.append(f"⚠️  {prov} accuracy collapsed: {val_prov_acc.get(prov, 0):.1f}%")
        
        # Check for offset loss near zero (cheating via cells)
        if val_losses['offset'] < 0.1:
            red_flags.append(f"⚠️  Offset loss near zero: {val_losses['offset']:.4f} (model may be cheating)")
        
        # Check for increasing validation error
        if val_med_err > best_val_error:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 5:
                red_flags.append(f"⚠️  Val median error increasing for {epochs_without_improvement} epochs")
        else:
            epochs_without_improvement = 0
        
        if red_flags:
            red_flag_msg = "\n🚨 RED FLAGS DETECTED:\n"
            for flag in red_flags:
                red_flag_msg += f"   {flag}\n"
            print(red_flag_msg)
            log_file.write(red_flag_msg)
            log_file.flush()
        
        # Log losses to CSV
        province_order = ["Sindh", "Punjab", "Khyber Pakhtunkhwa", "ICT", 
                         "Gilgit-Baltistan", "Balochistan", "Azad Kashmir"]
        
        train_accs = [train_prov_acc.get(prov, 0.0) for prov in province_order]
        csv_log_file.write(f"{epoch},train,{train_losses['total']:.6f},{train_losses['province']:.6f},")
        csv_log_file.write(f"{train_losses['geocell']:.6f},{train_losses['offset']:.6f},{train_losses['aux']:.6f},")
        csv_log_file.write(f"{train_med_err:.4f},{train_mean_err:.4f},{train_p90_err:.4f},")
        csv_log_file.write(",".join([f"{acc:.4f}" for acc in train_accs]) + "\n")
        
        val_accs = [val_prov_acc.get(prov, 0.0) for prov in province_order]
        csv_log_file.write(f"{epoch},val,{val_losses['total']:.6f},{val_losses['province']:.6f},")
        csv_log_file.write(f"{val_losses['geocell']:.6f},{val_losses['offset']:.6f},{val_losses['aux']:.6f},")
        csv_log_file.write(f"{val_med_err:.4f},{val_mean_err:.4f},{val_p90_err:.4f},")
        csv_log_file.write(",".join([f"{acc:.4f}" for acc in val_accs]) + "\n")
        csv_log_file.flush()
        
        # Save checkpoint for each epoch
        checkpoint_path = checkpoint_dir / f'phase2_epoch_{epoch}.pt'
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': val_losses,
            'province_accuracy': val_prov_acc,
            'median_error_km': val_med_err,
            'best_val_error': best_val_error,
            'temperature': checkpoint_info['temperature'],
            'phase1_checkpoint': str(phase1_checkpoint),
        }
        torch.save(checkpoint_data, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Save best model (based on validation median error)
        if val_med_err < best_val_error:
            improvement = best_val_error - val_med_err
            best_val_error = val_med_err
            best_path = checkpoint_dir / 'phase2_best.pt'
            torch.save(checkpoint_data, best_path)
            best_msg = f"🏆 New best validation median error: {best_val_error:.2f}km (improved by {improvement:.2f}km)\n"
            print(best_msg)
            log_file.write(best_msg)
            log_file.flush()
        
        separator = "=" * 70 + "\n"
        print(separator)
        log_file.write(separator)
        log_file.flush()
    
    completion_msg = "\n✅ Phase 2 training complete!\n"
    print(completion_msg)
    log_file.write(completion_msg)
    log_file.close()
    csv_log_file.close()
    print(f"📊 Losses logged to: {csv_log_path}")


def main():
    """Local training entry point"""
    train_main()


# ============================================================================
# Modal Setup for GPU Training
# ============================================================================

if MODAL_AVAILABLE:
    # Create Modal image with all dependencies
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .pip_install(
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "transformers>=4.30.0",
            "timm>=0.9.0",
            "pillow>=9.0.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "tqdm>=4.65.0",
            "scikit-learn>=1.3.0",
            "geopandas>=0.13.0",
            "shapely>=2.0.0",
            "huggingface-hub>=0.16.0",
        )
        .env({"HF_HOME": "/root/.cache/huggingface"})
        .add_local_dir(project_root / "model", "/root/geopak/model")
        .add_local_dir(project_root / "pipeline", "/root/geopak/pipeline")
    )
    
    # Add scene model file if it exists locally
    scene_model_local = project_root / "resnet50_places365.pth.tar"
    if scene_model_local.exists():
        image = image.add_local_file(scene_model_local, "/root/geopak/resnet50_places365.pth.tar")
    
    # Create Modal app
    app = modal.App("geopak-phase2-training", image=image)
    
    # Create volume for data and checkpoints
    volume = modal.Volume.from_name("geopak-data", create_if_missing=True)
    
    @app.function(
        gpu="L4",
        timeout=86400,  # 24 hours
        volumes={"/data": volume},
    )
    def train_on_modal(
        phase1_checkpoint_path="/data/checkpoints/phase1/phase1_best.pt",
        train_csv_path="/data/train.csv",
        val_csv_path="/data/test.csv",
        checkpoint_dir_path="/data/checkpoints/phase2",
        cell_metadata_path="/data/pipeline/geocells/cell_metadata.csv",
        batch_size=64,
        num_epochs=30,
        lr_heads=5e-4,
        lr_embeddings=5e-4,
        lr_fusion=5e-4,
        lr_clip=1e-5,
        lr_scene=5e-6,
        weight_decay=1e-5,
        beta=0.9995,
        use_lr_schedule=True,
        schedule_type="cosine",
    ):
        """Training function that runs on Modal GPU"""
        import sys
        from pathlib import Path
        
        modal_project_root = Path("/root/geopak")
        sys.path.insert(0, str(modal_project_root))
        
        import model.phase2.train_phase2 as train_module
        train_module.project_root = modal_project_root
        
        train_main(
            phase1_checkpoint_path=phase1_checkpoint_path,
            train_csv_path=train_csv_path,
            val_csv_path=val_csv_path,
            checkpoint_dir_path=checkpoint_dir_path,
            cell_metadata_path=cell_metadata_path,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr_heads=lr_heads,
            lr_embeddings=lr_embeddings,
            lr_fusion=lr_fusion,
            lr_clip=lr_clip,
            lr_scene=lr_scene,
            weight_decay=weight_decay,
            beta=beta,
            use_lr_schedule=use_lr_schedule,
            schedule_type=schedule_type,
            num_workers=4,
            pin_memory=True,
        )
        
        volume.commit()
        print("✅ Volume changes committed")
    
    @app.local_entrypoint()
    def main_modal(
        phase1_checkpoint: str = None,
        train_csv: str = None,
        val_csv: str = None,
        checkpoint_dir: str = None,
        cell_metadata: str = None,
        batch_size: int = 64,
        num_epochs: int = 30,
        lr_heads: float = 5e-4,
        lr_embeddings: float = 5e-4,
        lr_fusion: float = 5e-4,
        lr_clip: float = 1e-5,
        lr_scene: float = 5e-6,
        weight_decay: float = 1e-5,
        beta: float = 0.9995,
        use_lr_schedule: bool = True,
        schedule_type: str = "cosine",
        gpu: str = "A100",
        detach: bool = False,
    ):
        """
        Entry point for Modal training
        
        Args:
            detach: If True, spawns the function in detached mode
        """
        # Update GPU type
        train_on_modal.gpu = gpu
        
        # Prepare arguments
        kwargs = {
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'lr_heads': lr_heads,
            'lr_embeddings': lr_embeddings,
            'lr_fusion': lr_fusion,
            'lr_clip': lr_clip,
            'lr_scene': lr_scene,
            'weight_decay': weight_decay,
            'beta': beta,
            'use_lr_schedule': use_lr_schedule,
            'schedule_type': schedule_type,
        }
        
        # Override paths if provided
        if phase1_checkpoint:
            kwargs['phase1_checkpoint_path'] = phase1_checkpoint
        if train_csv:
            kwargs['train_csv_path'] = train_csv
        if val_csv:
            kwargs['val_csv_path'] = val_csv
        if checkpoint_dir:
            kwargs['checkpoint_dir_path'] = checkpoint_dir
        if cell_metadata:
            kwargs['cell_metadata_path'] = cell_metadata
        
        print("=" * 70)
        print("LAUNCHING PHASE 2 TRAINING ON MODAL")
        print("=" * 70)
        print(f"GPU: {gpu}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Detached mode: {detach}")
        print("=" * 70)
        
        if detach:
            train_on_modal.spawn(**kwargs)
            print("\n✅ Training job spawned in detached mode")
            print("   The job will continue even if you disconnect")
            print("   Check Modal dashboard for progress")
        else:
            train_on_modal.remote(**kwargs)
            print("\n✅ Training complete")


if __name__ == "__main__":
    main()
