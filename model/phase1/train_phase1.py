"""
"""Training script for Phase 1 - Geography Structure Learning"""
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

from pipeline.data_streaming.geo_dataset import GeopakDataset
from pipeline.data_streaming.transforms import get_train_transforms, get_val_test_transforms
from pipeline.data_streaming.balanced_sampler import ProvinceBalancedBatchSampler, DEFAULT_BATCH_SPLIT
from model.phase1.geopak_phase1 import GeopakPhase1Model
from model.phase1.losses import Phase1TotalLoss
from model.phase1.utils import haversine_km
from model.phase1.model_utils import create_phase1_model
from model.province.train_province import calculate_class_weights


def collate_fn(batch):
    """Custom collate function that filters out None values (missing images)"""
    # Filter out items with None images
    valid_batch = []
    missing_ids = []
    
    for item in batch:
        if item is None:
            continue
        # Check if image is None (missing image)
        if item.get('image') is None:
            missing_ids.append(item.get('id', 'unknown'))
        else:
            valid_batch.append(item)
    
    if len(valid_batch) == 0:
        return None
    
    # Log if some items were filtered with their IDs
    if len(missing_ids) > 0:
        filtered_count = len(missing_ids)
        # Show first 10 IDs to avoid spam
        ids_str = ', '.join(str(id) for id in missing_ids[:10])
        if len(missing_ids) > 10:
            ids_str += f", ... (+{len(missing_ids) - 10} more)"
        print(f"⚠️  Filtered {filtered_count} missing image(s) from batch - IDs: {ids_str}")
    
    from torch.utils.data._utils.collate import default_collate
    return default_collate(valid_batch)


def load_province_mapping(project_root_path=None):
    """Load province name to ID mapping (same as Phase 0)"""
    if project_root_path is None:
        project_root_path = project_root
    mapping_path = Path(project_root_path) / "model" / "province_mapping.json"
    with open(mapping_path, 'r') as f:
        return json.load(f)


def get_device():
    """Get the best available device"""
    # On Modal with GPU, always use CUDA
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train_epoch(model, train_loader, loss_fn, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_losses = defaultdict(float)
    running_counts = defaultdict(int)
    
    # Metrics
    province_correct = defaultdict(int)
    province_total = defaultdict(int)
    province_mapping = load_province_mapping()
    
    # For error calculation
    all_errors_km = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        if batch is None:
            continue
        
        images = batch['image'].to(device)
        provinces = batch['province']  # List of province names
        
        # Convert to tensors (handle both tensor and list inputs)
        if isinstance(batch['latitude'], torch.Tensor):
            latitudes = batch['latitude'].to(device)
        else:
            latitudes = torch.tensor(batch['latitude'], dtype=torch.float32, device=device)
        
        if isinstance(batch['longitude'], torch.Tensor):
            longitudes = batch['longitude'].to(device)
        else:
            longitudes = torch.tensor(batch['longitude'], dtype=torch.float32, device=device)
        
        # Get cell_id and province_id from dataset
        if isinstance(batch['cell_id'], torch.Tensor):
            cell_ids = batch['cell_id'].to(device)
        else:
            cell_ids = torch.tensor(batch['cell_id'], dtype=torch.long, device=device)
        
        if isinstance(batch['province_id'], torch.Tensor):
            province_ids = batch['province_id'].to(device)
        else:
            province_ids = torch.tensor(batch['province_id'], dtype=torch.long, device=device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images, cell_ids=cell_ids, province_ids=province_ids, return_all=False)
        
        # Compute loss with epoch for ramp-up
        loss_dict = loss_fn(outputs, province_ids, cell_ids, latitudes, longitudes, epoch=epoch)
        loss = loss_dict['total']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate losses (skip 'total' and 'offset_weight' - they're not individual losses)
        for key, value in loss_dict.items():
            if key not in ['total', 'offset_weight']:
                # Handle both tensor and float values
                if isinstance(value, torch.Tensor):
                    running_losses[key] += value.item()
                else:
                    running_losses[key] += float(value)
                running_counts[key] += 1
        
        # Province accuracy
        province_logits = outputs['province_logits']
        _, predicted = torch.max(province_logits.data, 1)
        for i, prov in enumerate(provinces):
            if prov in province_mapping:
                province_total[prov] += 1
                if predicted[i] == province_ids[i]:
                    province_correct[prov] += 1
        
        # Compute prediction errors (using mixture of hypotheses would be expensive, use cell center + offset)
        with torch.no_grad():
            offsets = outputs['offsets']
            if offsets is not None:
                # Get cell centers from model buffers
                cell_lats = model.cell_centers_lat[cell_ids]
                cell_lons = model.cell_centers_lon[cell_ids]
                pred_lats = cell_lats + offsets[:, 0]
                pred_lons = cell_lons + offsets[:, 1]
                
                # Compute errors
                errors = haversine_km(pred_lats, pred_lons, latitudes, longitudes)
                all_errors_km.extend(errors.cpu().numpy().tolist())
        
        # Update progress bar
        avg_loss = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
        avg_province_loss = running_losses['province'] / max(running_counts['province'], 1)
        avg_geocell_loss = running_losses['geocell'] / max(running_counts['geocell'], 1)
        avg_offset_loss = running_losses['offset'] / max(running_counts['offset'], 1)
        offset_weight = loss_dict.get('offset_weight', 1.0)
        
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'prov': f'{avg_province_loss:.4f}',
            'cell': f'{avg_geocell_loss:.4f}',
            'off': f'{avg_offset_loss:.4f}',
            'off_w': f'{offset_weight:.2f}'
        })
    
    # Calculate epoch metrics
    epoch_losses = {key: value / max(running_counts[key], 1) 
                   for key, value in running_losses.items()}
    epoch_losses['total'] = sum(epoch_losses.values())
    
    # Province accuracy
    province_acc = {}
    for prov in province_total:
        if province_total[prov] > 0:
            province_acc[prov] = 100 * province_correct[prov] / province_total[prov]
    
    # Error metrics
    if all_errors_km:
        median_error = np.median(all_errors_km)
        mean_error = np.mean(all_errors_km)
        p90_error = np.percentile(all_errors_km, 90)
    else:
        median_error = mean_error = p90_error = 0.0
    
    return epoch_losses, province_acc, median_error, mean_error, p90_error


def validate(model, val_loader, loss_fn, device, epoch):
    """Validate the model"""
    model.eval()
    running_losses = defaultdict(float)
    running_counts = defaultdict(int)
    
    # Metrics
    province_correct = defaultdict(int)
    province_total = defaultdict(int)
    province_mapping = load_province_mapping()
    
    # For error calculation
    all_errors_km = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch in pbar:
            if batch is None:
                continue
            
            images = batch['image'].to(device)
            provinces = batch['province']
            
            # Convert to tensors (handle both tensor and list inputs)
            if isinstance(batch['latitude'], torch.Tensor):
                latitudes = batch['latitude'].to(device)
            else:
                latitudes = torch.tensor(batch['latitude'], dtype=torch.float32, device=device)
            
            if isinstance(batch['longitude'], torch.Tensor):
                longitudes = batch['longitude'].to(device)
            else:
                longitudes = torch.tensor(batch['longitude'], dtype=torch.float32, device=device)
            
            # Get cell_id and province_id from dataset
            if isinstance(batch['cell_id'], torch.Tensor):
                cell_ids = batch['cell_id'].to(device)
            else:
                cell_ids = torch.tensor(batch['cell_id'], dtype=torch.long, device=device)
            
            if isinstance(batch['province_id'], torch.Tensor):
                province_ids = batch['province_id'].to(device)
            else:
                province_ids = torch.tensor(batch['province_id'], dtype=torch.long, device=device)
            
            # Forward pass
            outputs = model(images, cell_ids=cell_ids, province_ids=province_ids, return_all=False)
            
            # Compute loss
            loss_dict = loss_fn(outputs, province_ids, cell_ids, latitudes, longitudes, epoch=epoch)
            
            # Accumulate losses (skip 'total' and 'offset_weight' - they're not individual losses)
            for key, value in loss_dict.items():
                if key not in ['total', 'offset_weight']:
                    # Handle both tensor and float values
                    if isinstance(value, torch.Tensor):
                        running_losses[key] += value.item()
                    else:
                        running_losses[key] += float(value)
                    running_counts[key] += 1
            
            # Province accuracy
            province_logits = outputs['province_logits']
            _, predicted = torch.max(province_logits.data, 1)
            for i, prov in enumerate(provinces):
                if prov in province_mapping:
                    province_total[prov] += 1
                    if predicted[i] == province_ids[i]:
                        province_correct[prov] += 1
            
            # Compute prediction errors
            offsets = outputs['offsets']
            if offsets is not None:
                cell_lats = model.cell_centers_lat[cell_ids]
                cell_lons = model.cell_centers_lon[cell_ids]
                pred_lats = cell_lats + offsets[:, 0]
                pred_lons = cell_lons + offsets[:, 1]
                
                errors = haversine_km(pred_lats, pred_lons, latitudes, longitudes)
                all_errors_km.extend(errors.cpu().numpy().tolist())
            
            # Update progress bar
            total_loss = loss_dict['total']
            avg_loss = total_loss.item() if isinstance(total_loss, torch.Tensor) else float(total_loss)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'prov': f'{running_losses["province"] / max(running_counts["province"], 1):.4f}'
            })
    
    # Calculate epoch metrics
    epoch_losses = {key: value / max(running_counts[key], 1) 
                   for key, value in running_losses.items()}
    epoch_losses['total'] = sum(epoch_losses.values())
    
    # Province accuracy
    province_acc = {}
    for prov in province_total:
        if province_total[prov] > 0:
            province_acc[prov] = 100 * province_correct[prov] / province_total[prov]
    
    # Error metrics
    if all_errors_km:
        median_error = np.median(all_errors_km)
        mean_error = np.mean(all_errors_km)
        p90_error = np.percentile(all_errors_km, 90)
    else:
        median_error = mean_error = p90_error = 0.0
    
    return epoch_losses, province_acc, median_error, mean_error, p90_error


def print_province_metrics(province_acc, split="Train", log_file=None):
    """Print per-province accuracy metrics and optionally log to file"""
    province_mapping = load_province_mapping()
    
    # Print to console
    print(f"\n📊 {split} Accuracy by Province:")
    for province, province_id in sorted(province_mapping.items(), key=lambda x: x[1]):
        acc = province_acc.get(province, 0.0)
        print(f"   {province:<25} {acc:>5.2f}%")
    
    # Log to file if provided
    if log_file is not None:
        log_file.write(f"\n{split} Accuracy by Province:\n")
        for province, province_id in sorted(province_mapping.items(), key=lambda x: x[1]):
            acc = province_acc.get(province, 0.0)
            log_file.write(f"   {province:<25} {acc:>5.2f}%\n")
        log_file.flush()  # Ensure it's written immediately


def save_checkpoint(model, optimizer, epoch, losses, province_acc, median_error, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'province_accuracy': province_acc,
        'median_error_km': median_error,
    }
    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint saved: {filepath}")


def train_main(
    train_csv_path=None,
    val_csv_path=None,
    checkpoint_dir_path=None,
    cell_metadata_path=None,
    batch_size=64,
    num_epochs=30,
    learning_rate=1e-3,
    weight_decay=1e-5,
    beta=0.999,
    temperature=1.0,
    num_workers=4,
    pin_memory=False,
):
    """
    Main training function that can run locally or on Modal.
    
    Args:
        train_csv_path: Path to train.csv
        val_csv_path: Path to test.csv
        checkpoint_dir_path: Path to checkpoint directory
        cell_metadata_path: Path to cell_metadata.csv
        batch_size: Batch size
        num_epochs: Number of training epochs (25-30 for Phase 1)
        learning_rate: Learning rate (1e-3 for Phase 1)
        weight_decay: Weight decay
        beta: Effective-number weighting parameter
        temperature: Temperature scaling for province head
        num_workers: Number of data loader workers
        pin_memory: Enable pin_memory for faster GPU transfer
    """
    # Configuration
    if train_csv_path is None:
        train_csv = project_root / 'train.csv'
    else:
        train_csv = Path(train_csv_path)
    
    if val_csv_path is None:
        val_csv = project_root / 'test.csv'
    else:
        val_csv = Path(val_csv_path)
    
    if checkpoint_dir_path is None:
        checkpoint_dir = project_root / 'checkpoints' / 'phase1'
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
    
    # Open CSV file for structured loss logging (for visualization)
    csv_log_path = checkpoint_dir / 'losses.csv'
    csv_log_file = open(csv_log_path, 'w')
    # Write CSV header
    csv_log_file.write("epoch,split,loss_total,loss_province,loss_geocell,loss_offset,loss_aux,median_error_km,mean_error_km,p90_error_km,")
    csv_log_file.write("sindh_acc,punjab_acc,kpk_acc,ict_acc,gb_acc,balochistan_acc,ajk_acc\n")
    csv_log_file.flush()
    
    print("=" * 70)
    print("GEOPAK PHASE 1 TRAINING - Geography Structure Learning")
    print("=" * 70)
    
    # Log to file
    log_file.write("=" * 70 + "\n")
    log_file.write("GEOPAK PHASE 1 TRAINING - Geography Structure Learning\n")
    log_file.write("=" * 70 + "\n")
    log_file.flush()
    
    # Device
    device = get_device()
    device_msg = f"\n🖥️  Device: {device}\n"
    print(device_msg)
    log_file.write(device_msg)
    
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
    
    # Verify dataset has required columns
    required_cols = ['cell_id', 'province_id', 'latitude', 'longitude', 'province']
    missing_cols = [col for col in required_cols if col not in train_dataset.df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")
    
    # Calculate province weights (using same function as Phase 0)
    print(f"\n⚖️  Calculating province weights...")
    log_file.write(f"\n⚖️  Calculating province weights...\n")
    # Use Phase 0's calculate_class_weights function
    weight_tensor, province_counts = calculate_class_weights(train_dataset, beta=beta)
    
    # Convert to dict format expected by Phase1TotalLoss
    province_mapping = load_province_mapping()
    province_weights = {}
    for province, province_id in province_mapping.items():
        province_weights[province] = weight_tensor[province_id].item()
    
    # Log weights to file
    log_file.write("\nProvince Weights:\n")
    for province, province_id in sorted(province_mapping.items(), key=lambda x: x[1]):
        weight = province_weights[province]
        count = province_counts.get(province, 0)
        log_file.write(f"   {province:<25} Count: {count:>6,}, Weight: {weight:>5.2f}\n")
    log_file.flush()
    
    # Create model
    print(f"\n🤖 Initializing Phase 1 model...")
    log_file.write(f"\n🤖 Initializing Phase 1 model...\n")
    # Try to find scene model file
    scene_model_path = None
    for path in [
        checkpoint_dir.parent.parent / "resnet50_places365.pth.tar",
        project_root / "resnet50_places365.pth.tar",
        Path("/data/resnet50_places365.pth.tar"),
    ]:
        if path.exists():
            scene_model_path = path
            break
    
    # Try to find Phase 0 checkpoint to load weights from
    phase0_checkpoint_path = None
    for path in [
        checkpoint_dir.parent / "province" / "province_best.pt",
        project_root / "checkpoints" / "province" / "province_best.pt",
        Path("/data/checkpoints/province/province_best.pt"),
    ]:
        if path.exists():
            phase0_checkpoint_path = path
            break
    
    if phase0_checkpoint_path:
        phase0_msg = f"   Found Phase 0 checkpoint: {phase0_checkpoint_path}\n"
        print(phase0_msg)
        log_file.write(phase0_msg)
    else:
        phase0_warn = f"   ⚠️  Phase 0 checkpoint not found, will start from scratch\n"
        print(phase0_warn)
        log_file.write(phase0_warn)
    
    model = create_phase1_model(
        cell_metadata_path=cell_metadata,
        freeze_clip=True,
        freeze_scene=True,
        scene_model_path=str(scene_model_path) if scene_model_path else None,
        temperature=temperature,
        device=device,
        phase0_checkpoint_path=phase0_checkpoint_path
    )
    
    temp_msg = f"   Temperature scaling: {temperature}\n"
    print(temp_msg)
    log_file.write(temp_msg)
    
    # Count trainable parameters
    from model.phase1.model_utils import get_trainable_parameters
    param_counts = get_trainable_parameters(model)
    param_msg = f"   Trainable parameters: {param_counts['total']:,}\n"
    param_msg += f"     - Fusion & Projections: {param_counts['encoder_fusion'] + param_counts['encoder_projections']:,}\n"
    param_msg += f"     - Province head: {param_counts['province_head']:,}\n"
    param_msg += f"     - Geocell heads: {param_counts['geocell_heads']:,}\n"
    param_msg += f"     - Embeddings: {param_counts['embeddings']:,}\n"
    param_msg += f"     - Offset head: {param_counts['offset_head']:,}\n"
    param_msg += f"     - Aux head: {param_counts['aux_head']:,}\n"
    print(param_msg)
    log_file.write(param_msg)
    log_file.flush()
    
    # Create loss function
    print(f"\n📊 Creating loss function...")
    log_file.write(f"\n📊 Creating loss function...\n")
    loss_fn = Phase1TotalLoss(
        province_weights=province_weights,
        cell_metadata=model.cell_metadata_df,
        cell_neighbors=model.cell_neighbors,
        device=device,
        use_offset_rampup=True
    )
    
    # Create data loaders
    print(f"\n🔄 Creating data loaders...")
    log_file.write(f"\n🔄 Creating data loaders...\n")
    balanced_sampler = ProvinceBalancedBatchSampler(
        dataset=train_dataset,
        province_batch_split=DEFAULT_BATCH_SPLIT,
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
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Training loop
    train_start_msg = f"\nStarting training...\n"
    train_start_msg += f"   Epochs: {num_epochs}\n"
    train_start_msg += f"   Learning rate: {learning_rate}\n"
    train_start_msg += f"   Weight decay: {weight_decay}\n"
    train_start_msg += f"   Temperature: {temperature}\n"
    train_start_msg += "=" * 70 + "\n"
    print(train_start_msg)
    log_file.write(train_start_msg)
    log_file.flush()
    
    best_val_error = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_losses, train_prov_acc, train_med_err, train_mean_err, train_p90_err = train_epoch(
            model, train_loader, loss_fn, optimizer, device, epoch
        )
        
        # Validate
        val_losses, val_prov_acc, val_med_err, val_mean_err, val_p90_err = validate(
            model, val_loader, loss_fn, device, epoch
        )
        
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
        
        # Log losses to CSV for visualization
        province_mapping = load_province_mapping()
        province_order = ["Sindh", "Punjab", "Khyber Pakhtunkhwa", "ICT", 
                         "Gilgit-Baltistan", "Balochistan", "Azad Kashmir"]
        
        # Train losses
        train_accs = [train_prov_acc.get(prov, 0.0) for prov in province_order]
        csv_log_file.write(f"{epoch},train,{train_losses['total']:.6f},{train_losses['province']:.6f},")
        csv_log_file.write(f"{train_losses['geocell']:.6f},{train_losses['offset']:.6f},{train_losses['aux']:.6f},")
        csv_log_file.write(f"{train_med_err:.4f},{train_mean_err:.4f},{train_p90_err:.4f},")
        csv_log_file.write(",".join([f"{acc:.4f}" for acc in train_accs]) + "\n")
        
        # Val losses
        val_accs = [val_prov_acc.get(prov, 0.0) for prov in province_order]
        csv_log_file.write(f"{epoch},val,{val_losses['total']:.6f},{val_losses['province']:.6f},")
        csv_log_file.write(f"{val_losses['geocell']:.6f},{val_losses['offset']:.6f},{val_losses['aux']:.6f},")
        csv_log_file.write(f"{val_med_err:.4f},{val_mean_err:.4f},{val_p90_err:.4f},")
        csv_log_file.write(",".join([f"{acc:.4f}" for acc in val_accs]) + "\n")
        csv_log_file.flush()
        
        # Save checkpoint for each epoch
        checkpoint_path = checkpoint_dir / f'phase1_epoch_{epoch}.pt'
        save_checkpoint(
            model, optimizer, epoch, val_losses, val_prov_acc, val_med_err, checkpoint_path
        )
        
        # Save best model (based on validation median error)
        if val_med_err < best_val_error:
            best_val_error = val_med_err
            best_path = checkpoint_dir / 'phase1_best.pt'
            save_checkpoint(
                model, optimizer, epoch, val_losses, val_prov_acc, val_med_err, best_path
            )
            best_msg = f"New best validation median error: {best_val_error:.2f}km\n"
            print(best_msg)
            log_file.write(best_msg)
            log_file.flush()
        
        separator = "=" * 70 + "\n"
        print(separator)
        log_file.write(separator)
        log_file.flush()
    
    completion_msg = "\nTraining complete!\n"
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
    
    # Create Modal app
    app = modal.App("geopak-phase1-training", image=image)
    
    # Create volume for data and checkpoints
    volume = modal.Volume.from_name("geopak-data", create_if_missing=True)
    
    @app.function(
        gpu="A10",
        timeout=86400,  # 24 hours
        volumes={"/data": volume},
    )
    def train_on_modal(
        train_csv_path="/data/train.csv",
        val_csv_path="/data/test.csv",
        checkpoint_dir_path="/data/checkpoints/phase1",
        cell_metadata_path="/data/pipeline/geocells/cell_metadata.csv",
        batch_size=64,
        num_epochs=30,
        learning_rate=1e-3,
        weight_decay=1e-5,
        beta=0.999,
        temperature=1.0,
    ):
        """Training function that runs on Modal GPU"""
        import sys
        from pathlib import Path
        
        modal_project_root = Path("/root/geopak")
        sys.path.insert(0, str(modal_project_root))
        
        import model.phase1.train_phase1 as train_module
        train_module.project_root = modal_project_root
        
        train_main(
            train_csv_path=train_csv_path,
            val_csv_path=val_csv_path,
            checkpoint_dir_path=checkpoint_dir_path,
            cell_metadata_path=cell_metadata_path,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta=beta,
            temperature=temperature,
            num_workers=4,
            pin_memory=True,
        )
        
        volume.commit()
        print("✅ Volume changes committed")
    
    @app.local_entrypoint()
    def main_modal(
        train_csv: str = None,
        val_csv: str = None,
        checkpoint_dir: str = None,
        cell_metadata: str = None,
        batch_size: int = 64,
        num_epochs: int = 30,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        beta: float = 0.999,
        temperature: float = 1.0,
        gpu: str = "A100",
        detach: bool = False,
    ):
        """Entry point for Modal training
        
        Args:
            detach: If True, spawns the function in detached mode (continues even if local client disconnects)
        """
        if train_csv is None:
            train_csv = "/data/train.csv"
        if val_csv is None:
            val_csv = "/data/test.csv"
        if checkpoint_dir is None:
            checkpoint_dir = "/data/checkpoints/phase1"
        if cell_metadata is None:
            cell_metadata = "/data/pipeline/geocells/cell_metadata.csv"
        
        if detach:
            # Spawn in detached mode - continues even if local client disconnects
            print("🚀 Starting training in DETACHED mode...")
            print("   Training will continue even if this terminal disconnects.")
            print("   Check progress at: https://modal.com/apps")
            print("   You can also use: modal run model/phase1/train_phase1.py --modal --detach")
            try:
                # Try using spawn() if available
                train_on_modal.spawn(
                    train_csv_path=train_csv,
                    val_csv_path=val_csv,
                    checkpoint_dir_path=checkpoint_dir,
                    cell_metadata_path=cell_metadata,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    beta=beta,
                    temperature=temperature,
                )
            except AttributeError:
                # Fallback: spawn might not be available, use remote with note
                print("⚠️  spawn() not available. Using remote() - consider using 'modal run --detach' from command line")
                train_on_modal.remote(
                    train_csv_path=train_csv,
                    val_csv_path=val_csv,
                    checkpoint_dir_path=checkpoint_dir,
                    cell_metadata_path=cell_metadata,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    beta=beta,
                    temperature=temperature,
                )
        else:
            # Run in attached mode (blocks until completion)
            train_on_modal.remote(
                train_csv_path=train_csv,
                val_csv_path=val_csv,
                checkpoint_dir_path=checkpoint_dir,
                cell_metadata_path=cell_metadata,
                batch_size=batch_size,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                beta=beta,
                temperature=temperature,
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Phase 1 Model')
    parser.add_argument('--modal', action='store_true', help='Run on Modal GPU')
    parser.add_argument('--train-csv', type=str, help='Path to train.csv')
    parser.add_argument('--val-csv', type=str, help='Path to validation/test.csv')
    parser.add_argument('--checkpoint-dir', type=str, help='Path to checkpoint directory')
    parser.add_argument('--cell-metadata', type=str, help='Path to cell_metadata.csv')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--beta', type=float, default=0.999, help='Effective-number beta')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature scaling')
    parser.add_argument('--gpu', type=str, default='A100', help='GPU type for Modal')
    parser.add_argument('--detach', action='store_true', help='Run in detached mode (continues even if local client disconnects)')
    
    args = parser.parse_args()
    
    if args.modal:
        if not MODAL_AVAILABLE:
            print("❌ Modal is not installed. Install it with: pip install modal")
            sys.exit(1)
        
        with app.run():
            main_modal(
                train_csv=args.train_csv,
                val_csv=args.val_csv,
                checkpoint_dir=args.checkpoint_dir,
                cell_metadata=args.cell_metadata,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                beta=args.beta,
                temperature=args.temperature,
                gpu=args.gpu,
                detach=args.detach,
            )
    else:
        # Run locally
        train_main(
            train_csv_path=args.train_csv,
            val_csv_path=args.val_csv,
            checkpoint_dir_path=args.checkpoint_dir,
            cell_metadata_path=args.cell_metadata,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            beta=args.beta,
            temperature=args.temperature,
        )
