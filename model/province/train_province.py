"""
Training script for Province Head (Phase 0)
Uses weighted cross-entropy with effective-number weighting

Can run locally or on Modal with GPU support.

LOCAL USAGE:
    python model/province/train_province.py
    python model/province/train_province.py --num-epochs 50 --batch-size 64

MODAL USAGE (GPU Training):
    1. First, upload your data to Modal volume (create upload_data_to_modal.py):
       ```python
       import modal
       volume = modal.Volume.from_name("geopak-data", create_if_missing=True)
       with volume.mount() as mount:
           import shutil
           # Upload CSVs
           shutil.copy('train.csv', mount / 'train.csv')
           shutil.copy('test.csv', mount / 'test.csv')
           # Upload model file
           shutil.copy('resnet50_places365.pth.tar', mount / 'resnet50_places365.pth.tar')
           # Upload datasets directory (if needed)
           if Path('datasets').exists():
               shutil.copytree('datasets', mount / 'datasets', dirs_exist_ok=True)
       volume.commit()
       ```
    
    2. Run training on Modal:
       modal run model/province/train_province.py --modal --num-epochs 50 --gpu A100
    
    3. Checkpoints will be saved to the volume at /data/checkpoints/province/
    
    4. Download checkpoints from volume:
       ```python
       import modal
       volume = modal.Volume.from_name("geopak-data")
       with volume.mount() as mount:
           import shutil
           shutil.copytree(mount / 'checkpoints', './checkpoints', dirs_exist_ok=True)
       ```
"""
import sys
import argparse
from pathlib import Path
import json

# Add project root to Python path
# Handle both local execution and Modal execution
# In Modal, code is at /root/geopak (added via image.add_local_dir)
# Always add /root/geopak to path if it exists (for Modal imports)
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
from model.province.province_head import ProvinceHead


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
    """Load province name to ID mapping"""
    if project_root_path is None:
        project_root_path = project_root
    mapping_path = Path(project_root_path) / "model" / "province_mapping.json"
    with open(mapping_path, 'r') as f:
        return json.load(f)


def calculate_class_weights(dataset, beta=0.995):
    """
    Calculate effective-number class weights for weighted cross-entropy.
    
    Formula:
    1. E_p = (1 - β^n_p) / (1 - β)
    2. w_p = 1 / E_p
    3. w_p_normalized = w_p / mean(w_p)
    
    Args:
        dataset: GeopakDataset instance
        beta: Smoothing factor (default 0.999)
    
    Returns:
        weights: Tensor of shape [num_classes] with normalized weights
        province_counts: Dict mapping province names to counts
    """
    # Count samples per province
    province_counts = defaultdict(int)
    for idx in range(len(dataset)):
        row = dataset.df.iloc[idx]
        province = row.get('province', 'Unknown')
        province_counts[province] += 1
    
    # Load province mapping to get ordered list
    province_mapping = load_province_mapping()
    num_classes = len(province_mapping)
    
    # Calculate effective number per province
    effective_nums = {}
    weights = {}
    
    for province, count in province_counts.items():
        if province in province_mapping:
            # Effective number: E_p = (1 - β^n_p) / (1 - β)
            effective_num = (1 - beta ** count) / (1 - beta)
            effective_nums[province] = effective_num
            # Weight is inverse of effective number
            weights[province] = 1.0 / effective_num
    
    # Create weight tensor in province_id order
    weight_tensor = torch.zeros(num_classes)
    for province, province_id in province_mapping.items():
        if province in weights:
            weight_tensor[province_id] = weights[province]
        else:
            # If province not in dataset, use mean weight
            weight_tensor[province_id] = np.mean(list(weights.values()))
    
    # Normalize by mean (CRITICAL step)
    weight_tensor = weight_tensor / weight_tensor.mean()
    
    # Print weights for verification
    print("\n📊 Class Weights (Effective-Number Weighting):")
    for province, province_id in sorted(province_mapping.items(), key=lambda x: x[1]):
        weight = weight_tensor[province_id].item()
        count = province_counts.get(province, 0)
        print(f"   {province:<25} ID: {province_id}, Count: {count:>6,}, Weight: {weight:>5.2f}")
    
    return weight_tensor, province_counts


def get_device():
    """Get the best available device"""
    # On Modal with GPU, always use CUDA
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Province-specific metrics
    province_correct = defaultdict(int)
    province_total = defaultdict(int)
    province_mapping = load_province_mapping()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        if batch is None:
            continue
        
        images = batch['image'].to(device)
        provinces = batch['province']  # List of province names
        
        # Convert province names to IDs
        province_ids = torch.tensor([
            province_mapping.get(prov, 0) for prov in provinces
        ], dtype=torch.long, device=device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(images)  # [batch_size, 7]
        loss = criterion(logits, province_ids)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += province_ids.size(0)
        correct += (predicted == province_ids).sum().item()
        
        # Per-province metrics
        for i, prov in enumerate(provinces):
            if prov in province_mapping:
                province_total[prov] += 1
                if predicted[i] == province_ids[i]:
                    province_correct[prov] += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, province_correct, province_total


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Province-specific metrics
    province_correct = defaultdict(int)
    province_total = defaultdict(int)
    province_mapping = load_province_mapping()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch in pbar:
            if batch is None:
                continue
            
            images = batch['image'].to(device)
            provinces = batch['province']
            
            # Convert province names to IDs
            province_ids = torch.tensor([
                province_mapping.get(prov, 0) for prov in provinces
            ], dtype=torch.long, device=device)
            
            # Forward pass
            logits = model(images)
            loss = criterion(logits, province_ids)
            
            # Metrics
            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += province_ids.size(0)
            correct += (predicted == province_ids).sum().item()
            
            # Per-province metrics
            for i, prov in enumerate(provinces):
                if prov in province_mapping:
                    province_total[prov] += 1
                    if predicted[i] == province_ids[i]:
                        province_correct[prov] += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, province_correct, province_total


def print_province_metrics(province_correct, province_total, split="Train"):
    """Print per-province accuracy metrics"""
    print(f"\n📊 {split} Accuracy by Province:")
    province_mapping = load_province_mapping()
    for province, province_id in sorted(province_mapping.items(), key=lambda x: x[1]):
        total = province_total.get(province, 0)
        correct = province_correct.get(province, 0)
        if total > 0:
            acc = 100 * correct / total
            print(f"   {province:<25} {correct:>4}/{total:<4} ({acc:>5.2f}%)")
        else:
            print(f"   {province:<25} No samples")


def save_checkpoint(model, optimizer, epoch, loss, acc, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': acc,
    }
    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint saved: {filepath}")


def train_main(
    train_csv_path=None,
    val_csv_path=None,
    checkpoint_dir_path=None,
    batch_size=64,
    num_epochs=8,
    learning_rate=1e-4,
    weight_decay=1e-5,
    beta=0.999,
    temperature=1.0,
    num_workers=4,  # Use more workers on GPU
    pin_memory=False,  # Enable pin_memory on GPU
):
    """
    Main training function that can run locally or on Modal.
    
    Args:
        train_csv_path: Path to train.csv (if None, uses project_root/train.csv)
        val_csv_path: Path to test.csv (if None, uses project_root/test.csv)
        checkpoint_dir_path: Path to checkpoint directory (if None, uses project_root/checkpoints/province)
        batch_size: Batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        beta: Effective-number weighting parameter
        num_workers: Number of data loader workers (0 for local, 4+ for GPU)
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
        checkpoint_dir = project_root / 'checkpoints' / 'province'
    else:
        checkpoint_dir = Path(checkpoint_dir_path)
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("GEOPAK PROVINCE HEAD TRAINING")
    print("=" * 70)
    
    # Device
    device = get_device()
    print(f"\n🖥️  Device: {device}")
    
    # Load datasets
    print(f"\n📦 Loading datasets...")
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
    
    print(f"   Train: {len(train_dataset):,} samples")
    print(f"   Val:   {len(val_dataset):,} samples")
    
    # Calculate class weights
    print(f"\n⚖️  Calculating class weights...")
    class_weights, province_counts = calculate_class_weights(train_dataset, beta=beta)
    class_weights = class_weights.to(device)
    
    # Create data loaders
    print(f"\n🔄 Creating data loaders...")
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
    
    # Initialize model
    print(f"\n🤖 Initializing model...")
    # Try to find model file in multiple locations
    scene_model_path = None
    for path in [
        checkpoint_dir.parent.parent / "resnet50_places365.pth.tar",  # In volume /data/
        project_root / "resnet50_places365.pth.tar",  # In project root
        Path("/data/resnet50_places365.pth.tar"),  # In Modal volume
    ]:
        if path.exists():
            scene_model_path = path
            break
    
    if scene_model_path is None:
        print("⚠️  resnet50_places365.pth.tar not found, model will try to load from default location")
    
    model = ProvinceHead(
        fusion_dim=512,
        hidden_dim=256,
        freeze_clip=True,
        freeze_scene=True,
        scene_model_path=str(scene_model_path) if scene_model_path else None,
        temperature=temperature
    ).to(device)
    
    print(f"   Temperature scaling: {temperature}")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Training loop
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Weight decay: {weight_decay}")
    print(f"   Temperature: {temperature}")
    print("=" * 70)
    
    best_val_acc = 0.0
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_acc, train_prov_correct, train_prov_total = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc, val_prov_correct, val_prov_total = validate(
            model, val_loader, criterion, device
        )
        
        # Print metrics
        print(f"\nEpoch {epoch}/{num_epochs}:")
        print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        
        print_province_metrics(train_prov_correct, train_prov_total, "Train")
        print_province_metrics(val_prov_correct, val_prov_total, "Val")
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / f'province_epoch_{epoch}.pt'
        save_checkpoint(model, optimizer, epoch, val_loss, val_acc, checkpoint_path)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = checkpoint_dir / 'province_best.pt'
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, best_path)
            print(f"🏆 New best validation accuracy: {best_val_acc:.2f}%")
        
        print("=" * 70)
    
    print("\n✅ Training complete!")


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
    app = modal.App("geopak-province-training", image=image)
    
    # Create volume for data and checkpoints (persists across runs)
    volume = modal.Volume.from_name("geopak-data", create_if_missing=True)
    
    @app.function(
        gpu="T4",  # Use A100 GPU (change to "T4", "A10G", "H100" as needed)
        timeout=86400,  # 24 hours timeout
        volumes={"/data": volume},  # Mount volume at /data for datasets/checkpoints
    )
    def train_on_modal(
        train_csv_path="/data/train.csv",
        val_csv_path="/data/test.csv",
        checkpoint_dir_path="/data/checkpoints/province",
        batch_size=64,
        num_epochs=50,
        learning_rate=1e-4,
        weight_decay=1e-5,
        beta=0.999,
        temperature=1.0,
    ):
        """
        Training function that runs on Modal GPU.
        Assumes data is already uploaded to the volume.
        """
        import sys
        from pathlib import Path
        
        # Project code is included in the image at /root/geopak
        modal_project_root = Path("/root/geopak")
        sys.path.insert(0, str(modal_project_root))
        
        # Update module-level project_root for helper functions
        import model.province.train_province as train_module
        train_module.project_root = modal_project_root
        
        # Run training
        train_main(
            train_csv_path=train_csv_path,
            val_csv_path=val_csv_path,
            checkpoint_dir_path=checkpoint_dir_path,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta=beta,
            temperature=temperature,
            num_workers=4,  # Use multiple workers on GPU
            pin_memory=True,  # Enable for GPU
        )
        
        # Commit volume changes
        volume.commit()
        print("✅ Volume changes committed")
    
    @app.local_entrypoint()
    def main_modal(
        train_csv: str = None,
        val_csv: str = None,
        checkpoint_dir: str = None,
        batch_size: int = 64,
        num_epochs: int = 50,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        beta: float = 0.9995,
        temperature: float = 1.0,
        gpu: str = "A100",
    ):
        """
        Entry point for Modal training.
        
        Usage:
            modal run model/province/train_province.py --train-csv /data/train.csv --num-epochs 50
        """
        # Set default paths
        if train_csv is None:
            train_csv = "/data/train.csv"
        if val_csv is None:
            val_csv = "/data/test.csv"
        if checkpoint_dir is None:
            checkpoint_dir = "/data/checkpoints/province"
        
        # Run training on Modal
        train_on_modal.remote(
            train_csv_path=train_csv,
            val_csv_path=val_csv,
            checkpoint_dir_path=checkpoint_dir,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta=beta,
            temperature=temperature,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Province Head Model')
    parser.add_argument('--modal', action='store_true', help='Run on Modal GPU')
    parser.add_argument('--train-csv', type=str, help='Path to train.csv')
    parser.add_argument('--val-csv', type=str, help='Path to validation/test.csv')
    parser.add_argument('--checkpoint-dir', type=str, help='Path to checkpoint directory')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=8, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--beta', type=float, default=0.999, help='Effective-number beta')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature scaling for logits (1.0 = no scaling, >1.0 = softer, <1.0 = sharper)')
    parser.add_argument('--gpu', type=str, default='A100', help='GPU type for Modal (A100, T4, A10G, H100)')
    
    args = parser.parse_args()
    
    if args.modal:
        if not MODAL_AVAILABLE:
            print("❌ Modal is not installed. Install it with: pip install modal")
            sys.exit(1)
        
        # Run Modal training using the app defined above
        with app.run():
            main_modal(
                train_csv=args.train_csv,
                val_csv=args.val_csv,
                checkpoint_dir=args.checkpoint_dir,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                beta=args.beta,
                temperature=args.temperature,
                gpu=args.gpu,
            )
    else:
        # Run locally
        train_main(
            train_csv_path=args.train_csv,
            val_csv_path=args.val_csv,
            checkpoint_dir_path=args.checkpoint_dir,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            beta=args.beta,
            temperature=args.temperature,
        )
