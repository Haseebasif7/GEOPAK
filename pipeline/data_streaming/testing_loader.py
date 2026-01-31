from torch.utils.data import DataLoader
from geo_dataset import GeopakDataset
from transforms import get_train_transforms, get_val_test_transforms
from balanced_sampler import ProvinceBalancedBatchSampler, DEFAULT_BATCH_SPLIT 
from pathlib import Path
import matplotlib.pyplot as plt

def collate_fn(batch):
    """Custom collate function that filters out None values (missing images)"""
    # Filter out None values
    valid_batch = [item for item in batch if item is not None]
    
    if len(valid_batch) == 0:
        # If entire batch is None, skip this batch
        return None
    
    # Log if some items were filtered (only occasionally to avoid spam)
    if len(valid_batch) < len(batch):
        filtered_count = len(batch) - len(valid_batch)
        print(f"⚠️  Filtered {filtered_count} missing image(s) from batch")
    
    # Use default collate for the filtered batch
    from torch.utils.data._utils.collate import default_collate
    return default_collate(valid_batch)

if __name__ == '__main__':
    # Configuration
    csv_path = 'train.csv'
    batch_size = 64
    
    print("GEOPAK TRAINING SETUP")
    
    train_transform = get_train_transforms()
    
    train_dataset = GeopakDataset(
        csv_path=str(csv_path),
        transform=train_transform
    )
    
    balanced_sampler = ProvinceBalancedBatchSampler(
        dataset=train_dataset,
        province_batch_split=DEFAULT_BATCH_SPLIT,
        batches_per_epoch=1300,  # Fixed number of batches per epoch
        shuffle=True,
        seed=42
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=balanced_sampler,
        num_workers=0,  # macOS/MPS compatibility
        pin_memory=False,
        collate_fn=collate_fn
    )  
    
    for batch_idx, batch in enumerate(train_loader):
        if batch is None:
            continue
            
        images = batch['image']  # Shape: [batch_size, 3, 224, 224]
        ids = batch['id']
        provinces = batch['province']
        
        print(f"\nBatch {batch_idx}:")
        print(f"   Image shape: {images.shape}")
        print(f"   Batch size: {len(ids)}")
        plt.imshow(images[0].permute(1, 2, 0))
        plt.show()
        
        # Break after first batch for testing
        if batch_idx >= 0:
            break
    
    print("Setup complete! Ready for training.")
