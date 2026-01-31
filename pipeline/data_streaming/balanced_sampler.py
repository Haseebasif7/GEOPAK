"""Balanced batch sampler for province-aware training"""
from torch.utils.data import Sampler
import numpy as np
from collections import defaultdict
import random


class ProvinceBalancedBatchSampler(Sampler):
    """Sampler that ensures each batch has a balanced distribution of provinces."""
    
    def __init__(self, dataset, province_batch_split, batches_per_epoch=1300, shuffle=True, seed=42):
        """
        Args:
            dataset: GeopakDataset instance
            province_batch_split: Dict mapping province names to samples per batch
                Example: {'Sindh': 16, 'Punjab': 8, ...}
            batches_per_epoch: Fixed number of batches per epoch (default: 1300)
            shuffle: Whether to shuffle batches and samples
            seed: Random seed
        """
        self.dataset = dataset
        self.province_batch_split = province_batch_split
        self.batches_per_epoch = batches_per_epoch
        self.shuffle = shuffle
        self.seed = seed
        
        # Verify batch size
        self.batch_size = sum(province_batch_split.values())
        assert self.batch_size > 0, "Batch size must be > 0"
        
        # Group indices by province
        self.province_indices = defaultdict(list)
        for idx in range(len(dataset)):
            row = dataset.df.iloc[idx]
            province = row.get('province', 'Unknown')
            self.province_indices[province].append(idx)
        
        # Verify all provinces have samples (silently)
        for province, required in province_batch_split.items():
            available = len(self.province_indices.get(province, []))
            if available == 0:
                raise ValueError(f"No samples found for province: {province}")
        
        # Set number of batches (fixed, not limited by rarest province)
        self.num_batches = batches_per_epoch
    
    def __iter__(self):
        """Generate batches with balanced province distribution."""
        # Set random seed for reproducibility (but allow different order each epoch)
        # Use epoch number if available, otherwise use base seed
        epoch_seed = self.seed
        if self.shuffle:
            # Add some randomness for shuffling while keeping reproducibility
            epoch_seed = epoch_seed + hash(str(self.num_batches)) % 1000
        
        random.seed(epoch_seed)
        np.random.seed(epoch_seed)
        
        # Shuffle indices within each province (will repeat if needed)
        province_indices_list = {}
        for province, indices in self.province_indices.items():
            shuffled = indices.copy()
            if self.shuffle:
                random.shuffle(shuffled)
            province_indices_list[province] = shuffled
        
        # Generate batches (with repetition allowed)
        batch_indices = []
        
        for batch_idx in range(self.num_batches):
            batch = []
            
            # Sample from each province according to split
            for province, count in self.province_batch_split.items():
                province_list = province_indices_list[province]
                
                # Always sample randomly (with replacement allowed)
                # This ensures small provinces repeat more, large provinces repeat less
                # Augmentation makes repeated images look different
                if self.shuffle:
                    # Random sample with replacement (standard for imbalanced datasets)
                    sampled = random.choices(province_list, k=count)
                else:
                    # Sequential sampling with wrap-around
                    start_idx = (batch_idx * count) % len(province_list)
                    sampled = []
                    for i in range(count):
                        idx = (start_idx + i) % len(province_list)
                        sampled.append(province_list[idx])
                
                batch.extend(sampled)
            
            # Shuffle within batch to mix provinces
            if self.shuffle:
                random.shuffle(batch)
            
            batch_indices.append(batch)
        
        # Shuffle batches if requested
        if self.shuffle:
            random.shuffle(batch_indices)
        
        return iter(batch_indices)
    
    def __len__(self):
        return self.num_batches


# Default batch split configuration
DEFAULT_BATCH_SPLIT = {
    'Sindh': 16,
    'Punjab': 8,
    'Khyber Pakhtunkhwa': 8,
    'ICT': 8,
    'Balochistan': 8,
    'Gilgit-Baltistan': 8,
    'Azad Kashmir': 8
}

# Default batches per epoch (reasonable for large datasets)
DEFAULT_BATCHES_PER_EPOCH = 1300

# Verify default sums to 64
assert sum(DEFAULT_BATCH_SPLIT.values()) == 64, "Default batch split must sum to 64"
