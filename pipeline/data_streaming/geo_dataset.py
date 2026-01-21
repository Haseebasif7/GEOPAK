import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import pandas as pd


class GeopakDataset(Dataset):

    def __init__(self, csv_path, transform=None):
        """
        Args:
            csv_path: Path to CSV file with columns: id, latitude, longitude, province, province_id, cell_id, cell_center_lat, cell_center_lon, path
            transform: Optional image transforms
        """
        self.df = pd.read_csv(csv_path)
        self.transform = transform
    
    def _load_from_local(self, local_path):
        """Load image from local filesystem"""
        path = Path(local_path)
        
        # Handle absolute paths
        if path.is_absolute():
            if not path.exists():
                return None
        else:
            # Try relative paths (works with symlinks)
            # First try as-is (relative to current working directory)
            if path.exists():
                pass  # Use path as-is
            else:
                # Try relative to current working directory explicitly
                cwd_path = Path.cwd() / path
                if cwd_path.exists():
                    path = cwd_path
                else:
                    return None
        
        # Load the image
        try:
            return Image.open(path).convert('RGB')
        except Exception as e:
            # Silently fail - will be filtered by collate_fn
            return None
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['path']
        image_id = row['id']
        
        # Load image
        image = self._load_from_local(path)
        
        # If image fails to load, return None (will be filtered by collate_fn)
        if image is None:
            return None
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'id': image_id,
            'latitude': float(row['latitude']),
            'longitude': float(row['longitude']),
            'province': row['province']
        }

    def __len__(self):
        return len(self.df)
