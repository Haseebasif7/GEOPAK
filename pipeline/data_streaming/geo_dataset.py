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
        path_str = str(local_path)
        path = Path(local_path)
        
        # Detect if we're on Modal
        is_modal = Path('/data/datasets').exists()
        
        # Handle paths that reference datasets directory
        if '/Users/haseeb/Desktop/datasets' in path_str:
            modal_path_str = path_str.replace('/Users/haseeb/Desktop/datasets', '/data/datasets')
            modal_path = Path(modal_path_str)
            
            if is_modal:
                if modal_path.exists():
                    path = modal_path
                else:
                    return None
            else:
                if path.exists():
                    pass
                elif modal_path.exists():
                    path = modal_path
                else:
                    return None
        # Case 2: Relative path starting with datasets/
        elif path_str.startswith('datasets/'):
            # Convert relative path to Modal path
            modal_path_str = '/data/' + path_str
            modal_path = Path(modal_path_str)
            
            # Also try with /Users/haseeb/Desktop/ prefix for local
            local_abs_path = Path('/Users/haseeb/Desktop') / path_str
            
            if is_modal:
                if modal_path.exists():
                    path = modal_path
                else:
                    return None
            else:
                # Local: try local absolute path first
                if local_abs_path.exists():
                    path = local_abs_path
                elif path.exists():
                    pass  # Use relative path as-is
                elif modal_path.exists():
                    path = modal_path
                else:
                    return None
        elif path.is_absolute():
            # Other absolute paths - check if they exist
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
                    # Try with /Users/haseeb/Desktop/ prefix
                    desktop_path = Path('/Users/haseeb/Desktop') / path
                    if desktop_path.exists():
                        path = desktop_path
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
        
        # If image fails to load, return dict with None image but keep ID for logging
        if image is None:
            result = {
                'image': None,
                'id': image_id,
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude']),
                'province': row['province'],
                'path': path  # Include path for debugging
            }
            
            # Add cell_id and province_id if available
            if 'cell_id' in row:
                result['cell_id'] = int(row['cell_id'])
            if 'province_id' in row:
                result['province_id'] = int(row['province_id'])
            
            return result
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        result = {
            'image': image,
            'id': image_id,
            'latitude': float(row['latitude']),
            'longitude': float(row['longitude']),
            'province': row['province']
        }
        
        # Add cell_id and province_id if available (needed for Phase 1)
        if 'cell_id' in row:
            result['cell_id'] = int(row['cell_id'])
        if 'province_id' in row:
            result['province_id'] = int(row['province_id'])
        
        return result

    def __len__(self):
        return len(self.df)
