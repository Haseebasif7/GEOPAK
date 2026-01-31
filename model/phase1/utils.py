"""
"""Utility functions for Phase 1 model"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple, Optional


def load_cell_metadata(metadata_path: Path) -> pd.DataFrame:
    """Load cell metadata CSV"""
    return pd.read_csv(metadata_path)


def haversine_km(
    lat1: torch.Tensor,
    lon1: torch.Tensor,
    lat2: torch.Tensor,
    lon2: torch.Tensor
) -> torch.Tensor:
    """
    Compute Haversine distance in km between two sets of coordinates
    
    Args:
        lat1, lon1: [batch_size] or scalar
        lat2, lon2: [batch_size] or scalar
    
    Returns:
        distances: [batch_size] in km
    """
    R = 6371.0  # Earth radius in km
    
    lat1_rad = torch.deg2rad(lat1)
    lon1_rad = torch.deg2rad(lon1)
    lat2_rad = torch.deg2rad(lat2)
    lon2_rad = torch.deg2rad(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    
    return R * c


def compute_province_weights(
    train_csv_path: Path,
    beta: float = 0.999
) -> Dict[str, float]:
    """
    Compute effective-number based province weights
    
    Args:
        train_csv_path: Path to training CSV
        beta: Smoothing factor (default: 0.999)
    
    Returns:
        Dictionary mapping province name to normalized weight
    """
    df = pd.read_csv(train_csv_path)
    province_counts = df['province'].value_counts()
    
    # Calculate effective number per province
    effective_nums = {}
    for province, count in province_counts.items():
        effective_num = (1 - beta ** count) / (1 - beta)
        effective_nums[province] = effective_num
    
    # Weight is inverse of effective number
    weights = {province: 1.0 / eff_num for province, eff_num in effective_nums.items()}
    
    # Normalize by mean
    mean_weight = np.mean(list(weights.values()))
    weights_normalized = {province: w / mean_weight for province, w in weights.items()}
    
    return weights_normalized


def create_distance_aware_labels(
    true_cell_id: int,
    province_id: int,
    cell_metadata: pd.DataFrame,
    cell_neighbors: Dict[int, list],
    tau: float,
    device: torch.device
) -> torch.Tensor:
    """
    Create distance-aware soft labels for geocell classification
    
    Args:
        true_cell_id: True cell ID (global)
        province_id: Province ID
        cell_metadata: Cell metadata DataFrame
        cell_neighbors: Dictionary mapping cell_id -> list of neighbor cell_ids
        tau: Temperature parameter (from TAU_BY_PROVINCE)
        device: Device for tensors
    
    Returns:
        soft_labels: [num_cells_in_province] - soft probability distribution
    """
    # Get all cells for this province
    prov_cells = cell_metadata[cell_metadata['province_id'] == province_id].sort_values('cell_id')
    prov_cell_ids = prov_cells['cell_id'].values
    
    # Get true cell info
    true_cell = cell_metadata[cell_metadata['cell_id'] == true_cell_id].iloc[0]
    true_lat = true_cell['center_lat']
    true_lon = true_cell['center_lon']
    
    # Initialize soft labels
    num_cells = len(prov_cell_ids)
    soft_labels = torch.zeros(num_cells, device=device)
    
    # Get neighbors (including self)
    neighbors = cell_neighbors.get(true_cell_id, [])
    if true_cell_id not in neighbors:
        neighbors = [true_cell_id] + neighbors
    
    # Compute distances and soft labels
    for idx, cell_id in enumerate(prov_cell_ids):
        if cell_id in neighbors or cell_id == true_cell_id:
            cell = cell_metadata[cell_metadata['cell_id'] == cell_id].iloc[0]
            dist_km = haversine_km(
                torch.tensor(true_lat, device=device),
                torch.tensor(true_lon, device=device),
                torch.tensor(cell['center_lat'], device=device),
                torch.tensor(cell['center_lon'], device=device)
            )
            soft_labels[idx] = torch.exp(-dist_km / tau)
    
    # Renormalize
    soft_labels = soft_labels / (soft_labels.sum() + 1e-8)
    
    return soft_labels


# TAU values for distance-aware label smoothing (from plan.md)
TAU_BY_PROVINCE = {
    "Sindh": 30.0,
    "Punjab": 60.0,
    "Khyber Pakhtunkhwa": 50.0,
    "ICT": 10.0,
    "Azad Kashmir": 40.0,
    "Gilgit-Baltistan": 100.0,
    "Balochistan": 100.0,
}


def get_tau_for_province(province_name: str) -> float:
    """Get tau value for a province"""
    return TAU_BY_PROVINCE.get(province_name, 50.0)


# Province scales for offset clamping
PROVINCE_SCALES = {
    "Sindh": 1.0,
    "Punjab": 0.6,
    "Khyber Pakhtunkhwa": 1.0,
    "ICT": 0.6,
    "Gilgit-Baltistan": 1.4,
    "Balochistan": 1.4,
    "Azad Kashmir": 1.0,
}


def get_province_scale(province_name: str) -> float:
    """Get province scale for offset clamping"""
    return PROVINCE_SCALES.get(province_name, 1.0)
