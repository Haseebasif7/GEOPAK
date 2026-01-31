"""
Loss functions for Phase 1 training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np

from .utils import (
    haversine_km,
    create_distance_aware_labels,
    get_tau_for_province,
    TAU_BY_PROVINCE
)


class ProvinceLoss(nn.Module):
    """Weighted Cross-Entropy loss for province classification"""
    
    def __init__(self, province_weights: Dict[str, float], device: torch.device):
        super().__init__()
        # Create weight tensor in province order
        provinces = ["Sindh", "Punjab", "Khyber Pakhtunkhwa", "ICT", 
                     "Gilgit-Baltistan", "Balochistan", "Azad Kashmir"]
        weights = torch.tensor([province_weights.get(p, 1.0) for p in provinces], 
                               dtype=torch.float32, device=device)
        self.register_buffer('weights', weights)
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, 7]
            targets: [batch_size] - province IDs
        Returns:
            loss: scalar
        """
        return F.cross_entropy(logits, targets, weight=self.weights)


class GeocellLoss(nn.Module):
    """KL Divergence loss for geocell classification with distance-aware label smoothing"""
    
    def __init__(self, cell_metadata, cell_neighbors: Dict[int, list], device: torch.device):
        super().__init__()
        self.cell_metadata = cell_metadata
        self.cell_neighbors = cell_neighbors
        self.device = device
        
    def forward(
        self,
        local_logits: torch.Tensor,
        true_cell_ids: torch.Tensor,
        province_id: int,
        province_name: str
    ) -> torch.Tensor:
        """
        Args:
            local_logits: [batch_size, num_cells_in_province] - logits from province head
            true_cell_ids: [batch_size] - True global cell IDs
            province_id: Province ID
            province_name: Province name (for tau lookup)
        Returns:
            loss: scalar
        """
        batch_size = local_logits.shape[0]
        
        # Get tau for this province
        tau = get_tau_for_province(province_name)
        
        # Get province cells for mapping
        prov_cells = self.cell_metadata[self.cell_metadata['province_id'] == province_id].sort_values('cell_id')
        prov_cell_ids = prov_cells['cell_id'].values
        
        # Create soft labels for each sample
        soft_labels_list = []
        for i in range(batch_size):
            cell_id = true_cell_ids[i].item()
            soft_labels = create_distance_aware_labels(
                true_cell_id=cell_id,
                province_id=province_id,
                cell_metadata=self.cell_metadata,
                cell_neighbors=self.cell_neighbors,
                tau=tau,
                device=self.device
            )
            soft_labels_list.append(soft_labels)
        
        # Stack to [batch_size, num_cells]
        soft_labels = torch.stack(soft_labels_list)  # [batch_size, num_cells]
        
        # Compute KL divergence
        log_probs = F.log_softmax(local_logits, dim=1)
        loss = F.kl_div(log_probs, soft_labels, reduction='batchmean')
        
        return loss


class OffsetLoss(nn.Module):
    """Weighted Haversine loss for offset prediction"""
    
    def __init__(self, cell_centers_lat: np.ndarray, cell_centers_lon: np.ndarray, device: torch.device = None):
        super().__init__()
        # Convert to tensors and register as buffers
        # If device is provided, create tensors on that device
        if device is not None:
            self.register_buffer('cell_centers_lat', torch.tensor(cell_centers_lat, dtype=torch.float32, device=device))
            self.register_buffer('cell_centers_lon', torch.tensor(cell_centers_lon, dtype=torch.float32, device=device))
        else:
            self.register_buffer('cell_centers_lat', torch.tensor(cell_centers_lat, dtype=torch.float32))
            self.register_buffer('cell_centers_lon', torch.tensor(cell_centers_lon, dtype=torch.float32))
        
    def forward(
        self,
        offsets: torch.Tensor,
        cell_ids: torch.Tensor,
        true_lat: torch.Tensor,
        true_lon: torch.Tensor,
        province_probs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            offsets: [batch_size, 2] - predicted offsets (Δlat, Δlon)
            cell_ids: [batch_size] - cell IDs
            true_lat: [batch_size] - true latitude
            true_lon: [batch_size] - true longitude
            province_probs: [batch_size, 7] - optional province probabilities for weighting
        Returns:
            loss: scalar
        """
        # Get cell centers
        cell_lats = self.cell_centers_lat[cell_ids]  # [batch_size]
        cell_lons = self.cell_centers_lon[cell_ids]  # [batch_size]
        
        # Compute predictions: cell_center + offset
        pred_lats = cell_lats + offsets[:, 0]  # [batch_size]
        pred_lons = cell_lons + offsets[:, 1]  # [batch_size]
        
        # Compute Haversine distances
        distances = haversine_km(pred_lats, pred_lons, true_lat, true_lon)  # [batch_size]
        
        # Weight by province probabilities if provided
        if province_probs is not None:
            # Get province ID for each cell (would need to be passed in)
            # For now, just use uniform weighting
            weights = torch.ones_like(distances)
        else:
            weights = torch.ones_like(distances)
        
        # Weighted mean
        loss = (weights * distances).mean()
        
        return loss


class AuxiliaryLoss(nn.Module):
    """Haversine loss for auxiliary coarse regression"""
    
    def forward(
        self,
        aux_coords: torch.Tensor,
        true_lat: torch.Tensor,
        true_lon: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            aux_coords: [batch_size, 2] - predicted (lat, lon)
            true_lat: [batch_size]
            true_lon: [batch_size]
        Returns:
            loss: scalar
        """
        distances = haversine_km(
            aux_coords[:, 0], aux_coords[:, 1],
            true_lat, true_lon
        )
        return distances.mean()


class Phase1TotalLoss(nn.Module):
    """Combined loss for Phase 1"""
    
    def __init__(
        self,
        province_weights: Dict[str, float],
        cell_metadata,
        cell_neighbors: Dict[int, list],
        device: torch.device,
        loss_weights: Optional[Dict[str, float]] = None,
        use_offset_rampup: bool = True
    ):
        super().__init__()
        
        # Default loss weights from plan.md
        if loss_weights is None:
            loss_weights = {
                'province': 0.5,
                'geocell': 1.0,
                'offset': 1.0,
                'aux': 0.1
            }
        
        self.loss_weights = loss_weights
        self.use_offset_rampup = use_offset_rampup
        
        # Initialize individual losses
        self.province_loss_fn = ProvinceLoss(province_weights, device)
        self.geocell_loss_fn = GeocellLoss(cell_metadata, cell_neighbors, device)
        
        # Sort cell metadata by cell_id for consistent indexing
        cell_meta_sorted = cell_metadata.sort_values('cell_id').reset_index(drop=True)
        self.offset_loss_fn = OffsetLoss(
            cell_meta_sorted['center_lat'].values,
            cell_meta_sorted['center_lon'].values,
            device=device
        )
        self.aux_loss_fn = AuxiliaryLoss()
        
        # Store metadata
        self.cell_metadata = cell_metadata
        self.cell_metadata_sorted = cell_meta_sorted
        self.province_names = ["Sindh", "Punjab", "Khyber Pakhtunkhwa", "ICT", 
                              "Gilgit-Baltistan", "Balochistan", "Azad Kashmir"]
        
    def forward(
        self,
        outputs: Dict,
        province_ids: torch.Tensor,
        cell_ids: torch.Tensor,
        true_lat: torch.Tensor,
        true_lon: torch.Tensor,
        epoch: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses
        
        Args:
            outputs: Model outputs dict
            province_ids: [batch_size]
            cell_ids: [batch_size]
            true_lat: [batch_size]
            true_lon: [batch_size]
        
        Returns:
            Dict with individual losses and total loss
        """
        batch_size = province_ids.shape[0]
        
        # 1. Province loss
        province_logits = outputs['province_logits']
        l_province = self.province_loss_fn(province_logits, province_ids)
        
        # 2. Geocell loss (only for ground-truth province)
        l_geocell = torch.tensor(0.0, device=province_ids.device)
        geocell_logits = outputs['geocell_logits']
        geocell_count = 0
        
        for prov_id in province_ids.unique():
            prov_id = prov_id.item()
            if prov_id in geocell_logits:
                mask = (province_ids == prov_id)
                if mask.any():
                    local_logits = geocell_logits[prov_id]['local_logits']
                    batch_cell_ids = cell_ids[mask]
                    prov_name = geocell_logits[prov_id].get('province_name', self.province_names[prov_id])
                    
                    # Compute loss for all samples in this province at once
                    l_geocell += self.geocell_loss_fn(
                        local_logits,
                        batch_cell_ids,
                        prov_id,
                        prov_name
                    )
                    geocell_count += mask.sum().item()
        
        if geocell_count > 0:
            l_geocell = l_geocell * (batch_size / geocell_count)  # Normalize by actual count
        
        # 3. Offset loss
        offsets = outputs['offsets']
        if offsets is not None:
            # Get province probabilities for weighting
            province_probs = F.softmax(province_logits, dim=1)
            l_offset = self.offset_loss_fn(
                offsets, cell_ids, true_lat, true_lon, province_probs
            )
        else:
            l_offset = torch.tensor(0.0, device=province_ids.device)
        
        # 4. Auxiliary loss
        aux_coords = outputs['aux_coords']
        l_aux = self.aux_loss_fn(aux_coords, true_lat, true_lon)
        
        # 5. Total loss with offset ramp-up
        # Offset loss ramp-up: epochs 1-2 barely matter, epochs 5+ full strength
        if self.use_offset_rampup and epoch is not None:
            offset_weight = min(1.0, epoch / 5.0)
        else:
            offset_weight = self.loss_weights['offset']
        
        l_total = (
            self.loss_weights['province'] * l_province +
            self.loss_weights['geocell'] * l_geocell +
            offset_weight * l_offset +
            self.loss_weights['aux'] * l_aux
        )
        
        # Store actual offset weight used for logging
        loss_dict = {
            'province': l_province,
            'geocell': l_geocell,
            'offset': l_offset,
            'aux': l_aux,
            'total': l_total,
            'offset_weight': offset_weight if self.use_offset_rampup and epoch is not None else self.loss_weights['offset']
        }
        
        return loss_dict
        
