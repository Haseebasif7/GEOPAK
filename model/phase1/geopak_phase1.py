"""
GEOPAK-V3 Phase 1 Model Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
import pandas as pd

# Import encoder from province model
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from model.province.encoder import GeopakModel


class ProvinceHeadOnly(nn.Module):
    """Province classification head only (reuses logic from model.province.province_head)"""
    
    def __init__(self, fusion_dim=512, hidden_dim=256, num_provinces=7, temperature=1.0):
        super().__init__()
        self.linear = nn.Linear(fusion_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.linear_out = nn.Linear(hidden_dim, num_provinces)
        self.temperature = temperature
        
        # Initialize weights (same as model.province.province_head)
        nn.init.zeros_(self.linear_out.bias)  # Zero bias = no class preference
        nn.init.normal_(self.linear_out.weight, mean=0.0, std=0.01)  # Small weights
        
    def forward(self, e_img):
        """
        Args:
            e_img: [batch_size, fusion_dim] - already encoded features
        Returns:
            logits: [batch_size, num_provinces] - scaled by temperature
        """
        x = self.linear(e_img)
        x = self.norm(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear_out(x)
        x = x / self.temperature
        return x


class ProvinceGeocellHead(nn.Module):
    """Province-gated geocell classification head (one per province)"""
    
    def __init__(self, fusion_dim=512, num_cells=None):
        super().__init__()
        if num_cells is None:
            raise ValueError("num_cells must be specified")
        
        # Capacity adjustment: smaller provinces get reduced capacity
        if num_cells < 30:
            hidden_dim = 256
        else:
            hidden_dim = 512
        
        self.linear1 = nn.Linear(fusion_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.gelu = nn.GELU()
        self.linear_out = nn.Linear(hidden_dim, num_cells)
        
        # Initialize weights
        nn.init.zeros_(self.linear_out.bias)
        nn.init.normal_(self.linear_out.weight, mean=0.0, std=0.01)
        
    def forward(self, e_img):
        """
        Args:
            e_img: [batch_size, fusion_dim]
        Returns:
            logits: [batch_size, num_cells] - for this province's cells
        """
        x = self.linear1(e_img)
        x = self.norm(x)
        x = self.gelu(x)
        x = self.linear_out(x)
        return x


class CellAwareOffsetHead(nn.Module):
    """Cell-aware offset refinement head (Critical Precision Head)"""
    
    def __init__(self, fusion_dim=512, cell_embed_dim=96, province_embed_dim=32):
        super().__init__()
        input_dim = fusion_dim + cell_embed_dim + province_embed_dim  # 640
        
        # LayerNorm for inputs before concatenation
        self.norm_img = nn.LayerNorm(fusion_dim)
        self.norm_cell = nn.LayerNorm(cell_embed_dim)
        self.norm_prov = nn.LayerNorm(province_embed_dim)
        
        # MLP with residuals
        self.linear1 = nn.Linear(input_dim, 256)
        self.norm1 = nn.LayerNorm(256)
        self.gelu1 = nn.GELU()
        self.dropout = nn.Dropout(0.1)  # Prevents overfitting on offsets
        
        self.linear2 = nn.Linear(256, 256)
        self.norm2 = nn.LayerNorm(256)
        # Residual connection: input to linear2 output
        
        self.linear3 = nn.Linear(256, 128)
        self.gelu2 = nn.GELU()
        
        self.linear_out = nn.Linear(128, 2)  # Δlat, Δlon
        
        # Initialize weights
        nn.init.zeros_(self.linear_out.bias)
        nn.init.normal_(self.linear_out.weight, mean=0.0, std=0.01)
        
    def forward(self, e_img, cell_embed, province_embed):
        """
        Args:
            e_img: [batch_size, fusion_dim]
            cell_embed: [batch_size, cell_embed_dim]
            province_embed: [batch_size, province_embed_dim]
        Returns:
            offsets: [batch_size, 2] - (Δlat, Δlon)
        """
        # LayerNorm before concatenation (critical for early geography learning)
        e_img_norm = self.norm_img(e_img)
        cell_embed_norm = self.norm_cell(cell_embed)
        province_embed_norm = self.norm_prov(province_embed)
        
        # Concatenate normalized inputs
        x = torch.cat([e_img_norm, cell_embed_norm, province_embed_norm], dim=1)  # [batch_size, 640]
        
        # First layer
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.gelu1(x)
        x = self.dropout(x)  # Dropout to prevent overfitting
        
        # Second layer with residual (residual from before linear2)
        residual = x
        x = self.linear2(x)
        x = self.norm2(x)
        x = x + residual  # Residual connection
        
        # Third layer
        x = self.linear3(x)
        x = self.gelu2(x)
        
        # Output
        offsets = self.linear_out(x)  # [batch_size, 2]
        
        return offsets


class AuxiliaryCoarseRegressionHead(nn.Module):
    """Auxiliary coarse regression head (TRAINING ONLY)"""
    
    def __init__(self, fusion_dim=512):
        super().__init__()
        self.linear1 = nn.Linear(fusion_dim, 256)
        self.gelu = nn.GELU()
        self.linear_out = nn.Linear(256, 2)  # lat, lon
        
        # Initialize weights
        nn.init.zeros_(self.linear_out.bias)
        nn.init.normal_(self.linear_out.weight, mean=0.0, std=0.01)
        
    def forward(self, e_img):
        """
        Args:
            e_img: [batch_size, fusion_dim]
        Returns:
            coords: [batch_size, 2] - (lat, lon)
        """
        x = self.linear1(e_img)
        x = self.gelu(x)
        x = self.linear_out(x)
        return x


class GeopakPhase1Model(nn.Module):
    """
    Complete Phase 1 Model Architecture
    """
    
    def __init__(
        self,
        cell_metadata_path,
        fusion_dim=512,
        hidden_dim=256,
        cell_embed_dim=96,
        province_embed_dim=32,
        freeze_clip=True,
        freeze_scene=True,
        scene_model_path=None,
        temperature=1.0,
    ):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        # Updated embedding dimensions: CellEmbedding (N_cells, 96), ProvinceEmbedding (7, 32)
        self.cell_embed_dim = cell_embed_dim  # Default: 96
        self.province_embed_dim = province_embed_dim  # Default: 32
        
        # Load cell metadata to determine architecture
        import pandas as pd
        cell_meta = pd.read_csv(cell_metadata_path)
        
        # ============================================================
        # 1. Dual Encoder (CLIP + Scene with Gated Fusion)
        # ============================================================
        self.encoder = GeopakModel(
            fusion_dim=fusion_dim,
            freeze_clip=freeze_clip,
            freeze_scene=freeze_scene,
            scene_model_path=scene_model_path
        )
        
        # Ensure fusion gate and projections are trainable even when encoders are frozen
        if freeze_clip and freeze_scene:
            for param in self.encoder.fusion.parameters():
                param.requires_grad = True
            for param in self.encoder.clip_projection.parameters():
                param.requires_grad = True
            for param in self.encoder.scene_projection.parameters():
                param.requires_grad = True
        
        # ============================================================
        # 2. Province Head (head-only, reuses logic from model.province.province_head)
        # ============================================================
        self.province_head = ProvinceHeadOnly(
            fusion_dim=fusion_dim,
            hidden_dim=hidden_dim,
            num_provinces=7,
            temperature=temperature
        )
        
        # ============================================================
        # 3. Province-Gated Geocell Heads (one per province)
        # ============================================================
        # Count cells per province and create mapping
        self.province_cell_counts = {}
        self.province_cell_ranges = {}  # Maps province_id -> (min_cell_id, max_cell_id)
        self.local_to_global_cell_map = {}  # Maps (province_id, local_idx) -> global_cell_id
        
        provinces = ["Sindh", "Punjab", "Khyber Pakhtunkhwa", "ICT", 
                     "Gilgit-Baltistan", "Balochistan", "Azad Kashmir"]
        province_id_map = {name: i for i, name in enumerate(provinces)}
        
        self.geocell_heads = nn.ModuleDict()
        for province in provinces:
            prov_cells = cell_meta[cell_meta['province'] == province].sort_values('cell_id')
            if len(prov_cells) > 0:
                num_cells = len(prov_cells)
                prov_id = province_id_map[province]
                self.province_cell_counts[prov_id] = num_cells
                min_cell_id = int(prov_cells['cell_id'].min())
                max_cell_id = int(prov_cells['cell_id'].max())
                self.province_cell_ranges[prov_id] = (min_cell_id, max_cell_id)
                
                # Create mapping from local index to global cell_id
                global_cell_ids = prov_cells['cell_id'].values
                # Store as buffer so it moves with model to device
                buffer_name = f'local_to_global_map_{prov_id}'
                self.register_buffer(buffer_name, torch.tensor(global_cell_ids, dtype=torch.long))
                # Store buffer name in dict for easy access
                self.local_to_global_cell_map[prov_id] = buffer_name
                
                self.geocell_heads[province] = ProvinceGeocellHead(
                    fusion_dim=fusion_dim,
                    num_cells=num_cells
                )
        
        # ============================================================
        # 4. Cell & Province Embeddings
        # Updated dimensions: CellEmbedding (N_cells, 96), ProvinceEmbedding (7, 32)
        # ============================================================
        total_cells = len(cell_meta)
        self.cell_embedding = nn.Embedding(total_cells, cell_embed_dim)  # (N_cells, 96)
        self.province_embedding = nn.Embedding(7, province_embed_dim)  # (7, 32)
        
        # Initialize embeddings
        nn.init.normal_(self.cell_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.province_embedding.weight, mean=0.0, std=0.01)
        
        # ============================================================
        # 5. Cell-Aware Offset Head
        # ============================================================
        self.offset_head = CellAwareOffsetHead(
            fusion_dim=fusion_dim,
            cell_embed_dim=cell_embed_dim,
            province_embed_dim=province_embed_dim
        )
        
        # ============================================================
        # 6. Auxiliary Coarse Regression Head
        # ============================================================
        self.aux_head = AuxiliaryCoarseRegressionHead(fusion_dim=fusion_dim)
        
        # Store cell metadata for inference and loss computation
        # Sort by cell_id to ensure consistent indexing
        cell_meta_sorted = cell_meta.sort_values('cell_id').reset_index(drop=True)
        self.register_buffer('cell_centers_lat', torch.tensor(cell_meta_sorted['center_lat'].values, dtype=torch.float32))
        self.register_buffer('cell_centers_lon', torch.tensor(cell_meta_sorted['center_lon'].values, dtype=torch.float32))
        self.register_buffer('cell_radii_km', torch.tensor(cell_meta_sorted['radius_km'].values, dtype=torch.float32))
        self.register_buffer('cell_province_ids', torch.tensor(cell_meta_sorted['province_id'].values, dtype=torch.long))
        
        # Store neighbor information for distance-aware label smoothing
        self.cell_neighbors = {}
        for _, row in cell_meta_sorted.iterrows():
            cell_id = int(row['cell_id'])
            neighbors_str = row.get('neighbor_cell_ids', '')
            if pd.notna(neighbors_str) and str(neighbors_str).strip():
                neighbors = [int(x) for x in str(neighbors_str).split(',') if str(x).strip()]
                self.cell_neighbors[cell_id] = neighbors
            else:
                self.cell_neighbors[cell_id] = []
        
        # Province scales for offset clamping
        self.province_scales = {
            0: 1.0,   # Sindh
            1: 0.6,   # Punjab
            2: 1.0,   # Khyber Pakhtunkhwa
            3: 0.6,   # ICT
            4: 1.4,   # Gilgit-Baltistan
            5: 1.4,   # Balochistan
            6: 1.0,   # Azad Kashmir (default to 1.0)
        }
        
        # Store province names for mapping
        self.province_names = provinces
        self.province_id_to_name = {i: name for i, name in enumerate(provinces)}
        
        # Store full cell metadata for utilities
        self.cell_metadata_df = cell_meta_sorted.copy()
        
    def forward(self, images, cell_ids=None, province_ids=None, return_all=False):
        """
        Forward pass
        
        Args:
            images: [batch_size, 3, 224, 224]
            cell_ids: [batch_size] - cell IDs for offset head (optional, for training)
            province_ids: [batch_size] - province IDs for geocell head (optional, for training)
            return_all: If True, return all intermediate outputs
        
        Returns:
            If return_all=False:
                dict with keys: 'province_logits', 'geocell_logits', 'offsets', 'aux_coords'
            If return_all=True:
                Also includes 'e_img' (fused features)
        """
        batch_size = images.shape[0]
        
        # 1. Get fused image features
        e_img = self.encoder(images)  # [batch_size, fusion_dim]
        
        # 2. Province classification
        province_logits = self.province_head(e_img)  # [batch_size, 7]
        
        # 3. Province-gated geocell classification
        geocell_logits = {}
        if province_ids is not None:
            # During training: use ground-truth province to select head
            for prov_id in province_ids.unique():
                prov_id = prov_id.item()
                prov_name = self.province_names[prov_id]
                if prov_name in self.geocell_heads:
                    mask = (province_ids == prov_id)
                    if mask.any():
                        # Get local logits for this province
                        local_logits = self.geocell_heads[prov_name](e_img[mask])  # [n_samples, num_cells_province]
                        geocell_logits[prov_id] = {
                            'local_logits': local_logits,
                            'mask': mask,
                            'province_id': prov_id,
                            'province_name': prov_name
                        }
        else:
            # During inference: run all heads (will be weighted by province probs)
            for prov_name, head in self.geocell_heads.items():
                prov_id = self.province_names.index(prov_name)
                local_logits = head(e_img)  # [batch_size, num_cells_province]
                geocell_logits[prov_id] = {
                    'local_logits': local_logits,
                    'province_id': prov_id,
                    'province_name': prov_name
                }
        
        # 4. Get embeddings for offset head
        if cell_ids is not None:
            cell_embeds = self.cell_embedding(cell_ids)  # [batch_size, cell_embed_dim]
            province_embeds = self.province_embedding(province_ids)  # [batch_size, province_embed_dim]
            
            # 5. Cell-aware offset prediction
            offsets = self.offset_head(e_img, cell_embeds, province_embeds)  # [batch_size, 2]
            
            # 6. Clamp offsets based on cell radius and province scale
            cell_radii = self.cell_radii_km[cell_ids]  # [batch_size]
            province_scales = torch.tensor(
                [self.province_scales[pid.item()] for pid in province_ids],
                device=images.device, dtype=torch.float32
            )
            max_offset_km = cell_radii * province_scales  # [batch_size]
            
            # Convert km to degrees
            cell_lats = self.cell_centers_lat[cell_ids]  # [batch_size]
            max_offset_lat = max_offset_km / 111.0  # [batch_size]
            max_offset_lon = max_offset_km / (111.0 * torch.cos(torch.deg2rad(cell_lats)))  # [batch_size]
            
            # Clamp offsets (use torch.clamp which returns new tensor, not in-place)
            offsets = torch.stack([
                torch.clamp(offsets[:, 0], -max_offset_lat, max_offset_lat),
                torch.clamp(offsets[:, 1], -max_offset_lon, max_offset_lon)
            ], dim=1)  # [batch_size, 2]
        else:
            offsets = None
        
        # 6. Auxiliary coarse regression
        aux_coords = self.aux_head(e_img)  # [batch_size, 2]
        
        outputs = {
            'province_logits': province_logits,
            'geocell_logits': geocell_logits,
            'offsets': offsets,
            'aux_coords': aux_coords,
        }
        
        if return_all:
            outputs['e_img'] = e_img
        
        return outputs
