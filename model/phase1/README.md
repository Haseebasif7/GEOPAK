# Phase 1 Model Architecture

## Overview

Phase 1 implements **Geography Structure Learning** as described in `pipeline/plan.md`.

**Key characteristics:**
- CLIP Encoder: ❌ Frozen
- Scene Encoder: ❌ Frozen
- Train: Province head, geocell heads, offset heads, embeddings, fusion gate
- Epochs: 25-30
- LR: 1e-3

## Architecture Components

### 1. Dual Encoder (CLIP + Scene with Gated Fusion)
- **CLIP**: ViT-B/16 (768-dim) → projected to 512-dim
- **Scene**: ResNet-50 Places365 (2048-dim) → projected to 512-dim
- **Fusion**: Gated fusion learns when to trust scene encoder
- Both encoders are frozen in Phase 1

### 2. Province Head
- Input: `E_img` (512-dim)
- Architecture: `Linear(512→256) → LayerNorm → GELU → Dropout → Linear(256→7)`
- Output: Province logits (7 provinces)
- Loss: Weighted Cross-Entropy with effective-number weighting

### 3. Province-Gated Geocell Heads
- **One head per province** (not a single Pakistan-wide classifier)
- **Capacity adjustment**: Smaller provinces get reduced capacity to prevent overfitting
  - If `N_cells_province < 30`: `Linear(512→256) → LayerNorm → GELU → Linear(256→N_cells)`
  - Else: `Linear(512→512) → LayerNorm → GELU → Linear(512→N_cells)`
- Loss: KL Divergence with distance-aware label smoothing
- Uses province-specific tau values for label smoothing
- **Why**: Prevents overfitting on tiny provinces (Azad Kashmir: 13 cells, GB: 20 cells) and improves calibration

### 4. Cell & Province Embeddings
- **Cell Embedding**: `(N_cells_total, 96)` - learned parameters (increased from 64)
- **Province Embedding**: `(7, 32)` - learned parameters (increased from 16)
- Used as input to offset head
- **Why larger dimensions**: Better representation capacity for geographic structure

### 5. Cell-Aware Offset Head
- Input: `[E_img (512), CellEmbedding (96), ProvinceEmbedding (32)]` = 640-dim
- **Critical feature**: LayerNorm applied to each input **before** concatenation
  - `LayerNorm(E_img)`, `LayerNorm(cell_embed)`, `LayerNorm(prov_embed)`
  - Then concatenate
  - **Why**: Makes offsets use geography earlier, cleaner gradients, faster convergence
- Architecture with residuals and dropout:
  ```
  LayerNorm(E_img) + LayerNorm(cell_emb) + LayerNorm(prov_emb)
  ↓
  Concatenate → Linear(640→256) → LayerNorm → GELU → Dropout(0.1)
  ↓
  Linear(256→256) → LayerNorm → Residual
  ↓
  Linear(256→128) → GELU
  ↓
  Linear(128→2) → (Δlat, Δlon)
  ```
- **Dropout(0.1)** after first linear: Prevents overfitting (offsets overfit faster than classification)
- Offsets are clamped: `± cell_radius × province_scale`
- Province scales: Punjab/ICT: 0.6, Sindh/KPK: 1.0, GB/Balochistan: 1.4

### 6. Auxiliary Coarse Regression Head
- Input: `E_img` (512-dim)
- Architecture: `Linear(512→256) → GELU → Linear(256→2)`
- Output: Direct (lat, lon) prediction
- Used only for training stabilization

## Loss Functions

### Total Loss with Offset Ramp-Up
```
offset_weight = min(1.0, epoch / 5)
L_total = 0.5 × L_province + 1.0 × L_cell + offset_weight × L_offset + 0.1 × L_aux
```

**Offset Loss Ramp-Up**:
- Epochs 1-2: Offsets barely matter (weight = 0.2-0.4)
- Epochs 3-4: Gradually increasing (weight = 0.6-0.8)
- Epochs 5+: Full strength (weight = 1.0)
- **Why**: Dramatically stabilizes training by letting classification stabilize before fine-tuning offsets

### Individual Losses
1. **Province Loss**: Weighted Cross-Entropy
2. **Geocell Loss**: KL Divergence with distance-aware label smoothing
3. **Offset Loss**: Weighted Haversine (weighted by province probabilities)
4. **Auxiliary Loss**: Haversine distance

## Inference

Uses **Mixture of Hypotheses**:
1. Predict province probabilities
2. Select Top-2 provinces
3. For each province: Select Top-K = 5 cells
4. For each cell: `pred_i = cell_center + offset_i`
5. Final: `LatLon = Σ p_i × pred_i` where `p_i = P(province) × P(cell|province)`

## Files

- `geopak_phase1.py`: Main model architecture
- `losses.py`: All loss functions
- `utils.py`: Utility functions (province weights, distance-aware labels, etc.)
- `inference.py`: Mixture of hypotheses inference
- `model_utils.py`: Model creation and loading helpers

## Usage

```python
from model.phase1.model_utils import create_phase1_model
from model.phase1.losses import Phase1TotalLoss
from model.phase1.utils import compute_province_weights

# Create model (automatically loads Phase 0 weights if available)
model = create_phase1_model(
    cell_metadata_path=Path("pipeline/geocells/cell_metadata.csv"),
    freeze_clip=True,
    freeze_scene=True,
    phase0_checkpoint_path=Path("checkpoints/province/province_best.pt")  # Optional
)

# Create loss function
province_weights = compute_province_weights(Path("train.csv"))
loss_fn = Phase1TotalLoss(
    province_weights=province_weights,
    cell_metadata=model.cell_metadata_df,
    cell_neighbors=model.cell_neighbors,
    device=device,
    use_offset_rampup=True  # Enable offset loss ramp-up
)

# In training loop, pass epoch for ramp-up:
losses = loss_fn(outputs, province_ids, cell_ids, true_lat, true_lon, epoch=current_epoch)
```

## Weight Initialization

**Phase 0 Weights Transfer (Recommended)**:
- **Province head**: Loaded from `checkpoints/province/province_best.pt` (Phase 0 checkpoint)
- **Encoder** (CLIP, Scene, fusion, projections): Loaded from Phase 0 checkpoint
- This provides a strong starting point rather than training from scratch

**New Components** (initialized from scratch):
- **Geocell heads**: Zero bias, small weights (std=0.01)
- **Offset head**: Zero bias, small weights (std=0.01)
- **Embeddings**: Normal initialization (std=0.01)

**If Phase 0 checkpoint not found**:
- All components start from scratch with proper initialization
- Province head: Zero bias, small weights (std=0.01)
- Fusion gate: Bias initialized to -0.4 (favors CLIP initially)

## Key Design Decisions

1. **Province-gated geocell heads**: Each province has its own classifier to avoid confusion
2. **Geocell head capacity adjustment**: Smaller provinces (< 30 cells) use reduced capacity (256-dim) to prevent overfitting
3. **Distance-aware label smoothing**: Soft labels based on geographic distance (province-specific tau)
4. **LayerNorm before concatenation**: Applied to E_img, cell_embed, and prov_embed separately before concatenation in offset head
   - Makes offsets use geography earlier
   - Cleaner gradients
   - Faster convergence
5. **Dropout in offset head**: Dropout(0.1) after first linear prevents overfitting (offsets overfit faster)
6. **Offset clamping**: Prevents offsets from exceeding cell boundaries
7. **Residual connections**: In offset head for better gradient flow
8. **Offset loss ramp-up**: Gradually increases offset weight from 0 to 1.0 over first 5 epochs
9. **Mixture of hypotheses**: Not argmax - uses probability-weighted combination
10. **Larger embeddings**: CellEmbedding (96-dim) and ProvinceEmbedding (32-dim) for better representation capacity
