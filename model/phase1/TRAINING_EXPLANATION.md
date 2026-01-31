# Phase 1 Training Explanation

## Overview

Phase 1 implements **Geography Structure Learning** where the model learns the hierarchical geographic structure (province → geocell → offset) while keeping vision encoders frozen.

---

## 1. Setup & Initialization

### 1.1 Configuration
- **Epochs**: 25-30 (default: 30)
- **Learning Rate**: 1e-3
- **Batch Size**: 64
- **Weight Decay**: 1e-5
- **Temperature**: 1.0 (for province head)

### 1.2 Logging Setup
- **log.txt**: Human-readable training log
- **losses.csv**: Structured CSV for visualization (epoch, split, all losses, errors, accuracies)

### 1.3 Device Detection
- Automatically detects: CUDA → MPS → CPU
- On Modal: Always uses CUDA

---

## 2. Data Loading

### 2.1 Datasets
- **Train**: `train.csv` with required columns: `id`, `image`, `latitude`, `longitude`, `province`, `province_id`, `cell_id`
- **Validation**: `test.csv` (same structure)

### 2.2 Transforms
- **Train**: Geography-safe augmentations (RandomResizedCrop, ColorJitter, GammaCorrection, GaussianNoise)
- **Validation**: Simple resize + ToTensor (no augmentation)

### 2.3 Batch Sampling
- **Province-Balanced Batch Sampler**: Ensures each batch has balanced province distribution
  - Sindh: 16 samples (25%)
  - Others: 8 samples each (12.5%)
- **Fixed batches per epoch**: 1300 batches
- **With replacement**: Small provinces can repeat (standard for imbalanced datasets)

---

## 3. Model Initialization

### 3.1 Model Creation
```python
model = create_phase1_model(
    cell_metadata_path=cell_metadata.csv,
    freeze_clip=True,      # CLIP encoder frozen
    freeze_scene=True,      # Scene encoder frozen
    phase0_checkpoint_path=province_best.pt  # Load Phase 0 weights
)
```

### 3.2 Weight Loading from Phase 0
- **Encoder weights**: CLIP, Scene, fusion gate, projections (from Phase 0)
- **Province head weights**: Linear, LayerNorm, output layers (from Phase 0)
- **New components**: Geocell heads, offset head, embeddings (initialized from scratch)

### 3.3 Trainable Components
- ✅ Province head
- ✅ Province-gated geocell heads (one per province)
- ✅ Cell & Province embeddings
- ✅ Offset head
- ✅ Auxiliary regression head
- ✅ Fusion gate
- ✅ Projection layers
- ❌ CLIP encoder (frozen)
- ❌ Scene encoder (frozen)

---

## 4. Loss Function Setup

### 4.1 Province Weights
- Uses **effective-number weighting** (same as Phase 0)
- Formula: `E_p = (1 - β^n_p) / (1 - β)`, `w_p = 1/E_p`, normalized by mean
- Rare provinces get higher weights (Balochistan, AJK)

### 4.2 Phase1TotalLoss Components
1. **Province Loss**: Weighted Cross-Entropy (weight: 0.5)
2. **Geocell Loss**: KL Divergence with distance-aware label smoothing (weight: 1.0)
3. **Offset Loss**: Weighted Haversine (weight: 1.0, with ramp-up)
4. **Auxiliary Loss**: Haversine distance (weight: 0.1)

### 4.3 Offset Loss Ramp-Up
- **Epochs 1-2**: Weight = 0.2-0.4 (barely matters)
- **Epochs 3-4**: Weight = 0.6-0.8 (gradual increase)
- **Epochs 5+**: Weight = 1.0 (full strength)
- **Formula**: `offset_weight = min(1.0, epoch / 5)`

**Why**: Lets classification stabilize before fine-tuning offsets

---

## 5. Training Loop (Per Epoch)

### 5.1 Forward Pass
For each batch:

1. **Get inputs**:
   - `images`: [batch_size, 3, 224, 224]
   - `cell_ids`: [batch_size] - cell IDs from dataset
   - `province_ids`: [batch_size] - province IDs from dataset
   - `latitudes`, `longitudes`: [batch_size] - ground truth coordinates

2. **Model forward**:
   ```python
   outputs = model(images, cell_ids=cell_ids, province_ids=province_ids)
   ```
   
   Returns:
   - `province_logits`: [batch_size, 7] - province classification
   - `geocell_logits`: Dict with per-province logits
   - `offsets`: [batch_size, 2] - (Δlat, Δlon) predictions
   - `aux_coords`: [batch_size, 2] - auxiliary (lat, lon) predictions

3. **Loss computation**:
   ```python
   loss_dict = loss_fn(outputs, province_ids, cell_ids, latitudes, longitudes, epoch=epoch)
   ```
   
   Computes:
   - Province loss (weighted cross-entropy)
   - Geocell loss (KL divergence with distance-aware smoothing)
   - Offset loss (weighted Haversine, with ramp-up)
   - Auxiliary loss (Haversine)
   - Total loss (weighted sum)

4. **Backward pass**:
   ```python
   loss.backward()
   optimizer.step()
   ```

### 5.2 Metrics Tracking
- **Losses**: All components tracked separately
- **Province accuracy**: Per-province classification accuracy
- **Error metrics**: Median, mean, p90 error in km (using cell_center + offset)

### 5.3 Progress Bar
Shows:
- Current loss
- Individual loss components (prov, cell, off, aux)
- Offset weight (ramp-up progress)

---

## 6. Validation Loop

### 6.1 Process
- Same as training, but:
  - `model.eval()` (no gradients)
  - `torch.no_grad()` context
  - No optimizer step

### 6.2 Metrics
- Same metrics as training (losses, accuracies, errors)
- Used to select best model

---

## 7. Epoch Summary

### 7.1 Printed Metrics
```
Epoch 1/30:
   Train Loss: 2.3456 (prov: 0.1234, cell: 1.2345, off: 0.5678, aux: 0.1234)
   Val Loss:   2.1234 (prov: 0.1111, cell: 1.1111, off: 0.5000, aux: 0.1111)
   Train Error: median=45.23km, mean=52.34km, p90=78.90km
   Val Error:   median=42.11km, mean=48.76km, p90=72.34km
```

### 7.2 Province Accuracies
- Per-province accuracy for both train and validation
- Logged to both console and log.txt

### 7.3 CSV Logging
- One row per epoch per split (train/val)
- All losses, errors, and accuracies in CSV format
- Ready for visualization with pandas/matplotlib

---

## 8. Checkpointing

### 8.1 Per-Epoch Checkpoints
- **File**: `checkpoints/phase1/phase1_epoch_{epoch}.pt`
- **Contains**:
  - Model state dict
  - Optimizer state dict
  - Epoch number
  - Losses dict
  - Province accuracies
  - Median error

### 8.2 Best Model
- **File**: `checkpoints/phase1/phase1_best.pt`
- **Criterion**: Lowest validation median error
- Updated whenever validation error improves

---

## 9. Loss Computation Details

### 9.1 Province Loss
- **Type**: Weighted Cross-Entropy
- **Input**: Province logits [batch_size, 7], Province IDs [batch_size]
- **Weights**: Effective-number based (rare provinces weighted higher)

### 9.2 Geocell Loss
- **Type**: KL Divergence
- **Input**: Geocell logits (per-province), True cell IDs
- **Label Smoothing**: Distance-aware (soft labels based on geographic distance)
- **Tau values**: Province-specific (ICT: 10km, Sindh: 30km, Punjab: 60km, etc.)

### 9.3 Offset Loss
- **Type**: Weighted Haversine
- **Input**: Predicted offsets, True coordinates
- **Prediction**: `cell_center + offset`
- **Weight**: Ramp-up from 0.2 to 1.0 over first 5 epochs

### 9.4 Auxiliary Loss
- **Type**: Haversine distance
- **Input**: Direct (lat, lon) prediction from auxiliary head
- **Purpose**: Training stabilization

---

## 10. Key Features

### 10.1 Offset Clamping
- Offsets are clamped to: `± cell_radius × province_scale`
- Prevents offsets from exceeding cell boundaries
- Province scales: Punjab/ICT: 0.6, Sindh/KPK: 1.0, GB/Balochistan: 1.4

### 10.2 Distance-Aware Label Smoothing
- For geocell classification, soft labels are created based on distance
- Neighbor cells get non-zero probability
- Prevents overconfident predictions

### 10.3 Province-Gated Geocell Heads
- Each province has its own geocell classifier
- Prevents confusion between provinces
- Capacity adjustment: Small provinces (<30 cells) use reduced capacity (256-dim)

---

## 11. Training Flow Summary

```
1. Setup (device, paths, logging)
   ↓
2. Load datasets (train.csv, test.csv)
   ↓
3. Calculate province weights (effective-number)
   ↓
4. Create model (load Phase 0 weights if available)
   ↓
5. Create loss function (with offset ramp-up)
   ↓
6. Create data loaders (province-balanced sampling)
   ↓
7. Create optimizer (AdamW, lr=1e-3)
   ↓
8. For each epoch (1 to 30):
   ├─ Train epoch:
   │  ├─ For each batch:
   │  │  ├─ Forward pass (all heads)
   │  │  ├─ Compute losses (with ramp-up)
   │  │  ├─ Backward pass
   │  │  └─ Update weights
   │  └─ Calculate epoch metrics
   │
   ├─ Validate epoch:
   │  ├─ For each batch:
   │  │  ├─ Forward pass (no gradients)
   │  │  └─ Compute metrics
   │  └─ Calculate epoch metrics
   │
   ├─ Log metrics (console + file + CSV)
   ├─ Save epoch checkpoint
   └─ Save best model (if improved)
   ↓
9. Close log files
```

---

## 12. Output Files

### 12.1 Checkpoints
- `phase1_epoch_{epoch}.pt`: Checkpoint for each epoch
- `phase1_best.pt`: Best model (lowest validation median error)

### 12.2 Logs
- `log.txt`: Human-readable training log
- `losses.csv`: Structured CSV for visualization

### 12.3 CSV Format
```csv
epoch,split,loss_total,loss_province,loss_geocell,loss_offset,loss_aux,
median_error_km,mean_error_km,p90_error_km,
sindh_acc,punjab_acc,kpk_acc,ict_acc,gb_acc,balochistan_acc,ajk_acc
1,train,2.345678,0.123456,1.234567,0.567890,0.123456,45.23,52.34,78.90,95.2,92.1,88.5,...
1,val,2.123456,0.111111,1.111111,0.500000,0.111111,42.11,48.76,72.34,94.5,91.2,87.3,...
```

---

## 13. What the Model Learns

During Phase 1, the model learns:

1. **Province Classification**: Refines province predictions (initialized from Phase 0)
2. **Geocell Structure**: Learns which geocell within each province
3. **Geographic Relationships**: Distance-aware label smoothing teaches spatial relationships
4. **Offset Refinement**: Learns fine-grained offsets within cells (with ramp-up)
5. **Fusion Calibration**: Fusion gate learns when to trust scene encoder vs CLIP
6. **Embeddings**: Cell and province embeddings learn geographic representations

**Key Constraint**: Vision encoders stay frozen, so the model learns geographic structure without encoder drift.

---

## 14. Training Tips

1. **Monitor offset weight**: Should ramp from 0.2 to 1.0 over first 5 epochs
2. **Watch province accuracy**: Should improve from Phase 0 baseline
3. **Check error metrics**: Median error should decrease over time
4. **Balance losses**: All loss components should decrease (not just total)
5. **Province accuracies**: Rare provinces (Balochistan, AJK) should improve with weighted loss

---

This training process implements the hierarchical geolocation approach where the model learns to predict: **Province → Geocell → Offset**, all while keeping vision encoders frozen to prevent overfitting.
