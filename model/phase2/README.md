# Phase 2: Partial Vision Adaptation

## Overview

Phase 2 carefully adapts visual features without breaking the geographic structure learned in Phase 1. This is achieved by:

1. **Loading Phase 1 checkpoint** (`phase1_best.pt`) - all components initialized
2. **Unfreezing top encoder layers** - CLIP top 30%, Scene top 20%
3. **Training with different learning rates** - 5 parameter groups with different LRs
4. **Monitoring for degradation** - Red flags for rare province collapse

## 🎯 Goals

- Improve visual feature extraction while maintaining geographic structure
- Reduce validation median error from Phase 1 baseline
- Prevent rare province collapse (GB, Balochistan, AJK)
- Keep offset head functional (not cheating via cells)

## 🔧 Encoder Unfreezing Strategy

### CLIP Encoder (ViT-B/16)
- **Total blocks**: 12 transformer blocks
- **Unfreeze**: Top 30% = last ~4 blocks (blocks 8-11)
- **Keep frozen**: Patch embedding + early blocks (0-7)
- **Also unfreeze**: Final layer norm + projection

### Scene Encoder (ResNet-50 Places365)
- **Total layers**: stem, layer1, layer2, layer3, layer4
- **Unfreeze**: layer4 only (top ~20%)
- **Keep frozen**: stem, layer1-3
- **Rationale**: Scene encoder is brittle, very conservative updates

## ⚙️ Optimizer Configuration

### Parameter Groups

| Module | Learning Rate | Weight Decay |
|--------|---------------|--------------|
| Province + cell + offset heads | 5e-4 | 1e-5 |
| Cell & province embeddings | 5e-4 | 1e-5 |
| Fusion gate | 5e-4 | 1e-5 |
| CLIP (unfrozen layers) | 1e-5 | 1e-5 |
| Scene (unfrozen layers) | 5e-6 | 1e-5 |

### Optimizer
- **Type**: AdamW
- **Weight decay**: 1e-5 (same as Phase 1)

### Learning Rate Schedule (Optional)
- **Cosine decay**: Smooth decay to 0 over all epochs
- **Step decay**: Drop ×0.3 at epoch 20

## 📊 Training Configuration

### Unchanged from Phase 1
- **Batch size**: 64
- **Sampler**: Same province-balanced sampler
  - Sindh: 16, Punjab: 8, KPK: 8, ICT: 8, Balochistan: 8, GB: 8, AJK: 8
- **Loss function**: Same weights
  - L_total = 0.5 × L_province + 1.0 × L_cell + 1.0 × L_offset + 0.1 × L_aux
- **Temperature**: Same as Phase 1
- **Label smoothing**: Same province-dependent τ

### Phase 2 Specific
- **Epochs**: 30-40 (recommended: 35)
- **LR schedule**: Cosine or StepLR

## 📈 Monitoring & Success Criteria

### Primary Metrics (Must Improve or Hold)
- ✅ **Val median error (km)** - main metric
- ✅ **Val p90 error** - tail performance

### Secondary Metrics (Allowed to Fluctuate)
- ⚠️ **Province accuracy** - may dip slightly
- ⚠️ **Train losses** - may fluctuate early

### Red Flags (STOP if seen)
- ❌ **Val median error ↑ for 5+ epochs**
- ❌ **Rare provinces collapse** (GB / Baloch / AJK accuracy < 30%)
- ❌ **Offset loss → near zero** (model cheating via cells)

### Expected Behavior
- ✔ Province accuracy may drop slightly (normal)
- ✔ Cell accuracy may wobble early (normal)
- ✔ Offset loss should decrease smoothly
- ✔ Median error should improve slowly but steadily

### When to Stop
- Val median error plateaus for ~5 epochs
- Rare provinces start degrading
- Improvements < 0.05 km

## 🚀 Usage

### Local Training
```bash
# Default settings
python model/phase2/train_phase2.py

# Custom settings
python model/phase2/train_phase2.py --num-epochs 35 --batch-size 64
```

### Modal Training (GPU)
```bash
# Standard run
modal run model/phase2/train_phase2.py --num-epochs 35 --gpu A100

# Detached mode (continues after disconnect)
modal run model/phase2/train_phase2.py --num-epochs 35 --gpu A100 --detach

# Custom learning rates
modal run model/phase2/train_phase2.py \
  --num-epochs 35 \
  --lr-clip 1e-5 \
  --lr-scene 5e-6 \
  --gpu A100
```

## 📁 Output

### Checkpoints
- `checkpoints/phase2/phase2_epoch_{N}.pt` - per-epoch checkpoints
- `checkpoints/phase2/phase2_best.pt` - best model (lowest val median error)

### Logs
- `checkpoints/phase2/log.txt` - detailed training log
- `checkpoints/phase2/losses.csv` - structured metrics for visualization

### Checkpoint Contents
```python
{
    'epoch': int,
    'model_state_dict': dict,
    'optimizer_state_dict': dict,
    'losses': dict,
    'province_accuracy': dict,
    'median_error_km': float,
    'best_val_error': float,
    'temperature': float,
    'phase1_checkpoint': str,  # Path to Phase 1 checkpoint used
}
```

## 🔍 Verification

### Before Training
1. Verify Phase 1 checkpoint exists: `checkpoints/phase1/phase1_best.pt`
2. Check dataset paths are correct
3. Verify cell metadata exists

### During Training
1. Monitor val median error (should decrease)
2. Watch for red flags in logs
3. Check rare province accuracies
4. Verify offset loss is decreasing (not near zero)

### After Training
1. Compare `phase2_best.pt` vs `phase1_best.pt` median error
2. Check province accuracy distribution
3. Verify rare provinces didn't collapse
4. Analyze loss curves in `losses.csv`

## 🎓 Key Principles

1. **Conservative unfreezing**: Only top layers, very low LRs
2. **Maintain structure**: Same sampler, loss weights, temperature
3. **Monitor degradation**: Red flags for early stopping
4. **Gradual improvement**: Expect slow but steady progress

## 🚨 Troubleshooting

### Val error increasing
- Reduce encoder learning rates (lr_clip, lr_scene)
- Increase weight decay
- Stop training early

### Rare provinces collapsing
- Check sampler is working correctly
- Verify province weights are applied
- Consider stopping training

### Offset loss near zero
- Model is cheating via cell predictions
- Stop training, use earlier checkpoint

### Province accuracy dropping significantly
- Normal if < 5% drop
- Concerning if > 10% drop
- May recover in later epochs

## 📚 References

- Phase 1 training: `model/phase1/README.md`
- Model architecture: `model/phase1/geopak_phase1.py`
- Loss function: `model/phase1/losses.py`
