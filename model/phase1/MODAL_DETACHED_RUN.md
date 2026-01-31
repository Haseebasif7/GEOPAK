# Running Phase 1 Training in Detached Mode on Modal

## Problem
When running training on Modal, if your local terminal disconnects (network issue, terminal closed, etc.), Modal stops the remote execution with the error:
```
Stopping app - local client disconnected. Use `modal run --detach` to keep apps
```

## Solution: Use Detached Mode

### Option 1: Using `--detach` flag (Recommended)
```bash
python model/phase1/train_phase1.py --modal --detach
```

### Option 2: Using `modal run --detach` command (Most Reliable)
```bash
modal run --detach model/phase1/train_phase1.py --modal
```

This is the **most reliable** method as it's Modal's native detached execution.

### Option 3: Using Modal CLI directly
```bash
modal run --detach model/phase1/train_phase1.py::main_modal
```

## What Detached Mode Does

- ✅ Training continues even if your local terminal disconnects
- ✅ Training continues even if you close your laptop
- ✅ Training continues even if your network drops
- ✅ You can monitor progress at: https://modal.com/apps

## Monitoring Detached Training

1. **Modal Dashboard**: Visit https://modal.com/apps to see running apps
2. **Logs**: Check logs in the Modal dashboard
3. **Checkpoints**: Checkpoints are saved to your Modal volume at `/data/checkpoints/phase1/`

## Stopping Detached Training

If you need to stop a detached training run:

1. Go to https://modal.com/apps
2. Find your running app
3. Click "Stop" or "Terminate"

## Example Full Command

```bash
modal run --detach model/phase1/train_phase1.py --modal \
  --batch-size 64 \
  --num-epochs 30 \
  --learning-rate 1e-3
```

## Notes

- Detached mode is **essential** for long training runs
- Your training will complete even if you're not connected
- Checkpoints are automatically saved to the Modal volume
- You can resume monitoring at any time via the Modal dashboard
