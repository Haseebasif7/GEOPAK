# 🇵🇰 GEOPAK-V3

**Province-Aware Hierarchical Mixture Geolocation Network**

**Task:**
Image → Latitude, Longitude
**Scope:** Pakistan only
**Inputs at inference:** RGB image only
**Labels available at training:** lat, lon, province

---

# 0️⃣ Core Design Philosophy (Read This Once)

This model is built on **five principles**:

1. **Hierarchy beats regression**
2. **Geography must be learned before vision is fine-tuned**
3. **Uncertainty must be modeled, not ignored**
4. **Rare regions must be protected from collapse**
5. **Offsets should never fight classification**

Everything below follows from these principles.

---

# 1️⃣ Dataset & Preprocessing (MANDATORY)

## 1.1 Dataset Schema

Each sample:

```json
{
  "id": string,
  "image": RGB image,
  "latitude": float,
  "longitude": float,
  "province": one of [
    Sindh,
    Punjab,
    Khyber Pakhtunkhwa,
    Islamabad Capital Territory,
    Gilgit-Baltistan,
    Balochistan,
    Azad Kashmir
  ]
}
```

---

## 1.2 Province-Aware Geocell Construction (OFFLINE STEP)

### Step 1 — Split by Province

Process **each province independently**.

### Step 2 — Coordinate Projection

Convert lat/lon → meters (UTM).

### Step 3 — Clustering

Use **HDBSCAN** (preferred) or **K-Means** per province.

Constraints:

* Min samples per cell: **40**
* Target radius:

  * Urban: **3–8 km**
  * Rural: **25–50 km**
* Merge undersized clusters.

### Step 4 — Save Metadata

For every cell:

```json
{
  "cell_id": int,
  "province_id": int,
  "center_lat",
  "center_lon",
  "radius_km",
  "neighbor_cell_ids"
}
```

### Target Cell Counts

| Province    | Cells |
| ----------- | ----- |
| Sindh       | 450   |
| Punjab      | 400   |
| KPK         | 220   |
| ICT         | 100   |
| GB          | 180   |
| Balochistan | 100   |
| AJK         | 80    |

**Total ≈ 1,530 cells**

---

## 1.3 Final Training Sample Fields

Each image now has:

```
image
province_id
cell_id
cell_center_latlon
```

---

# 2️⃣ Model Architecture (FULL SPEC)

---

## 2.1 Vision Encoder (Backbone)

### Backbone (Recommended)

**ViT-B/16 (CLIP-pretrained)**

Input: `3 × 224 × 224`
Output: `D = 768`

### Best Practices

* Use **LayerNorm everywhere**
* No classifier head
* Initial state: **fully frozen**

---

## 2.2 Shared Projection Block (Geographic Representation)

This block is critical. Use residuals.

```
Input: 768
↓
Linear(768 → 512)
LayerNorm
GELU
Dropout(0.2)
↓
Linear(512 → 512)
LayerNorm
↓
Residual Add
Output: 512-dim embedding
```

Call this output **E_img**

---

# 3️⃣ Head 0 — Province Classification (Top-Level Gate)

### Architecture

```
E_img (512)
↓
Linear(512 → 256)
LayerNorm
GELU
↓
Linear(256 → 7)
Softmax
```

### Loss

**Weighted Cross-Entropy**

Weights = inverse province frequency
(Balochistan, AJK, GB heavily upweighted)

---

# 4️⃣ Head 1 — Province-Gated Geocell Classification

### Design Rule

👉 **NO single Pakistan-wide classifier**

### Implementation

* One classifier **per province**
* Each classifier only sees its province’s cells

Example (Sindh):

```
Linear(512 → 512)
LayerNorm
GELU
↓
Linear(512 → N_cells_sindh)
Softmax
```

### Training

* Use **only the ground-truth province head**
* Ignore others

---

## 4.1 Distance-Aware Label Smoothing (Inside Province Only)

For true cell `c`:

```
y_i = exp( -dist_km(c, i) / τ )
```

* τ = **60 km**
* Apply only to neighbor cells
* Renormalize

### Loss

```
L_cell = KLDiv(y_soft || p_pred)
```

---

# 5️⃣ Cell & Province Embeddings

### Embeddings

```
CellEmbedding: (N_cells_total, 64)
ProvinceEmbedding: (7, 16)
```

These are **learned parameters**.

---

# 6️⃣ Head 2 — Cell-Aware Offset Refinement (Critical Precision Head)

### Input (Concatenated)

```
[E_img (512),
 CellEmbedding (64),
 ProvinceEmbedding (16)]
→ 592 dims
```

### MLP (WITH RESIDUALS)

```
Linear(592 → 256)
LayerNorm
GELU
↓
Linear(256 → 256)
LayerNorm
↓
Residual
↓
Linear(256 → 128)
GELU
↓
Linear(128 → 2) → Δlat, Δlon
```

### Constraints

* Clamp offsets to:

```
± cell_radius × province_scale
```

Province scale:

* Punjab / ICT: 0.6
* Sindh / KPK: 1.0
* GB / Balochistan: 1.4

---

# 7️⃣ Head 3 — Auxiliary Coarse Regression (TRAINING ONLY)

```
E_img
↓
Linear(512 → 256)
GELU
↓
Linear(256 → 2) → lat, lon
```

Used only to stabilize training.

---

# 8️⃣ Inference Logic (Mixture of Hypotheses)

1. Predict province probabilities
2. Select **Top-2 provinces**
3. For each province:

   * Select **Top-K = 5 cells**
4. For each cell:

   ```
   pred_i = cell_center + offset_i
   ```
5. Final output:

```
LatLon = Σ p_i × pred_i
```

This is **not argmax**. This is critical.

---

# 9️⃣ Loss Functions (Exact)

### 9.1 Province Loss

```
L_province = Weighted Cross-Entropy
```

### 9.2 Geocell Loss

```
L_cell = KLDiv
```

### 9.3 Offset Loss

```
L_offset = Σ p_i × Haversine(pred_i, GT)
```

### 9.4 Auxiliary Loss

```
L_aux = Haversine(aux_pred, GT)
```

---

## 9.5 Total Loss

```
L_total =
  0.5 × L_province
+ 1.0 × L_cell
+ 1.0 × L_offset
+ 0.1 × L_aux
```

---

# 🔟 Training Pipeline (STRICTLY FOLLOW)

---

## Phase 0 — Province Warm-Up (VERY IMPORTANT)

* Encoder: ❌ Frozen
* Train: Province head only
* Epochs: 5–8
* Target: >95% accuracy

---

## Phase 1 — Geography Structure Learning

* Encoder: ❌ Frozen
* Train:

  * Province head
  * Province geocell heads
  * Offset heads
  * Embeddings
* Epochs: 25–30
* LR: 1e-3

---

## Phase 2 — Partial Vision Adaptation

* Unfreeze top **30%** of encoder
* LR:

  * Encoder: 1e-5
  * Heads: 5e-4
* Epochs: 30–40
* **Province-balanced batches**

---

## Phase 3 — Optional Full Fine-Tune

* Only if validation improves
* Encoder LR: 5e-6
* Heads LR: 1e-4

---

# 1️⃣1️⃣ Batch Sampling (MANDATORY)

Each batch:

* Equal samples per province
* Oversample rare provinces
* Strong augmentation only for Sindh/Punjab

Without this, you **will not win**.

---

# 1️⃣2️⃣ Data Augmentation (Geography-Safe)

✅ Allowed:

* Random resized crop
* Color jitter
* Weather simulation
* Mild blur / noise
* Seasonal color shift

❌ Forbidden:

* Horizontal flip
* Large rotations
* Perspective warp

---

# 1️⃣3️⃣ Evaluation Metrics (ONLY THESE)

Report:

* Median error (km)
* 90th percentile error
* Accuracy @ 1km / 5km / 25km
* Per-province breakdown
* Urban vs rural

---

# 🏆 Final Guarantee (Honest)

If:

* Your labels are clean
* Geocells are well built
* Batches are balanced
* You follow this strictly

👉 **No global model will beat this inside Pakistan.**
👉 **No regression-only model will come close.**

---

If you want next:

* PyTorch class-by-class code skeleton
* Geocell clustering script
* Debug checklist for failure modes
* Confidence calibration & uncertainty radius

Just tell me what to generate next.
