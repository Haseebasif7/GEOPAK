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

======================================================================
Province distribution (for rows WITH province):
======================================================================
  Sindh: 89,252 (69.41%)
  Punjab: 24,645 (19.17%)
  Khyber Pakhtunkhwa: 5,881 (4.57%)
  Islamabad Capital Territory: 4,841 (3.76%)
  Gilgit-Baltistan: 3,125 (2.43%)
  Balochistan: 550 (0.43%)
  Azad Kashmir: 297 (0.23%)

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

## 2.1 Dual-Encoder Architecture (Complementary Representation Learning)

### Why Dual-Encoder Works

The model faces two different image regimes:

* **Scenery-rich images** → roads, mountains, skylines, landscapes → strong spatial cues
* **Scenery-poor images** → people, shops, objects, partial views → weak but contextual cues

👉 No single encoder is optimal for both.

**Solution:** Split responsibilities with complementary encoders.

| Encoder | Role |
|---------|------|
| **CLIP** | Robust general context, culture, objects, ambiguity handling |
| **Scene Encoder** | Strong geometry, layout, environment structure |

This is **complementary representation learning**, not redundancy.

---

### 2.1.1 Encoder A — CLIP (Primary, Robust)

**ViT-B/16 (CLIP-pretrained)**

* Input: `3 × 224 × 224`
* Output: `D = 768`
* Initial state: **fully frozen**
* Handles all images safely

---

### 2.1.2 Encoder B — Scene Encoder (Specialist)

Choose **ONE** of the following:

* **ResNet-50 (Places365-pretrained)**
* **ConvNeXt-Tiny (Places365-pretrained)**
* **ViT-Small (Places or ImageNet-Places hybrid)**

* Output: typically `512` or `768`
* ⚠️ This encoder is **not trusted blindly** — it's a specialist that helps when scenery is informative

---

## 2.2 Projection Before Fusion (CRITICAL)

**Never fuse raw encoder outputs directly.**

**Why?**
* Feature scales differ
* Semantics differ
* One encoder will dominate

**Correct approach:**

```
CLIP_feat (768)
↓
Linear(768 → 512)
LayerNorm
GELU
→ E_clip (512)

Scene_feat (512/768)
↓
Linear(→ 512)
LayerNorm
GELU
→ E_scene (512)
```

Now both live in **compatible geometry space** (512-dim).

---

## 2.3 Fusion Strategy

### ❌ What NOT to do

* Simple concatenation only
* Simple averaging
* Letting scene encoder dominate early

### ✅ Correct Fusion (Recommended)

**Option 1 — Gated Fusion (BEST)**

```
α = sigmoid( Linear([E_clip, E_scene]) )
E_fused = α · E_scene + (1 − α) · E_clip
```

**Interpretation:**
* If scenery is informative → trust scene encoder (α → 1)
* If image is ambiguous → fall back to CLIP (α → 0)
* This is **learned trust calibration**

**Option 2 — Residual Fusion (Simpler, still good)**

```
E_fused = E_clip + β · E_scene
```

Where β is:
* Small (e.g., initialized to 0.1)
* Learnable

CLIP stays dominant unless scene signal is strong.

---

## 2.4 Complete Feature Flow

```
Image (3 × 224 × 224)
↓
┌─────────────────┬─────────────────┐
│ CLIP Encoder    │ Scene Encoder   │
│ (frozen)        │ (frozen)         │
└────────┬────────┴────────┬─────────┘
         │                │
         ↓                ↓
    CLIP_feat (768)  Scene_feat (512/768)
         │                │
         ↓                ↓
    Proj → E_clip    Proj → E_scene
         │                │
         └────────┬────────┘
                  ↓
           Fusion Module
                  ↓
            E_img (512)
```

👉 **Everything else stays exactly the same** — province head, cell heads, offsets, losses — unchanged.

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

**Weighted Cross-Entropy with Effective-Number Weighting**

#### 3.1 Class Weights (Concrete Values)

Use **effective-number weighting**, not raw inverse frequency.

**Step-by-Step Formula:**

1. **Effective number** (per province):
   ```
   E_p = (1 − β^n_p) / (1 − β)
   ```

2. **Weight** (inverse of effective number):
   ```
   w_p = 1 / E_p
   ```

3. **Normalize by mean** (critical step):
   ```
   w_p_normalized = w_p / mean(w_p)
   ```

Where:
* `β = 0.9995` (smoothing factor)
* `n_p` = number of samples in province `p`

**Why this works:**
* Avoids exploding weights for rare provinces (AJK, Balochistan)
* More stable than raw inverse frequency
* Better generalization for imbalanced classes
* Normalization ensures weights are on a reasonable scale

**Implementation:**

```python
beta = 0.999
# Step 1: Calculate effective number per province
effective_num = (1 - beta ** n_per_province) / (1 - beta)
# Step 2: Weight is inverse of effective number
weights = 1.0 / effective_num
# Step 3: Normalize by mean (CRITICAL)
weights = weights / weights.mean()
```

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
TAU_BY_PROVINCE = {
    "ICT": 10.0,
    "Sindh": 30.0,
    "Punjab": 60.0,
    "Khyber Pakhtunkhwa": 50.0,
    "Azad Kashmir": 40.0,
    "Gilgit-Baltistan": 100.0,
    "Balochistan": 100.0,
}

tau = TAU_BY_PROVINCE[province]
y_i = exp(-distance_km(true_cell, i) / tau)
y = y / y.sum()

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
p_i​=P(province p​∣image) × P(cell c​∣image,province p​)

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

* **CLIP Encoder**: ❌ Frozen
* **Scene Encoder**: ❌ Frozen
* Train: Province head only
* **Fusion gate learns when scene helps**
* Epochs: 5–8
* Target: >95% accuracy

**Why freeze both?**
* You want calibration, not feature drift
* Fusion gate learns trust without encoder updates

---

## Phase 1 — Geography Structure Learning

* **CLIP Encoder**: ❌ Frozen
* **Scene Encoder**: ❌ Frozen
* Train:

  * Province head
  * Province geocell heads
  * Offset heads
  * Embeddings
  * **Fusion gate**
* Epochs: 25–30
* LR: 1e-3

**Now the model learns:**
* "Scene features help here, but not there"
* Geographic structure without encoder drift

---

## Phase 2 — Partial Vision Adaptation (VERY CAREFUL)

* Unfreeze:
  * Top **30%** of CLIP encoder
  * Top **20%** of Scene encoder
* LR:

  * CLIP: 1e-5
  * Scene: 5e-6 (smaller — more brittle and shortcut-prone)
  * Heads: 5e-4
* Epochs: 30–40
* **Province-balanced batches**

**Why smaller LR for scene encoder?**
* It is more brittle and shortcut-prone
* Conservative updates prevent overfitting to scene shortcuts

---

## Phase 3 — Optional Full Fine-Tune

* Only if validation improves (especially on indoor/object subset)
* CLIP Encoder LR: 5e-6
* Scene Encoder LR: 2e-6 (even more conservative)
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

