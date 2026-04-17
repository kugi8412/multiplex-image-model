# ImmuVis Experiment Plan

All experiments: **200 epochs**, checkpoints every **10 epochs**, **ViT-Base** (768-dim embeddings), patch size 8.
Mamba experiments use the masked-model training script with ViM encoder blocks.

---

## Exp 0a — Compute Marker Statistics

Update `configs/marker_metadata.csv` with real per-marker mean and std from training data.

```bash
python compute_marker_stats.py \
    --panels-config configs/all_panels_config.yaml \
    --tokenizer-config configs/all_markers_tokenizer.yaml \
    --metadata-csv configs/marker_metadata.csv \
    --split train
```

---

## Exp 0b — Prepare VirTues-Preprocessed Data

Invert arcsinh(x/5) → raw intensities, then apply full VirTues preprocessing (99th-percentile quantile clip → log1p → Gaussian blur → z-standardize) and save to new folders. Per-dataset statistics (quantiles, means, stds) are computed automatically and stored alongside the images. Skips datasets that are already preprocessed.

```bash
python prepare_virtues_data.py \
    --panel-config configs/all_panels_config.yaml \
    --output-suffix _virtues
```

Output is written to `{original_data_path}_virtues/{dataset}/imgs/*.npy` with stats in `{original_data_path}_virtues/stats/{dataset}/`. The DINO training script (`train_kronos_dinov2v3_virtues.py`) automatically detects pre-computed VirTues data and skips runtime preprocessing.

---

## Exp 1 — KRONOS DINOv2 + VirTues-Lite on arcsinh data

CLS-only self-distillation (no iBOT, no KoLeo). VirTues-Lite pipeline on arcsinh(x/5) data: Gaussian blur → z-standardize (marker_metadata.csv) + channel dropout. No butterworth filter, no clip normalization.

```bash
python train_kronos_dinov2v3_virtues.py configs/exp1_kronos_dinov2_virtues.yaml
```

**Config:** `configs/exp1_kronos_dinov2_virtues.yaml`

---

## Exp 2 — KRONOS DINOv3 (no Gram) + VirTues-Lite on arcsinh data

CLS + iBOT + KoLeo, SwiGLU FFN, register tokens. No Gram anchoring. VirTues-Lite pipeline on arcsinh(x/5) data.

```bash
python train_kronos_dinov2v3_virtues.py configs/exp2_kronos_dinov3_virtues.yaml
```

**Config:** `configs/exp2_kronos_dinov3_virtues.yaml`

---

## Exp 3 — KRONOS DINOv2 + ImmuVis Preprocessing

Same as Exp 1 but with ImmuVis pipeline (butterworth + clip normalization).

```bash
python train_kronos_dinov2v3.py configs/exp3_kronos_dinov2_immuvis.yaml
```

**Config:** `configs/exp3_kronos_dinov2_immuvis.yaml`

---

## Exp 4 — KRONOS DINOv3 (no Gram) + ImmuVis Preprocessing

Same as Exp 2 but with ImmuVis preprocessing.

```bash
python train_kronos_dinov2v3.py configs/exp4_kronos_dinov3_immuvis.yaml
```

**Config:** `configs/exp4_kronos_dinov3_immuvis.yaml`

---

## Exp 5a — Finetune HF DINOv2 backbone (unfrozen, MSE)

Load pretrained DINOv2 from HuggingFace, unfrozen backbone, minimize MSE reconstruction.

```bash
python train_masked_model.py configs/exp5a_finetune_dinov2.yaml
```

**Config:** `configs/exp5a_finetune_dinov2.yaml`

---

## Exp 5b — Finetune HF DINOv3 backbone (unfrozen, MSE)

Same but DINOv3 backbone.

```bash
python train_masked_model.py configs/exp5b_finetune_dinov3.yaml
```

**Config:** `configs/exp5b_finetune_dinov3.yaml`

---

## Exp 6a — ImmuKRONOS v2 (DINOv2 self-distillation)

KRONOS-ImmuVis hybrid with Hyperkernel + DINOv2 CLS distillation.

```bash
python train_kronos_immuvis_v2.py configs/exp6a_immukronos_v2.yaml
```

**Config:** `configs/exp6a_immukronos_v2.yaml`

---

## Exp 6b — ImmuKRONOS v3 (DINOv3 self-distillation)

KRONOS-ImmuVis hybrid with DINOv3 (CLS + iBOT + KoLeo).

```bash
python train_kronos_immuvis_v3.py configs/exp6b_immukronos_v3.yaml
```

**Config:** `configs/exp6b_immukronos_v3.yaml`

---

## Exp 7a — Mamba Isotropic (ViM, flat 16-layer)

Flat 16-layer Vision Mamba (ViM) encoder at 768 dimensions. Hyperkernel with kernel/stride 8.
Masked reconstruction with learnable mask tokens, evidential uncertainty.

```bash
python train_masked_model.py configs/exp7a_mamba_isotropic.yaml
```

**Config:** `configs/exp7a_mamba_isotropic.yaml`

---

## Exp 7b — Mamba Hierarchical (ViM, multi-scale)

Hierarchical Vision Mamba: 4-layer marker-aggregation at 64d, then 4+4+4 layers at 192→384→768d.
Hyperkernel with kernel/stride 1 + ViM patch_size 2.

```bash
python train_masked_model.py configs/exp7b_mamba_hierarchical.yaml
```

**Config:** `configs/exp7b_mamba_hierarchical.yaml`
