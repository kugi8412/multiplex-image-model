# ImmuVis Experiment Plan

End-to-end pipeline: data preparation → training → embedding extraction → downstream evaluation.

All training experiments: **200 epochs**, checkpoints every **10 epochs**, **ViT-Base** (768-dim embeddings), patch size 8.
Mamba experiments use the masked-model training script with ViM encoder blocks.

---

# Phase 0 — Data Preparation

## Exp 0a — Compute Marker Statistics

Update `configs/marker_metadata.csv` with real per-marker mean and std from training data.

```bash
python compute_marker_stats.py \
    --panels-config configs/all_panels_config.yaml \
    --tokenizer-config configs/all_markers_tokenizer.yaml \
    --metadata-csv configs/marker_metadata.csv \
    --split train
```

## Exp 0b — Prepare VirTues-Preprocessed Data

Invert arcsinh(x/5) → raw intensities, then apply full VirTues preprocessing (99th-percentile quantile clip → log1p → Gaussian blur → z-standardize) and save to new folders. Per-dataset statistics (quantiles, means, stds) are computed automatically and stored alongside the images. Skips datasets that are already preprocessed.

```bash
python prepare_virtues_data.py \
    --panel-config configs/all_panels_config.yaml \
    --output-root /net/tscratch/people/plgjgiezgala04/immuvis_split_patches_onlyarcsinh
```

Output is written to `{output_root}/{split}/{dataset}/imgs/*.npy` with stats in `{output_root}/{split}/stats/{dataset}/`. The DINO training script (`train_kronos_dinov2v3_virtues.py`) automatically detects pre-computed VirTues data and skips runtime preprocessing.

---

# Phase 1 — Training

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

## Exp 6b — ImmuKRONOS v3 (DINOv3 self-distillation)

KRONOS-ImmuVis hybrid with DINOv3: Hyperkernel + iBOT + KoLeo + SwiGLU FFN + 2D RoPE. Learnable mask tokens, register tokens. No Gram anchoring (no pretrained teacher available yet — can be enabled later by setting `use_gram_loss: true` + `gram_teacher_checkpoint`).

```bash
python train_kronos_immuvis_v3.py configs/exp6b_immukronos_v3.yaml
```

**Config:** `configs/exp6b_immukronos_v3.yaml`

---

## Exp 7 — Mamba vs ViT Comparison

All three sub-experiments share identical training hyperparameters (lr, warmup, weight decay, masking, activation, decoder) for a controlled comparison. Encoder architecture is the only variable.

### Exp 7a — Mamba Isotropic (ViM, flat 16-layer)

Flat 16-layer pure Vision Mamba (ViM) encoder at 768d. 4-way cross-scan SSM + FFN per block (no attention). Hyperkernel with kernel/stride 8. SSM expand=2. Direct comparison to Exp 7c.

```bash
python train_masked_model.py configs/exp7a_mamba_isotropic.yaml
```

**Config:** `configs/exp7a_mamba_isotropic.yaml`

### Exp 7b — Mamba Hierarchical (ViM, multi-scale)

Hierarchical Vision Mamba: 4-layer marker-agnostic at 64d (stem patch_size=2), then 4+4+4 layers at 192→384→768d with 2× downsampling between stages. SSM expand=2. Hyperkernel stride=1.

```bash
python train_masked_model.py configs/exp7b_mamba_hierarchical.yaml
```

**Config:** `configs/exp7b_mamba_hierarchical.yaml`

### Exp 7c — ViT Baseline (flat 16-layer)

ViT isotropic baseline — identical to Exp 7a but with standard ViT encoder (self-attention + MLP). Same 16 layers × 768d, same hyperkernel, decoder, and training config.

```bash
python train_masked_model.py configs/exp7c_vit_baseline.yaml
```

**Config:** `configs/exp7c_vit_baseline.yaml`

---

# Phase 2 — Embedding Extraction

Extract frozen embeddings from all trained models using `generate_embeddings.py`. Each model type produces a `(N, 768)` embedding per image patch.

| Experiment | `--model-type` | Embedding source |
|---|---|---|
| 1, 2, 3, 4 | `kronos_dinov2v3` | CLS token |
| 5a, 5b | `dino_finetune` | Global avg-pooled encoder |
| 6a | `kronos_immuvis_v2` | CLS token |
| 6b | `kronos_immuvis_v3` | CLS token |
| 7a, 7b | `mamba` | Global avg-pooled encoder |
| 7c | `mamba` | Global avg-pooled encoder |

```bash
# Example: extract embeddings for Exp 3
python generate_embeddings.py --model-type kronos_dinov2v3 \
    --config configs/exp3_kronos_dinov2_immuvis.yaml \
    --checkpoint checkpoints/exp3_best.pth \
    --input_dir /raid_encrypted/immucan/immuvis_split_patches_onlyarcsinh/test/ \
    --output_dir embeddings/exp3/

# Example: Exp 6b (ImmuKronos v3)
python generate_embeddings.py --model-type kronos_immuvis_v3 \
    --config configs/exp6b_immukronos_v3.yaml \
    --checkpoint checkpoints/exp6b_best.pth \
    --input_dir /raid_encrypted/immucan/immuvis_split_patches_onlyarcsinh/test/ \
    --output_dir embeddings/exp6b/

# Example: Exp 7a (Mamba isotropic)
python generate_embeddings.py --model-type mamba \
    --config configs/exp7a_mamba_isotropic.yaml \
    --checkpoint checkpoints/exp7a_best.pth \
    --input_dir /raid_encrypted/immucan/immuvis_split_patches_onlyarcsinh/test/ \
    --output_dir embeddings/exp7a/
```

Repeat for all experiments, adjusting `--model-type`, `--config`, and `--checkpoint`.

---

# Phase 3 — Downstream Evaluation

Two downstream tasks evaluate the quality of learned representations.

## Task A: Cell Typing (Classification)

Classify cell phenotypes from frozen embeddings. Supported methods:

| Method | Description | When to use |
|---|---|---|
| `logistic` | Logistic regression + Optuna search (KRONOS paper default) | Primary benchmark, linear separability |
| `mlp` | 2-layer MLP (BN + ReLU + Dropout) | When linear is insufficient |
| `linear_probe` | Single linear layer (SGD) | Large-scale, fast |
| `knn` | k-Nearest Neighbors (cosine) | Quick non-parametric baseline |
| `random_forest` | Random Forest | Ensemble baseline |

**Metrics:** F1 (macro/weighted), balanced accuracy, AUROC.

```bash
# Logistic regression (KRONOS default) — all experiments
python downstream_eval.py cell_typing \
    --embeddings-dir embeddings/exp3/ \
    --labels-csv data/cell_annotations.csv \
    --output-dir results/exp3_cell_typing_logistic/ \
    --method logistic --n-trials 50

# MLP classifier — for richer evaluation
python downstream_eval.py cell_typing \
    --embeddings-dir embeddings/exp6b/ \
    --labels-csv data/cell_annotations.csv \
    --output-dir results/exp6b_cell_typing_mlp/ \
    --method mlp --epochs 50 --lr 1e-3 --mlp-hidden 512

# k-NN quick baseline
python downstream_eval.py cell_typing \
    --embeddings-dir embeddings/exp1/ \
    --labels-csv data/cell_annotations.csv \
    --output-dir results/exp1_cell_typing_knn/ \
    --method knn --knn-k 20
```

## Task B: Virtual Staining

Two approaches depending on the model type:

### B1: Autoencoder models (Exp 5a/5b, 7a/7b/7c) — direct reconstruction

These models have a decoder. Use `virtual_staining.py` for leave-one-out Pearson correlation:

```bash
python virtual_staining.py configs/exp7a_mamba_isotropic.yaml \
    --checkpoint checkpoints/exp7a_best.pth \
    --output results/exp7a_virtual_stain.csv
```

### B2: Encoder-only models (Exp 1–4, 6a/6b) — MLP decoder on frozen embeddings

DINO-based models have no decoder. Train a small MLP head on frozen embeddings to predict per-channel mean intensity, then evaluate Pearson correlation:

```bash
python downstream_eval.py virtual_stain \
    --embeddings-dir embeddings/exp3/ \
    --patches-dir /raid_encrypted/immucan/immuvis_split_patches_onlyarcsinh/test/ \
    --output-dir results/exp3_virtual_stain/ \
    --method mlp --epochs 100 --lr 1e-3 \
    --tokenizer-config configs/all_markers_tokenizer.yaml \
    --skip-markers DNA1 DNA2
```

**Metrics:** Per-marker Pearson r, overall mean Pearson r.

---

# Phase 4 — Comparison Summary

Run all downstream tasks for all experiments and collect results in a single table:

| Experiment | Model type | Preprocessing | Cell typing F1 | Virtual stain Pearson |
|---|---|---|---|---|
| 1 | KRONOS DINOv2 | VirTues-Lite | | |
| 2 | KRONOS DINOv3 | VirTues-Lite | | |
| 3 | KRONOS DINOv2 | ImmuVis | | |
| 4 | KRONOS DINOv3 | ImmuVis | | |
| 5a | Finetune DINOv2 | ImmuVis | | |
| 5b | Finetune DINOv3 | ImmuVis | | |
| 6a | ImmuKRONOS v2 | ImmuVis | | |
| 6b | ImmuKRONOS v3 | ImmuVis | | |
| 7a | ViM Isotropic | ImmuVis | | |
| 7b | ViM Hierarchical | ImmuVis | | |
| 7c | ViT Baseline | ImmuVis | | |

### Key comparisons

1. **Preprocessing:** Exp 1 vs 3, Exp 2 vs 4 (VirTues-Lite vs ImmuVis)
2. **DINOv2 vs DINOv3:** Exp 1 vs 2, Exp 3 vs 4, Exp 6a vs 6b
3. **Architecture:** Exp 7a vs 7b vs 7c (ViM isotropic vs hierarchical vs ViT)
4. **Hyperkernel integration:** Exp 3 vs 5a, Exp 4 vs 5b (KRONOS DINO vs finetuned HF DINO)
5. **End-to-end vs DINO:** Exp 5a vs 6a (autoencoder vs self-distillation)

---

# Script Reference

| Script | Purpose |
|---|---|
| `compute_marker_stats.py` | Compute per-marker mean/std for normalization |
| `prepare_virtues_data.py` | Precompute VirTues-preprocessed patches |
| `train_kronos_dinov2v3_virtues.py` | KRONOS DINOv2/v3 with VirTues preprocessing |
| `train_kronos_dinov2v3.py` | KRONOS DINOv2/v3 with ImmuVis preprocessing |
| `train_kronos_immuvis_v2.py` | ImmuKRONOS v2 (Hyperkernel + DINOv2) |
| `train_kronos_immuvis_v3.py` | ImmuKRONOS v3 (Hyperkernel + DINOv3) |
| `train_masked_model.py` | Masked autoencoder (ViT/Mamba/DINO encoder) |
| `generate_embeddings.py` | Extract frozen embeddings from any model |
| `downstream_eval.py` | Cell typing + virtual staining on frozen embeddings |
| `virtual_staining.py` | Leave-one-out virtual staining (autoencoder models) |
