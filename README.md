# ImmuVis — Multiplex Image Foundation Model

A multi-panel, marker-agnostic autoencoder for multiplex proteomics imaging (IMC, CODEX, MIBI, etc.) with Hyperkernel architecture, evidential uncertainty, and self-supervised representation learning.

## Overview

```
                    ┌─────────────────────────────────────┐
   Multiplex Image  │  Hyperkernel (marker → embedding)   │
   (B, C, H, W)    │  ↓                                  │
   C varies per     │  Pan-Marker Encoder                 │
   panel            │  (ConvNeXt / ViT / Swin / MambaSwin)│
                    │  ↓                                  │
                    │  Latent Z: (B, D, H', W')           │
                    │  ↓                                  │
                    │  Hyperkernel Decoder → PixelShuffle  │
                    │  ↓                                  │
                    │  Output: (B, C, H, W, 4)            │
                    │  γ, ν, α, β (evidential NIG)        │
                    └─────────────────────────────────────┘
```

The model accepts **any number of marker channels** because the Hyperkernel dynamically generates per-marker convolution weights from a shared marker embedding table. This enables training on 28+ heterogeneous panels simultaneously.

---

## Installation

```bash
pip install -e .
```

This installs all dependencies from `pyproject.toml` (PyTorch 2.7, torchvision, comet-ml, pydantic, scikit-image, etc.).

---

## Data Preparation

Organize your data in the following directory structure:

```
data/
├── train/
│   ├── dataset1/
│   │   └── imgs/
│   │       ├── sample_001.npy    # shape: (C, H, W) per panel
│   │       └── sample_002.npy
│   └── dataset2/
│       └── imgs/
└── test/
    ├── dataset1/
    │   └── imgs/
    └── dataset2/
        └── imgs/
```

Each `.npy` (or `.tiff`) file is a single multiplex image patch with shape `(C, H, W)` where `C` is the number of markers for that panel.

### Panel Configuration (`configs/all_panels_config.yaml`)

```yaml
paths:
  train: /path/to/data/train
  test: /path/to/data/test

datasets:
  - dataset1
  - dataset2

clip_limits:          # per-dataset normalization upper bounds
  dataset1: 4.9
  dataset2: 3.7

markers:              # ordered marker names per dataset (must match channel order)
  dataset1:
    - CD3
    - CD8a
    - PDL1
    - ...
  dataset2:
    - CD20
    - CD68
    - ...
```

### Marker Tokenizer (`configs/all_markers_tokenizer.yaml`)

A flat `{marker_name: token_id}` mapping for all markers across all panels:

```yaml
CD3: 0
CD8a: 1
CD20: 2
PDL1: 3
# ... up to N markers
```

---

## Training

### 1. ImmuVis Masked Autoencoder (Reconstruction)

The core training script that trains the Hyperkernel autoencoder with channel masking, spatial masking, and evidential uncertainty:

```bash
python train_masked_model.py configs/train_vit_config.yaml
```

Available encoder configs:

| Config | Encoder | Latent dim | Notes |
|---|---|---|---|
| `train_vit_config.yaml` | ViT-16 | 512 | Isotropic transformer |
| `train_swin_config.yaml` | Swin Transformer | hierarchical | Shifted window attention |
| `train_mambaswin_config.yaml` | MambaSwin | 96→768 | 4-way SSM + window attention, rotation invariant |
| `train_dino_finetune.yaml` | DINOv3-Base (timm) | 768 | Pretrained backbone, finetuned |
| `train_dino_finetune_light.yaml` | DINOv3-Base | 768 | 1 decoder block (lighter) |
| `train_dino_finetune_light_patch8.yaml` | DINOv3-Base (patch 8) | 768 | Finer spatial resolution |

**Key config fields:**

```yaml
# Masking
mask_strategy: learnable          # zero | negative | learnable
output_activation: sigmoswish     # sigmoid | hard_sigmoid | sigmoswish (recommended)
activation_beta: 2.0

# Uncertainty
uncertainty_method: evidential    # evidential (4 outputs: γ,ν,α,β) | beta_nll (2 outputs: μ,logσ²)
evidence_reg_coeff: 0.01
```

**Resume from checkpoint:**

```bash
python train_masked_model.py configs/train_vit_config.yaml
# (set from_checkpoint: "checkpoints/last_checkpoint-<run_name>.pth" in config)
```

### 2. KRONOS DINO — Self-Supervised Representation Learning (V2)

DINOv2 self-distillation using the KRONOS ViT backbone with marker-aware patch embedding:

```bash
python train_kronos_dino.py configs/train_kronos_config.yaml
python train_kronos_dino.py configs/train_kronos_config.yaml --from-checkpoint checkpoints/kronos_dino-epoch_50.pth
```

This produces a student encoder that maps multiplex patches to rich CLS-token and patch-token features for downstream tasks. Uses multi-crop augmentation (2 global 128×128 + 8 local 64×64).

### 3. ImmunoKronos V3 — DINOv3 with iBOT + KoLeo + Gram Anchoring

The full DINOv3 pipeline with marker-aware Hyperkernel, RoPE, SwiGLU FFN, register tokens, iBOT patch-level distillation, and KoLeo diversity loss:

```bash
python train_kronos_immuvis_v3.py configs/train_immukronos_config.yaml
```

Key additions over V2:
- **iBOT**: Patch-level self-distillation with learnable mask tokens
- **KoLeo**: Uniform hypersphere regularizer preventing representation collapse  
- **Register tokens**: Extra CLS-like tokens that absorb global information
- **Gram anchoring**: Optional Stage-2 distillation from a frozen Stage-1 teacher

### 4. PixelCNN — Latent Space Density Estimation

Train a PixelCNN on frozen encoder latent representations for anomaly detection and counterfactual reasoning:

```bash
python train_pixelcnn.py configs/train_mambaswin_config.yaml \
    --encoder-checkpoint checkpoints/final_model.pth \
    --distribution evidential \
    --epochs 100 \
    --cache-latents latents/mambaswin_latents.pt
```

Three distribution heads: `gaussian`, `discretized_logistic_mixture`, `evidential`.

---

## Evaluation

### Virtual Staining (Leave-One-Out Pearson Correlation)

For each test image and each marker, remove that marker from the input and predict it from the remaining markers. Measure Pearson correlation with the ground truth:

```bash
python virtual_staining.py configs/train_vit_config.yaml \
    --checkpoint checkpoints/final_model.pth \
    --output results/virtual_staining_pearson.csv \
    --save-recons \
    --recons-dir results/reconstructions/
```

**Output:** A CSV with columns `id, marker, pearson`. The script also prints per-marker summary statistics:

```
Per-marker Pearson (mean ± std):
                     CD3: 0.8234 ± 0.0512  (n=1200)
                    CD8a: 0.7891 ± 0.0623  (n=1200)
                    PDL1: 0.6543 ± 0.1021  (n=800)
...
Overall mean Pearson: 0.7456
```

**Evaluate any trained model** — just point `--checkpoint` at your model and use the matching config:

```bash
# Evaluate a MambaSwin model
python virtual_staining.py configs/train_mambaswin_config.yaml \
    --checkpoint checkpoints/mambaswin_final.pth \
    --output results/mambaswin_pearson.csv

# Evaluate a DINO-finetuned model
python virtual_staining.py configs/train_dino_finetune_light.yaml \
    --checkpoint checkpoints/dino_light_final.pth \
    --output results/dino_light_pearson.csv
```

**What to look for:**
- Per-marker Pearson > 0.7 indicates reliable virtual staining for that marker
- Structural markers (DNA1, DNA2) are skipped by default (`--skip-markers DNA1 DNA2`)
- Save reconstructions with `--save-recons` for visual inspection

### Evaluating ImmunoKronos V2/V3 on Downstream Clinical Tasks

After training KRONOS DINO (V2) or ImmunoKronos (V3), use the downstream evaluation pipeline in the tutorials:

#### Step 1: Load the Trained Model

```python
# For ImmunoKronos V3 (train_kronos_immuvis_v3.py output)
from multiplex_model.kronos.tutorials.immukronos import load_immukronos_model

model = load_immukronos_model(
    checkpoint_path="checkpoints/immukronos_v3-final.pth",
    num_markers=512,       # len(TOKENIZER)
    embed_dim=384,
    depth=12,
    num_heads=6,
    patch_size=8,
    out_dim=65536,
    num_register_tokens=4,
    ibot_out_dim=8192,
    device="cuda",
)

# For KRONOS DINO V2 (train_kronos_dino.py output)
from multiplex_model.kronos.vision_transformer import vit_small
import torch

backbone = vit_small(patch_size=16, num_markers=512)
ckpt = torch.load("checkpoints/kronos_dino-final.pth", weights_only=True)
backbone.load_state_dict(ckpt["student_state_dict"], strict=False)
backbone.eval().cuda()

# Extract features: (patch_features, marker_features, spatial_features)
patch_feat, marker_feat, spatial_feat = backbone(image, marker_ids=marker_ids_list)
```

#### Step 2: Feature Extraction

```python
from multiplex_model.kronos.tutorials.immukronos import (
    FeatureExtractor, ImmuVisFeatureExtractor, PatchDataset
)

# Extract features for a directory of patches
extractor = ImmuVisFeatureExtractor(model, device="cuda")
features = extractor.extract_from_directory(
    patch_dir="data/test/dataset1/imgs/",
    marker_ids=marker_ids,
    batch_size=32,
)
# features: dict with 'cls_tokens', 'patch_tokens', 'filenames'
```

#### Step 3: Cell Phenotyping

Classify cell types from marker expression at single-cell resolution:

```python
from multiplex_model.kronos.tutorials.immukronos import CellPhenotyping

phenotyper = CellPhenotyping(model, marker_names=["CD3", "CD8a", "CD20", "CD68", ...])
cell_labels = phenotyper.predict(image, cell_masks, marker_ids)
# → assigns phenotype labels (T-cell, B-cell, Macrophage, ...) per cell
```

See notebook: `multiplex_model/kronos/tutorials/2 - Cell-phenotyping.ipynb`

#### Step 4: Patch Phenotyping

Classify tissue microenvironment at the patch level:

```python
from multiplex_model.kronos.tutorials.immukronos import PatchPhenotyping

patch_classifier = PatchPhenotyping(model)
patch_labels = patch_classifier.classify(features)
# → tissue type per patch (tumor, stroma, immune-rich, necrosis, ...)
```

See notebook: `multiplex_model/kronos/tutorials/3 - Patch-phenotyping.ipynb`

#### Step 5: Patient Stratification (Clinical Outcome)

Multiple Instance Learning (MIL) on patient-level bags of patch features for survival prediction or treatment response:

```python
from multiplex_model.kronos.tutorials.immukronos import (
    PatientStratification, BagModel, MilDataset
)

# Build MIL bags from patch features
mil_dataset = MilDataset(
    patient_features=patient_patch_features,  # {patient_id: (N_patches, D)}
    labels=patient_labels,                     # {patient_id: 0/1}
)

# Train attention-based MIL classifier
bag_model = BagModel(input_dim=384, hidden_dim=128, n_classes=2)
stratifier = PatientStratification(bag_model, device="cuda")
stratifier.train(mil_dataset, epochs=50, lr=1e-4)

# Evaluate
auc, predictions = stratifier.evaluate(test_mil_dataset)
print(f"Patient-level AUC: {auc:.4f}")
```

See notebook: `multiplex_model/kronos/tutorials/7 - Patient-stratification.ipynb`

#### Step 6: Tissue Search (Retrieval)

Find similar tissue samples using cosine + Frobenius distance on encoder embeddings:

```python
from multiplex_model.utils import TissueSearchEngine

# Extract embeddings (works with any ImmuVis or KRONOS model)
engine = TissueSearchEngine(alpha=0.7)  # 0.7=70% cosine, 30% Frobenius

# Option A: From pre-saved .npy embeddings
engine.build_index(support_dir="embeddings/support/")
results = engine.query(query_dir="embeddings/query/", topk=5)
results.to_csv("tissue_search_results.csv")

# Option B: Extract directly from model + dataloader
filenames, embeddings = TissueSearchEngine.extract_embeddings(
    model, dataloader, device="cuda", output_dir="embeddings/support/"
)
engine.build_index(filenames=filenames, embeddings=embeddings)
matches = engine.query_single(query_embedding, name="query_001", topk=5)
```

See notebook: `multiplex_model/kronos/tutorials/6 - Tissue-search.ipynb`

---

## Explainability (XAI)

Three complementary XAI approaches in `xai/`:

### 1. Attribution — Per-Marker Saliency Maps

Which input markers drive the prediction for a target marker?

```bash
python -m xai.attribution \
    --config configs/train_mambaswin_config.yaml \
    --checkpoint checkpoints/final_model.pth \
    --target-marker PDL1 \
    --method integrated_gradients \
    --output-dir xai_results/attribution/
```

Methods: `gradient_x_input`, `integrated_gradients`, `uncertainty_weighted_ig`

### 2. Counterfactual Perturbation — Virtual Marker Knockout

Remove markers and optimize remaining pixels to maximize/minimize a target:

```bash
python -m xai.counterfactual \
    --config configs/train_mambaswin_config.yaml \
    --checkpoint checkpoints/final_model.pth \
    --target-markers PDL1 PD1 \
    --steps 300 \
    --output-dir xai_results/counterfactual/
```

### 3. Latent Likelihood — PixelCNN-Based Anomaly Detection

Requires a trained PixelCNN on the frozen encoder's latent space:

```bash
# Surprise maps (spatial anomaly detection)
python -m xai.latent_likelihood \
    --config configs/train_mambaswin_config.yaml \
    --encoder-checkpoint checkpoints/encoder.pth \
    --pixelcnn-checkpoint checkpoints/pixelcnn_best.pth \
    --image data/test/dataset1/imgs/sample.npy \
    --method surprise \
    --output-dir xai_results/surprise/

# Marker dependency graph
python -m xai.latent_likelihood ... --method dependency

# Latent uncertainty decomposition (evidential PixelCNN)
python -m xai.latent_likelihood ... --method uncertainty

# Counterfactual latent sampling
python -m xai.latent_likelihood ... --method counterfactual --n-samples 5 --decode
```

---

## Cross-Sparse Autoencoders

Dictionary learning on frozen latent representations to compare models and
interpret marker-level behaviour.  Supports comparing ImmuVis models against
each other or against [VirTues](https://github.com/bunnelab/virtues) (Wenckstern et al., 2025).

### Train SAE on a Single Model

```bash
python -m cross_sae.train_sae \
    --source immuvis \
    --config configs/train_vit_config.yaml \
    --checkpoint checkpoints/vit_final.pth \
    --sae-variant topk --k 32 --expansion 8 \
    --epochs 30 --output-dir sae_models/immuvis_vit/
```

Three SAE variants: `vanilla` (L1), `topk` (exact sparsity), `gated` (selection/magnitude split).

### Cross-SAE: Compare Two Models

Train paired SAEs and run cross-activation analysis to identify shared and unique features:

```bash
python -m cross_sae.train_cross_sae \
    --config-a configs/train_vit_config.yaml \
    --checkpoint-a checkpoints/vit_final.pth \
    --config-b configs/train_mambaswin_config.yaml \
    --checkpoint-b checkpoints/mambaswin_final.pth \
    --sae-variant topk --k 32 \
    --output-dir cross_sae_results/
```

Outputs: cross-activation scatter plots, decoder cosine similarity heatmaps,
shared/unique feature counts.

### Interpretation

```bash
python -m cross_sae.interpret \
    --config configs/train_vit_config.yaml \
    --model-checkpoint checkpoints/vit_final.pth \
    --sae-checkpoint sae_models/immuvis_vit/sae_checkpoint.pth \
    --output-dir interpretation_results/
```

Analyses: feature–marker attribution (leave-one-out), marker variance decomposition,
predictive feature identification (correlation with per-marker Pearson quality).

### Spatial Visualisation

```bash
python -m cross_sae.visualize \
    --config configs/train_vit_config.yaml \
    --model-checkpoint checkpoints/vit_final.pth \
    --sae-checkpoint sae_models/immuvis_vit/sae_checkpoint.pth \
    --image-idx 0 5 10 \
    --output-dir sae_visualizations/
```

---

## Project Structure

```
├── train_masked_model.py          # ImmuVis masked autoencoder training
├── train_kronos_dino.py           # KRONOS DINO V2 self-distillation
├── train_kronos_immuvis_v2.py     # ImmunoKronos V2 (DINOv2 baseline)
├── train_kronos_immuvis_v3.py     # ImmunoKronos V3 (DINOv3 + iBOT + KoLeo)
├── train_pixelcnn.py              # PixelCNN on frozen latent space
├── virtual_staining.py             # Virtual staining evaluation (Pearson)
├── receptive_field_saliency.py     # Effective receptive field analysis
│
├── configs/
│   ├── all_panels_config.yaml     # Panel definitions (paths, markers, clip_limits)
│   ├── all_markers_tokenizer.yaml # Marker → token ID mapping
│   ├── train_vit_config.yaml      # ViT encoder config
│   ├── train_swin_config.yaml     # Swin Transformer config
│   ├── train_mambaswin_config.yaml # MambaSwin (rotation-invariant) config
│   ├── train_dino_finetune*.yaml  # DINOv3 backbone finetuning configs
│   ├── train_kronos_config.yaml   # KRONOS DINO V2 config
│   └── train_immukronos_config.yaml # ImmunoKronos V3 config
│
├── multiplex_model/
│   ├── data.py                    # DatasetFromTIFF, PanelBatchSampler
│   ├── losses.py                  # NLL, β-NLL, evidential, Swishoid, LearnableOutputActivation
│   ├── modules/
│   │   ├── immuvis.py             # Hyperkernel, Encoder, Decoder, Autoencoder
│   │   ├── convnext.py            # ConvNeXt blocks and encoder
│   │   ├── vit.py                 # Vision Transformer blocks and encoder
│   │   ├── swin.py                # Swin Transformer blocks and encoder
│   │   ├── mamba.py               # MambaSwin (pure PyTorch, no mamba_ssm dep)
│   │   ├── dino.py                # DINOv2/v3 encoder via timm
│   │   ├── resnet.py              # ResNet blocks and encoder
│   │   ├── kronos.py              # KRONOS hybrid encoder
│   │   ├── pixelcnn.py            # PixelCNN for latent density estimation
│   │   └── registry.py            # BLOCK_REGISTRY, ENCODER_REGISTRY
│   ├── utils/
│   │   ├── configuration.py       # Pydantic TrainingConfig
│   │   ├── masking.py             # Channel + spatial masking
│   │   ├── optim.py               # Scheduler, ClampWithGrad
│   │   ├── train_logging.py       # Comet.ml logging utilities
│   │   └── tissue_search.py       # TissueSearchEngine (KD-Tree + Frobenius)
│   └── kronos/                    # KRONOS ViT backbone (from KRONOS paper)
│       ├── vision_transformer.py  # DinoVisionTransformer with marker embedding
│       ├── dino_head.py           # DINO projection head
│       ├── patch_embed.py         # Per-channel patch embedding
│       ├── block.py               # Transformer block with NestedTensor support
│       ├── attention.py           # Standard + MemEfficient attention
│       ├── inference.py           # Model loading utilities
│       └── tutorials/             # Downstream task notebooks
│           ├── 1 - Data-Download-And-Preprocessing.ipynb
│           ├── 2 - Cell-phenotyping.ipynb
│           ├── 3 - Patch-phenotyping.ipynb
│           ├── 4 - Region-and-artifact-detection.ipynb
│           ├── 5 - Unsupervised-tissue-phenotyping.ipynb
│           ├── 6 - Tissue-search.ipynb
│           ├── 7 - Patient-stratification.ipynb
│           └── immukronos/        # Downstream task scripts
│               ├── inference.py
│               ├── feature_extraction.py
│               ├── cell_phenotyping.py
│               ├── patch_phenotyping.py
│               ├── region_artifact_detection.py
│               └── patient_stratification.py
│
└── xai/
    ├── attribution.py             # Integrated Gradients, Gradient×Input
    ├── counterfactual.py          # Virtual marker knockout
    ├── latent_likelihood.py       # PixelCNN-based XAI (surprise, dependency, ...)
    └── xai.ipynb                  # Interactive XAI notebook
│
└── cross_sae/
    ├── sparse_autoencoder.py      # VanillaSAE, TopKSAE, GatedSAE
    ├── train_sae.py               # Train SAE on ImmuVis or VirTues latents
    ├── train_cross_sae.py         # Cross-SAE: compare two models
    ├── interpret.py               # Feature–marker attribution, variance, prediction
    └── visualize.py               # Spatial activation maps, feature dashboards
```

---

## Key Concepts

### Evidential Uncertainty (Normal-Inverse-Gamma)

The model predicts 4 parameters per pixel per marker:

| Parameter | Meaning |
|---|---|
| $\gamma$ | Predicted intensity (mean) |
| $\nu$ | Evidence — virtual observation count |
| $\alpha$ | Inv-Gamma shape |
| $\beta$ | Inv-Gamma rate |

Uncertainty decomposition:
- **Aleatoric** (data noise): $\frac{\beta}{\alpha - 1}$
- **Epistemic** (model uncertainty): $\frac{\beta}{\nu(\alpha - 1)}$
- **Total**: $\frac{\beta(1 + \nu)}{\nu(\alpha - 1)}$

### SigmoSwish Activation

The default output activation `sigmoswish(x) = clamp(x · σ(βx), 0, 1)` where `β=2.0`. Unlike sigmoid, it satisfies `f(0) = 0` exactly — critical for zero-inflated mass spectrometry data where true zeros are common.

### Learnable Mask Token

When `mask_strategy: learnable`, masked spatial regions are replaced by a learnable parameter in the encoder's feature space (after the marker-agnostic encoder, before the Hyperkernel). This provides an unambiguous "masked" signal vs. zero-fill which is indistinguishable from genuine background.

---

## Complete Evaluation Workflow Example

```bash
# 1. Train a MambaSwin autoencoder (reconstruction + evidential uncertainty)
python train_masked_model.py configs/train_mambaswin_config.yaml

# 2. Evaluate virtual staining quality
python virtual_staining.py configs/train_mambaswin_config.yaml \
    --checkpoint checkpoints/final_model.pth \
    --output results/mambaswin_virtual_staining.csv \
    --save-recons --recons-dir results/mambaswin_recons/

# 3. Train ImmunoKronos V3 for representation learning
python train_kronos_immuvis_v3.py configs/train_immukronos_config.yaml

# 4. Train PixelCNN on frozen encoder latents
python train_pixelcnn.py configs/train_mambaswin_config.yaml \
    --encoder-checkpoint checkpoints/final_model.pth \
    --distribution evidential \
    --cache-latents latents/mambaswin.pt

# 5. Run XAI analyses
python -m xai.attribution \
    --config configs/train_mambaswin_config.yaml \
    --checkpoint checkpoints/final_model.pth \
    --target-marker PDL1 --method integrated_gradients \
    --output-dir results/xai/

python -m xai.latent_likelihood \
    --config configs/train_mambaswin_config.yaml \
    --encoder-checkpoint checkpoints/final_model.pth \
    --pixelcnn-checkpoint checkpoints/pixelcnn_best.pth \
    --image data/test/dataset1/imgs/sample.npy \
    --method surprise --output-dir results/xai/surprise/
```

---

## License

MIT

