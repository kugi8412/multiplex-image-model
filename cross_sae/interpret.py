"""Marker-level interpretation of SAE features.

Analyses performed:

1. **Feature–marker attribution** — For each SAE feature, measure how its
   activation changes when each input marker is removed (leave-one-out).
   Features that respond strongly to a specific marker's removal are
   "encoding" that marker's signal.

2. **Marker variance explained** — Decompose the variance of each marker's
   contribution to the latent space across SAE features.  Shows which
   dictionary elements capture each marker's variation.

3. **Predictive feature identification** — For each marker, rank SAE features
   by their correlation with the reconstruction quality (Pearson of that
   marker).  Identifies which latent features drive accurate virtual staining.

4. **Cross-model marker comparison** — Given two SAE checkpoints (from
   cross_sae training), compare how the same marker is represented
   differently in each model.

Usage::

    # Feature–marker attribution for a single model
    python -m cross_sae.interpret \\
        --config configs/train_vit_config.yaml \\
        --model-checkpoint checkpoints/vit_final.pth \\
        --sae-checkpoint sae_models/immuvis_vit/sae_checkpoint.pth \\
        --output-dir interpretation_results/

    # Cross-model marker comparison
    python -m cross_sae.interpret \\
        --config configs/train_vit_config.yaml \\
        --model-checkpoint checkpoints/vit_final.pth \\
        --sae-checkpoint cross_sae_results/cross_sae_checkpoint.pth \\
        --cross-model \\
        --output-dir interpretation_cross/
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

from cross_sae.sparse_autoencoder import build_sae


# ===================================================================
# 1. Feature–marker attribution (leave-one-out)
# ===================================================================

@torch.no_grad()
def feature_marker_attribution(
    model: torch.nn.Module,
    sae: torch.nn.Module,
    dataloader,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    inv_tokenizer: dict,
    device: str = "cuda",
    max_batches: int | None = None,
) -> dict:
    """Compute how each SAE feature responds to each marker's removal.

    For each image and each marker channel, removes that channel from
    the input, encodes the masked input, runs through the SAE, and
    records the change in feature activations compared to the full input.

    Returns:
        dict with keys:
            attribution: (num_features, num_markers) array — mean absolute
                change in feature activation when each marker is removed.
            marker_names: list of marker name strings.
            feature_top_markers: (num_features,) — index of most-attributed marker.
    """
    model.eval()
    sae.eval()
    latent_mean = latent_mean.to(device)
    latent_std = latent_std.to(device)

    # Collect all unique markers across the dataset
    all_markers = set()
    sample_data = []

    for batch_idx, (img, channel_ids, _panel, _path) in enumerate(
        tqdm(dataloader, desc="Collecting samples")
    ):
        if max_batches is not None and batch_idx >= max_batches:
            break
        img = img.to(device, dtype=torch.float32)
        channel_ids = channel_ids.to(device, dtype=torch.long)
        sample_data.append((img, channel_ids))

        for c in range(channel_ids.shape[1]):
            token = channel_ids[0, c].item()
            name = inv_tokenizer.get(token, f"ch{token}")
            all_markers.add(name)

    marker_names = sorted(all_markers)
    marker_to_idx = {m: i for i, m in enumerate(marker_names)}
    hidden_dim = sae.hidden_dim if hasattr(sae, "hidden_dim") else sae.encoder.out_features

    # (num_features, num_markers) accumulator
    attr_sum = np.zeros((hidden_dim, len(marker_names)))
    attr_count = np.zeros(len(marker_names))

    for img, channel_ids in tqdm(sample_data, desc="Feature–marker attribution"):
        # Full input encoding
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            z_full = model.encode(img, channel_ids)["output"]
        B, D, Hp, Wp = z_full.shape
        z_full_flat = z_full.permute(0, 2, 3, 1).reshape(-1, D).float()
        z_full_norm = (z_full_flat - latent_mean) / latent_std
        h_full = sae.encode(z_full_norm)  # (N, hidden_dim)

        num_ch = channel_ids.shape[1]
        for c in range(num_ch):
            token = channel_ids[0, c].item()
            marker_name = inv_tokenizer.get(token, f"ch{token}")
            if marker_name not in marker_to_idx:
                continue
            m_idx = marker_to_idx[marker_name]

            # Remove channel c
            keep = torch.ones(num_ch, dtype=torch.bool)
            keep[c] = False
            masked_img = img[:, keep]
            active_ids = channel_ids[:, keep]

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                z_masked = model.encode(masked_img, active_ids)["output"]
            z_masked_flat = z_masked.permute(0, 2, 3, 1).reshape(-1, D).float()
            z_masked_norm = (z_masked_flat - latent_mean) / latent_std
            h_masked = sae.encode(z_masked_norm)

            # Absolute change in feature activation
            delta = (h_full - h_masked).abs().mean(dim=0).cpu().numpy()
            attr_sum[:, m_idx] += delta
            attr_count[m_idx] += 1

    # Average
    attribution = attr_sum / (attr_count[None, :] + 1e-8)
    feature_top_markers = attribution.argmax(axis=1)

    return {
        "attribution": attribution,
        "marker_names": marker_names,
        "feature_top_markers": feature_top_markers,
    }


# ===================================================================
# 2. Marker variance explained
# ===================================================================

@torch.no_grad()
def marker_variance_explained(
    model: torch.nn.Module,
    sae: torch.nn.Module,
    dataloader,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    inv_tokenizer: dict,
    device: str = "cuda",
    max_batches: int | None = None,
) -> dict:
    """Decompose per-marker latent variance across SAE features.

    For each marker, collects the SAE feature activations on images
    *containing* that marker, then computes how much variance each SAE
    feature explains for images where that marker is present.

    Returns:
        dict with:
            variance_explained: (num_features, num_markers) — variance of
                each feature's activation conditioned on marker presence.
            marker_names: list of marker names.
    """
    model.eval()
    sae.eval()
    latent_mean = latent_mean.to(device)
    latent_std = latent_std.to(device)

    hidden_dim = sae.hidden_dim if hasattr(sae, "hidden_dim") else sae.encoder.out_features

    # Collect: per marker, list of feature activation vectors
    marker_features = {}

    for batch_idx, (img, channel_ids, _panel, _path) in enumerate(
        tqdm(dataloader, desc="Marker variance")
    ):
        if max_batches is not None and batch_idx >= max_batches:
            break
        img = img.to(device, dtype=torch.float32)
        channel_ids = channel_ids.to(device, dtype=torch.long)

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            z = model.encode(img, channel_ids)["output"]
        B, D, Hp, Wp = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D).float()
        z_norm = (z_flat - latent_mean) / latent_std
        h = sae.encode(z_norm).cpu()  # (N, hidden_dim)

        # Mean pooled features per image
        h_mean = h.reshape(B, Hp * Wp, hidden_dim).mean(dim=1)  # (B, hidden_dim)

        for c in range(channel_ids.shape[1]):
            token = channel_ids[0, c].item()
            name = inv_tokenizer.get(token, f"ch{token}")
            if name not in marker_features:
                marker_features[name] = []
            marker_features[name].append(h_mean)

    # Compute variance per marker per feature
    marker_names = sorted(marker_features.keys())
    variance = np.zeros((hidden_dim, len(marker_names)))

    for m_idx, name in enumerate(marker_names):
        if not marker_features[name]:
            continue
        stacked = torch.cat(marker_features[name], dim=0)  # (N_images, hidden_dim)
        var = stacked.var(dim=0).numpy()  # (hidden_dim,)
        variance[:, m_idx] = var

    return {
        "variance_explained": variance,
        "marker_names": marker_names,
    }


# ===================================================================
# 3. Predictive features — correlation with per-marker Pearson
# ===================================================================

@torch.no_grad()
def predictive_features(
    model: torch.nn.Module,
    sae: torch.nn.Module,
    dataloader,
    activation_fn,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    inv_tokenizer: dict,
    uncertainty_method: str = "evidential",
    device: str = "cuda",
    max_batches: int | None = None,
) -> dict:
    """Find SAE features correlated with reconstruction quality per marker.

    For each image + marker (leave-one-out), computes:
    - The Pearson correlation of the virtual staining prediction.
    - The spatially-averaged SAE feature activations.

    Then computes the correlation between each SAE feature and the
    per-sample Pearson score for each marker.

    Returns:
        dict with:
            feature_pearson_corr: (num_features, num_markers) — correlation
                of each feature's activation with the Pearson quality score.
            marker_names: list of marker names.
    """
    from scipy.stats import pearsonr

    model.eval()
    sae.eval()
    latent_mean = latent_mean.to(device)
    latent_std = latent_std.to(device)

    hidden_dim = sae.hidden_dim if hasattr(sae, "hidden_dim") else sae.encoder.out_features

    # Collect per marker: list of (feature_vec, pearson_score)
    marker_data: dict[str, list[tuple[np.ndarray, float]]] = {}

    for batch_idx, (img, channel_ids, _panel, _path) in enumerate(
        tqdm(dataloader, desc="Predictive features")
    ):
        if max_batches is not None and batch_idx >= max_batches:
            break
        img = img.to(device, dtype=torch.float32)
        channel_ids = channel_ids.to(device, dtype=torch.long)

        num_ch = channel_ids.shape[1]
        for c in range(num_ch):
            token = channel_ids[0, c].item()
            name = inv_tokenizer.get(token, f"ch{token}")

            keep = torch.ones(num_ch, dtype=torch.bool)
            keep[c] = False
            masked_img = img[:, keep]
            active_ids = channel_ids[:, keep]

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                z = model.encode(masked_img, active_ids)["output"]
                out = model(masked_img, active_ids, channel_ids)["output"]

            # SAE features (mean-pooled)
            B, D, Hp, Wp = z.shape
            z_flat = z.permute(0, 2, 3, 1).reshape(-1, D).float()
            z_norm = (z_flat - latent_mean) / latent_std
            h = sae.encode(z_norm).cpu()
            h_mean = h.reshape(B, Hp * Wp, hidden_dim).mean(dim=1)[0].numpy()

            # Pearson for this marker
            if uncertainty_method == "evidential":
                mi = activation_fn(out[..., 0])
            else:
                mi = activation_fn(out[..., 0])
            pred = mi[0, c].float().cpu().numpy().flatten()
            gt = img[0, c].cpu().numpy().flatten()

            if pred.std() < 1e-8 or gt.std() < 1e-8:
                continue
            r = pearsonr(pred, gt).statistic

            if name not in marker_data:
                marker_data[name] = []
            marker_data[name].append((h_mean, r))

    # Compute feature–Pearson correlations
    marker_names = sorted(marker_data.keys())
    corr_matrix = np.zeros((hidden_dim, len(marker_names)))

    for m_idx, name in enumerate(marker_names):
        data = marker_data[name]
        if len(data) < 5:
            continue
        features_arr = np.stack([d[0] for d in data])  # (N, hidden_dim)
        pearsons = np.array([d[1] for d in data])  # (N,)

        for f in range(hidden_dim):
            feat_col = features_arr[:, f]
            if feat_col.std() < 1e-8:
                continue
            corr_matrix[f, m_idx] = pearsonr(feat_col, pearsons).statistic

    return {
        "feature_pearson_corr": corr_matrix,
        "marker_names": marker_names,
    }


# ===================================================================
# Plotting
# ===================================================================

def plot_feature_marker_attribution(attribution: np.ndarray, marker_names: list[str],
                                    output_dir: str, top_k: int = 30):
    """Heatmap of feature–marker attribution (top-k most selective features)."""
    os.makedirs(output_dir, exist_ok=True)

    # Select top features by max attribution
    max_attr = attribution.max(axis=1)
    top_idx = np.argsort(-max_attr)[:top_k]

    fig, ax = plt.subplots(figsize=(max(12, len(marker_names) * 0.5), top_k * 0.3 + 2))
    im = ax.imshow(attribution[top_idx], cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(marker_names)))
    ax.set_xticklabels(marker_names, rotation=90, fontsize=7)
    ax.set_ylabel("SAE Feature (sorted by selectivity)")
    ax.set_title(f"Feature–Marker Attribution (top {top_k} features)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Mean |Δ activation|")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_marker_attribution.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_marker_variance(variance: np.ndarray, marker_names: list[str], output_dir: str):
    """Bar plot of total variance explained per marker."""
    os.makedirs(output_dir, exist_ok=True)

    total_var = variance.sum(axis=0)  # per-marker total
    order = np.argsort(-total_var)

    fig, ax = plt.subplots(figsize=(max(10, len(marker_names) * 0.4), 5))
    ax.bar(range(len(marker_names)), total_var[order], color="tab:blue", alpha=0.7)
    ax.set_xticks(range(len(marker_names)))
    ax.set_xticklabels([marker_names[i] for i in order], rotation=90, fontsize=7)
    ax.set_ylabel("Total variance across SAE features")
    ax.set_title("Marker Variance in Latent Space")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "marker_variance.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_predictive_features(corr: np.ndarray, marker_names: list[str],
                             output_dir: str, top_k: int = 10):
    """Per-marker bar plots of top predictive SAE features."""
    os.makedirs(output_dir, exist_ok=True)

    n_markers = len(marker_names)
    cols = min(4, n_markers)
    rows = (n_markers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_2d(axes)

    for m_idx, name in enumerate(marker_names):
        ax = axes[m_idx // cols, m_idx % cols]
        feat_corr = corr[:, m_idx]
        top_features = np.argsort(-np.abs(feat_corr))[:top_k]
        colors = ["tab:green" if feat_corr[f] > 0 else "tab:red" for f in top_features]
        ax.barh(range(top_k), feat_corr[top_features], color=colors, alpha=0.7)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([f"F{f}" for f in top_features], fontsize=6)
        ax.set_xlabel("Corr with Pearson")
        ax.set_title(name, fontsize=9)
        ax.invert_yaxis()

    # Hide empty subplots
    for idx in range(n_markers, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    plt.suptitle("Top Predictive SAE Features per Marker", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "predictive_features.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="SAE marker interpretation")
    parser.add_argument("--config", required=True, help="ImmuVis config YAML")
    parser.add_argument("--model-checkpoint", required=True, help="ImmuVis model checkpoint")
    parser.add_argument("--sae-checkpoint", required=True, help="SAE or cross-SAE checkpoint")
    parser.add_argument("--cross-model", action="store_true",
                        help="Use cross-SAE checkpoint (analyses SAE_A)")
    parser.add_argument("--output-dir", default="interpretation_results/")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip-markers", nargs="*", default=["DNA1", "DNA2"],
        help="Markers to skip in predictive analysis",
    )
    args = parser.parse_args()

    from ruamel.yaml import YAML
    from torch.utils.data import DataLoader

    from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler, TestCrop
    from multiplex_model.losses import get_output_activation
    from multiplex_model.modules import MultiplexAutoencoder
    from multiplex_model.utils.configuration import TrainingConfig

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Load ImmuVis model
    yaml_loader = YAML(typ="safe")
    with open(args.config) as f:
        raw_config = yaml_loader.load(f)
    config = TrainingConfig(**raw_config)

    PANEL_CONFIG = YAML().load(open(config.panel_config))
    TOKENIZER = YAML().load(open(config.tokenizer_config))
    INV_TOKENIZER = {v: k for k, v in TOKENIZER.items()}
    SIZE = config.input_image_size

    dataset = DatasetFromTIFF(
        panels_config=PANEL_CONFIG, split="test", marker_tokenizer=TOKENIZER,
        transform=TestCrop(SIZE[0]), use_preprocessing=False,
        use_butterworth_filter=True, use_clip_normalization=True,
    )
    sampler = PanelBatchSampler(dataset, batch_size=1, shuffle=False)
    dataloader = DataLoader(dataset, batch_sampler=sampler,
                            num_workers=config.num_workers, pin_memory=True)

    decoder_cfg = config.decoder_config.model_dump()
    decoder_cfg["num_outputs"] = 4 if config.uncertainty_method == "evidential" else 2

    model = MultiplexAutoencoder(
        num_channels=len(TOKENIZER),
        encoder_config=config.encoder_config.model_dump(),
        decoder_config=decoder_cfg,
    ).to(device)

    ckpt = torch.load(args.model_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()

    activation_fn = get_output_activation(
        config.output_activation, beta=config.activation_beta,
        window=getattr(config, "activation_window", 0.5),
    )

    # Load SAE
    sae_ckpt = torch.load(args.sae_checkpoint, map_location=device, weights_only=True)
    if args.cross_model:
        sae_state = sae_ckpt["sae_a_state_dict"]
        latent_mean = sae_ckpt["history_a"]["latent_mean"]
        latent_std = sae_ckpt["history_a"]["latent_std"]
    else:
        sae_state = sae_ckpt["sae_state_dict"]
        latent_mean = sae_ckpt["latent_mean"]
        latent_std = sae_ckpt["latent_std"]

    sae = build_sae(
        sae_ckpt["sae_variant"], sae_ckpt["input_dim"],
        sae_ckpt["hidden_dim"], **sae_ckpt["sae_kwargs"],
    )
    sae.load_state_dict(sae_state)
    sae = sae.to(device)
    sae.eval()
    latent_mean = latent_mean.to(device)
    latent_std = latent_std.to(device)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Feature–marker attribution
    print("=" * 60)
    print("Running feature–marker attribution...")
    print("=" * 60)
    attr_result = feature_marker_attribution(
        model, sae, dataloader, latent_mean, latent_std,
        INV_TOKENIZER, device=device, max_batches=args.max_batches,
    )
    plot_feature_marker_attribution(
        attr_result["attribution"], attr_result["marker_names"], args.output_dir,
    )
    np.savez(
        os.path.join(args.output_dir, "feature_marker_attribution.npz"),
        **attr_result,
    )

    # 2. Marker variance
    print("\n" + "=" * 60)
    print("Computing marker variance decomposition...")
    print("=" * 60)
    var_result = marker_variance_explained(
        model, sae, dataloader, latent_mean, latent_std,
        INV_TOKENIZER, device=device, max_batches=args.max_batches,
    )
    plot_marker_variance(
        var_result["variance_explained"], var_result["marker_names"], args.output_dir,
    )
    np.savez(
        os.path.join(args.output_dir, "marker_variance.npz"),
        **var_result,
    )

    # 3. Predictive features
    print("\n" + "=" * 60)
    print("Finding predictive features per marker...")
    print("=" * 60)
    pred_result = predictive_features(
        model, sae, dataloader, activation_fn, latent_mean, latent_std,
        INV_TOKENIZER, uncertainty_method=config.uncertainty_method,
        device=device, max_batches=args.max_batches,
    )
    plot_predictive_features(
        pred_result["feature_pearson_corr"], pred_result["marker_names"], args.output_dir,
    )
    np.savez(
        os.path.join(args.output_dir, "predictive_features.npz"),
        **pred_result,
    )

    print(f"\nAll interpretation results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
