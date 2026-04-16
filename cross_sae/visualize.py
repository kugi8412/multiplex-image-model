"""Spatial visualisation utilities for SAE feature analysis.

Generates per-image spatial activation maps and feature dashboards that
overlay SAE feature activations on the original multiplex images.

Functions
---------
spatial_activation_map
    For a single image, show which spatial locations activate each
    SAE feature.  Useful for understanding what tissue structures
    each dictionary element captures.

feature_dashboard
    Multi-panel figure combining: the original marker channels,
    the SAE feature activation heatmap, the top-k firing features,
    and the reconstruction error map.

cross_model_spatial_comparison
    Side-by-side spatial activation maps from two models' SAEs on
    the same image, highlighting features that are shared vs. unique.

Usage::

    python -m cross_sae.visualize \\
        --config configs/train_vit_config.yaml \\
        --model-checkpoint checkpoints/vit_final.pth \\
        --sae-checkpoint sae_models/immuvis_vit/sae_checkpoint.pth \\
        --image-idx 0 5 10 \\
        --output-dir sae_visualizations/
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from torch.amp import autocast
from tqdm import tqdm

from cross_sae.sparse_autoencoder import build_sae


# ===================================================================
# Spatial activation maps
# ===================================================================

@torch.no_grad()
def compute_spatial_features(
    model: torch.nn.Module,
    sae: torch.nn.Module,
    img: torch.Tensor,
    channel_ids: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-spatial-position SAE feature activations.

    Args:
        model: Frozen ImmuVis autoencoder.
        sae: Trained SAE.
        img: (1, C, H, W) input image.
        channel_ids: (1, C) channel token IDs.
        latent_mean, latent_std: SAE normalisation stats.

    Returns:
        feature_map: (hidden_dim, H', W') spatial feature activations.
        latent_map: (D, H', W') raw latent activations.
    """
    img = img.to(device, dtype=torch.float32)
    channel_ids = channel_ids.to(device, dtype=torch.long)
    latent_mean = latent_mean.to(device)
    latent_std = latent_std.to(device)

    with autocast(device_type="cuda", dtype=torch.bfloat16):
        z = model.encode(img, channel_ids)["output"]  # (1, D, H', W')

    _, D, Hp, Wp = z.shape
    z_flat = z[0].permute(1, 2, 0).reshape(-1, D).float()  # (H'*W', D)
    z_norm = (z_flat - latent_mean) / latent_std
    h = sae.encode(z_norm)  # (H'*W', hidden_dim)

    feature_map = h.reshape(Hp, Wp, -1).permute(2, 0, 1).cpu().numpy()
    latent_map = z[0].float().cpu().numpy()

    return feature_map, latent_map


def plot_spatial_activation_map(
    feature_map: np.ndarray,
    img: np.ndarray,
    marker_names: list[str],
    feature_indices: list[int] | None = None,
    top_k: int = 8,
    output_path: str | None = None,
):
    """Plot spatial activation maps for selected SAE features.

    Args:
        feature_map: (hidden_dim, H', W') feature activations.
        img: (C, H, W) original image.
        marker_names: Names of input marker channels.
        feature_indices: Specific features to visualise. If None, picks
            top-k by total activation.
        top_k: Number of features to show (if feature_indices is None).
        output_path: If given, save to file instead of showing.
    """
    if feature_indices is None:
        total_act = feature_map.sum(axis=(1, 2))
        feature_indices = np.argsort(-total_act)[:top_k].tolist()

    n_features = len(feature_indices)
    # Show 3 representative markers + n_features activation maps
    n_markers_show = min(3, img.shape[0])
    n_cols = max(n_features, n_markers_show)

    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6))
    if n_cols == 1:
        axes = axes[:, None]

    # Row 1: marker channels
    for i in range(n_cols):
        ax = axes[0, i]
        if i < n_markers_show:
            ax.imshow(img[i], cmap="inferno", vmin=0, vmax=1)
            ax.set_title(marker_names[i] if i < len(marker_names) else f"Ch{i}", fontsize=8)
        ax.axis("off")

    # Row 2: SAE feature activation maps
    for i in range(n_cols):
        ax = axes[1, i]
        if i < n_features:
            f_idx = feature_indices[i]
            im = ax.imshow(feature_map[f_idx], cmap="hot", interpolation="nearest")
            ax.set_title(f"Feature {f_idx}", fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046)
        ax.axis("off")

    plt.suptitle("Spatial SAE Feature Activations", fontsize=11)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ===================================================================
# Feature dashboard — comprehensive per-image view
# ===================================================================

def plot_feature_dashboard(
    feature_map: np.ndarray,
    img: np.ndarray,
    marker_names: list[str],
    recon: np.ndarray | None = None,
    top_k: int = 6,
    output_path: str | None = None,
):
    """Multi-panel dashboard for a single image.

    Panels:
    - Row 1: Input marker channels (up to 6).
    - Row 2: Top-k SAE feature activation maps.
    - Row 3 (optional): Reconstruction and error map.
    """
    n_markers = min(6, img.shape[0])
    n_features = top_k
    n_cols = max(n_markers, n_features)

    total_act = feature_map.sum(axis=(1, 2))
    top_feat_idx = np.argsort(-total_act)[:n_features]

    n_rows = 3 if recon is not None else 2
    fig = plt.figure(figsize=(3 * n_cols, 3 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig)

    # Row 1: Marker channels
    for i in range(n_cols):
        ax = fig.add_subplot(gs[0, i])
        if i < n_markers:
            ax.imshow(img[i], cmap="inferno", vmin=0, vmax=1)
            name = marker_names[i] if i < len(marker_names) else f"Ch{i}"
            ax.set_title(name, fontsize=7)
        ax.axis("off")

    # Row 2: SAE features
    for i in range(n_cols):
        ax = fig.add_subplot(gs[1, i])
        if i < n_features:
            f_idx = top_feat_idx[i]
            ax.imshow(feature_map[f_idx], cmap="hot", interpolation="nearest")
            ax.set_title(f"F{f_idx} (Σ={total_act[f_idx]:.1f})", fontsize=7)
        ax.axis("off")

    # Row 3: Reconstruction + error
    if recon is not None:
        for i in range(n_cols):
            ax = fig.add_subplot(gs[2, i])
            if i < n_markers:
                if i == 0:
                    ax.imshow(recon[i], cmap="inferno", vmin=0, vmax=1)
                    ax.set_title("Reconstruction", fontsize=7)
                elif i == 1:
                    error = np.abs(img[0] - recon[0])
                    ax.imshow(error, cmap="Reds", vmin=0, vmax=0.3)
                    ax.set_title("Error map", fontsize=7)
                else:
                    ax.imshow(recon[i], cmap="inferno", vmin=0, vmax=1)
                    ax.set_title(f"Recon {marker_names[i]}", fontsize=7)
            ax.axis("off")

    plt.suptitle("SAE Feature Dashboard", fontsize=11)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ===================================================================
# Cross-model spatial comparison
# ===================================================================

def plot_cross_model_comparison(
    feature_map_a: np.ndarray,
    feature_map_b: np.ndarray,
    img: np.ndarray,
    shared_features_a: list[int],
    unique_features_a: list[int],
    label_a: str = "Model A",
    label_b: str = "Model B",
    top_k: int = 4,
    output_path: str | None = None,
):
    """Side-by-side spatial maps highlighting shared and unique features."""
    n_shared = min(top_k, len(shared_features_a))
    n_unique = min(top_k, len(unique_features_a))
    n_cols = max(n_shared + n_unique, 1)

    fig, axes = plt.subplots(3, n_cols, figsize=(3 * n_cols, 9))
    if n_cols == 1:
        axes = axes[:, None]

    # Row 1: Original markers
    for i in range(n_cols):
        ax = axes[0, i]
        if i < img.shape[0]:
            ax.imshow(img[i], cmap="inferno", vmin=0, vmax=1)
        ax.axis("off")
    axes[0, 0].set_ylabel("Input", fontsize=9)

    # Row 2: Shared features (Model A activations)
    for i in range(n_cols):
        ax = axes[1, i]
        if i < n_shared:
            f_idx = shared_features_a[i]
            ax.imshow(feature_map_a[f_idx], cmap="Greens", interpolation="nearest")
            ax.set_title(f"Shared F{f_idx}", fontsize=7, color="green")
        ax.axis("off")
    axes[1, 0].set_ylabel(f"{label_a} (shared)", fontsize=9)

    # Row 3: Unique features (Model A only)
    for i in range(n_cols):
        ax = axes[2, i]
        if i < n_unique:
            f_idx = unique_features_a[i]
            ax.imshow(feature_map_a[f_idx], cmap="Reds", interpolation="nearest")
            ax.set_title(f"Unique F{f_idx}", fontsize=7, color="red")
        ax.axis("off")
    axes[2, 0].set_ylabel(f"{label_a} (unique)", fontsize=9)

    plt.suptitle(f"Cross-SAE: {label_a} vs {label_b}", fontsize=11)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="SAE spatial visualisation")
    parser.add_argument("--config", required=True, help="ImmuVis config YAML")
    parser.add_argument("--model-checkpoint", required=True, help="ImmuVis checkpoint")
    parser.add_argument("--sae-checkpoint", required=True, help="SAE checkpoint")
    parser.add_argument(
        "--image-idx", type=int, nargs="+", default=[0, 1, 2],
        help="Test image indices to visualise",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Top features to show")
    parser.add_argument("--output-dir", default="sae_visualizations/")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from ruamel.yaml import YAML
    from torch.utils.data import DataLoader

    from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler, TestCrop
    from multiplex_model.modules import MultiplexAutoencoder
    from multiplex_model.utils.configuration import TrainingConfig

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

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
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)

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

    # Load SAE
    sae_ckpt = torch.load(args.sae_checkpoint, map_location=device, weights_only=True)
    sae = build_sae(
        sae_ckpt["sae_variant"], sae_ckpt["input_dim"],
        sae_ckpt["hidden_dim"], **sae_ckpt["sae_kwargs"],
    )
    sae.load_state_dict(sae_ckpt["sae_state_dict"])
    sae = sae.to(device).eval()
    latent_mean = sae_ckpt["latent_mean"].to(device)
    latent_std = sae_ckpt["latent_std"].to(device)

    os.makedirs(args.output_dir, exist_ok=True)

    target_indices = set(args.image_idx)
    for batch_idx, (img, channel_ids, _panel, img_path) in enumerate(
        tqdm(dataloader, desc="Generating visualisations")
    ):
        if batch_idx not in target_indices:
            continue

        marker_names = [
            INV_TOKENIZER.get(channel_ids[0, c].item(), f"ch{c}")
            for c in range(channel_ids.shape[1])
        ]

        feature_map, latent_map = compute_spatial_features(
            model, sae, img, channel_ids, latent_mean, latent_std, device,
        )

        img_np = img[0].cpu().numpy()

        # Spatial activation map
        plot_spatial_activation_map(
            feature_map, img_np, marker_names, top_k=args.top_k,
            output_path=os.path.join(args.output_dir, f"spatial_map_{batch_idx}.png"),
        )

        # Full dashboard
        plot_feature_dashboard(
            feature_map, img_np, marker_names, top_k=min(args.top_k, 6),
            output_path=os.path.join(args.output_dir, f"dashboard_{batch_idx}.png"),
        )

        target_indices.discard(batch_idx)
        if not target_indices:
            break

    print(f"Saved visualisations to {args.output_dir}/")


if __name__ == "__main__":
    main()
