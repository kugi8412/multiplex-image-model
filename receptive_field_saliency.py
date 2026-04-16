#!/usr/bin/env python
# -*- coding: utf-8 -*-
# receptive_field_saliency.py


"""
Effective receptive field analysis via input-gradient saliency.

For each test image (leave-one-out virtual staining setup on HN dataset):
  1. Set input to require gradients.
  2. Forward pass through the autoencoder.
  3. Extract the 4 central output pixels.
  4. Compute scalar = ||central||² and backpropagate.
  5. The resulting input gradient is the saliency map.
  6. Group pixels by L∞ distance from the image centre and compute
     the mean saliency norm at each distance.
  7. Plot distance vs. mean saliency norm.

This reveals the effective receptive field: how far from the centre
the model looks when predicting the central pixels.

Usage::

    python receptive_field_saliency.py configs/train_vit_config.yaml \\
        --checkpoint checkpoints/final_model.pth \\
        --output-dir erf_results/

    python receptive_field_saliency.py configs/train_mambaswin_config.yaml \\
        --checkpoint checkpoints/mambaswin.pth \\
        --num-images 50 --output-dir erf_mambaswin/
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from tqdm import tqdm

from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler, TestCrop
from multiplex_model.losses import get_output_activation, LearnableOutputActivation
from multiplex_model.modules import MultiplexAutoencoder
from multiplex_model.utils.configuration import TrainingConfig


def build_linf_distance_map(H: int, W: int) -> np.ndarray:
    """Return (H, W) array where each entry is the L∞ distance to centre."""
    cy, cx = H // 2, W // 2
    ys = np.abs(np.arange(H) - cy)
    xs = np.abs(np.arange(W) - cx)
    return np.maximum(ys[:, None], xs[None, :])


def compute_saliency(
    model,
    img: torch.Tensor,
    channel_ids: torch.Tensor,
    full_channel_ids: torch.Tensor,
    activation_fn,
    uncertainty_method: str,
    device: str,
) -> np.ndarray:
    """Compute input-gradient saliency for the 4 central output pixels.

    Args:
        model: Autoencoder in eval mode.
        img: Input image (1, C, H, W) on device, requires_grad will be set.
        channel_ids: Active channel IDs for the (masked) input.
        full_channel_ids: Full channel IDs for the output.
        activation_fn: Output activation function.
        uncertainty_method: 'evidential' or 'beta_nll'.
        device: Device string.

    Returns:
        saliency: (C, H, W) gradient magnitude map.
    """
    img = img.detach().clone().requires_grad_(True)

    output = model(img, channel_ids, full_channel_ids)["output"]

    if uncertainty_method == "evidential":
        gamma_raw = output[..., 0]
        mi = activation_fn(gamma_raw)
    else:
        mi_raw = output[..., 0]
        mi = activation_fn(mi_raw)

    # Extract 4 central pixels (2×2 block)
    _, _, H, W = mi.shape
    cy, cx = H // 2, W // 2
    central = mi[:, :, cy - 1 : cy + 1, cx - 1 : cx + 1]
    scalar = torch.norm(central) ** 2
    scalar.backward()

    # Saliency = absolute gradient on input
    saliency = img.grad.detach().abs()  # (1, C, H, W)
    return saliency[0].cpu().numpy()  # (C, H, W)


def aggregate_saliency_by_distance(
    saliency: np.ndarray, dist_map: np.ndarray
) -> dict[int, float]:
    """Average saliency norm over pixels at each L∞ distance from centre.

    Args:
        saliency: (C, H, W) gradient magnitudes.
        dist_map: (H, W) integer L∞ distances.

    Returns:
        Dict mapping distance → mean L2 norm of the gradient vector at that distance.
    """
    # Per-pixel L2 norm across channels: (H, W)
    norm_map = np.linalg.norm(saliency, axis=0)

    max_dist = int(dist_map.max())
    result = {}
    for d in range(max_dist + 1):
        mask = dist_map == d
        if mask.any():
            result[d] = float(norm_map[mask].mean())
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Effective receptive field saliency analysis"
    )
    parser.add_argument(
        "config", help="Training config YAML (e.g. configs/train_vit_config.yaml)"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--output-dir", default="erf_results", help="Directory for plots and data"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Max number of test images to process (default: all)",
    )
    parser.add_argument(
        "--skip-markers",
        nargs="*",
        default=["DNA1", "DNA2"],
        help="Structural markers to skip",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # Load config
    yaml_loader = YAML(typ="safe")
    with open(args.config) as f:
        raw_config = yaml_loader.load(f)
    config = TrainingConfig(**raw_config)

    device = args.device or config.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Using device: {device}")

    PANEL_CONFIG = YAML().load(open(config.panel_config))
    TOKENIZER = YAML().load(open(config.tokenizer_config))
    INV_TOKENIZER = {v: k for k, v in TOKENIZER.items()}
    num_channels = len(TOKENIZER)

    # ---- Dataset ----
    SIZE = config.input_image_size
    test_transform = TestCrop(SIZE[0])

    test_dataset = DatasetFromTIFF(
        panels_config=PANEL_CONFIG,
        split="test",
        marker_tokenizer=TOKENIZER,
        transform=test_transform,
        use_preprocessing=False,
        use_butterworth_filter=True,
        use_clip_normalization=True,
    )

    test_batch_sampler = PanelBatchSampler(test_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(
        test_dataset,
        batch_sampler=test_batch_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Model
    decoder_cfg = config.decoder_config.model_dump()
    decoder_cfg["num_outputs"] = 4 if config.uncertainty_method == "evidential" else 2

    model = MultiplexAutoencoder(
        num_channels=num_channels,
        encoder_config=config.encoder_config.model_dump(),
        decoder_config=decoder_cfg,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(
        ckpt.get("model_state_dict", ckpt), strict=False
    )
    model.eval()
    print(f"Loaded model from {args.checkpoint}")

    # Activation function
    if config.learnable_activation_params and "activation_state_dict" in ckpt:
        activation_fn = LearnableOutputActivation(
            name=config.output_activation,
            beta=config.activation_beta,
            window=config.activation_window,
            learnable=True,
        ).to(device)
        activation_fn.load_state_dict(ckpt["activation_state_dict"])
        activation_fn.eval()
        print(f"Loaded learnable activation: {activation_fn}")
    else:
        activation_fn = get_output_activation(
            config.output_activation,
            beta=config.activation_beta,
            window=config.activation_window,
        )

    H, W = SIZE
    dist_map = build_linf_distance_map(H, W)
    max_dist = int(dist_map.max())

    # Accumulators: sum of saliency norms and counts per distance
    sum_per_dist = np.zeros(max_dist + 1)
    count_per_dist = np.zeros(max_dist + 1)

    os.makedirs(args.output_dir, exist_ok=True)
    skip_set = set(args.skip_markers)

    # Saliency loop
    n_processed = 0
    for _, (img, channel_ids, dataset_name, img_path) in enumerate(
        tqdm(test_dataloader, desc="Computing saliency")
    ):
        if args.num_images is not None and n_processed >= args.num_images:
            break

        img = img.to(device, dtype=torch.float32)
        channel_ids = channel_ids.to(device, dtype=torch.long)
        _, num_ch, _, _ = img.shape

        # Leave-one-out
        for c in range(num_ch):
            marker_token = channel_ids[0, c].item()
            marker_name = INV_TOKENIZER.get(marker_token, f"ch{marker_token}")
            if marker_name in skip_set:
                continue

            keep_mask = torch.ones(num_ch, dtype=torch.bool)
            keep_mask[c] = False
            masked_img = img[:, keep_mask]
            active_ids = channel_ids[:, keep_mask]

            saliency = compute_saliency(
                model,
                masked_img,
                active_ids,
                channel_ids,
                activation_fn,
                config.uncertainty_method,
                device,
            )

            dist_means = aggregate_saliency_by_distance(saliency, dist_map)
            for d, val in dist_means.items():
                sum_per_dist[d] += val
                count_per_dist[d] += 1

        n_processed += 1
        torch.cuda.empty_cache()

    # Aggregate
    valid = count_per_dist > 0
    distances = np.arange(max_dist + 1)[valid]
    mean_saliency = sum_per_dist[valid] / count_per_dist[valid]

    # Normalise to [0, 1] for easier comparison across models
    if mean_saliency.max() > 0:
        mean_saliency_norm = mean_saliency / mean_saliency.max()
    else:
        mean_saliency_norm = mean_saliency

    # Save raw data
    data_path = os.path.join(args.output_dir, "erf_data.npz")
    np.savez(
        data_path,
        distances=distances,
        mean_saliency=mean_saliency,
        mean_saliency_normalised=mean_saliency_norm,
        count_per_dist=count_per_dist[valid],
    )
    print(f"Saved ERF data to {data_path}")

    # Make plot
    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute saliency
    axes[0].plot(distances, mean_saliency, "b-", linewidth=1.5)
    axes[0].fill_between(distances, 0, mean_saliency, alpha=0.2, color="blue")
    axes[0].set_xlabel("L00 distance from centre (pixels)")
    axes[0].set_ylabel("Mean gradient norm")
    axes[0].set_title("Effective Receptive Field — Absolute Saliency")
    axes[0].grid(True, alpha=0.3)

    # Normalised saliency (log-scale y for tail behaviour)
    axes[1].semilogy(distances, mean_saliency_norm + 1e-10, "r-", linewidth=1.5)
    axes[1].set_xlabel("L∞ distance from centre (pixels)")
    axes[1].set_ylabel("Normalised mean gradient norm (log)")
    axes[1].set_title("Effective Receptive Field — Log Scale")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "erf_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved ERF plot to {plot_path}")

    # Print summary
    print(f"\nProcessed {n_processed} images")
    print(f"Max distance: {distances[-1]} pixels")
    for threshold, label in [(0.5, "50%"), (0.1, "10%")]:
        idx = np.where(mean_saliency_norm <= threshold)[0]
        if len(idx) > 0:
            print(f"Saliency drops to {label} of peak at distance {distances[idx[0]]} px")
        else:
            print(f"Saliency never drops below {label} within the image")


if __name__ == "__main__":
    main()
