"""CKA (Centered Kernel Alignment) comparison between two SAE feature spaces.

Compares the learned SAE dictionary representations from two models by
computing CKA on their feature activations over a shared set of images.

Usage::

    # Compare ViT baseline SAE vs ImmunoKRONOS SAE
    python -m cross_sae.cka_comparison \
        --sae-a sae_models/exp7c_vit/sae_checkpoint.pth \
        --sae-b sae_models/exp6a_kronos/sae_checkpoint.pth \
        --config-a configs/exp7c_vit_baseline.yaml \
        --checkpoint-a checkpoints/exp7c_vit_baseline/final_model-<RUN>.pth \
        --config-b configs/exp6a_immukronos_v2.yaml \
        --checkpoint-b checkpoints/exp6a_immukronos_v2/kronos_immukronos_v2-<RUN>-epoch_199.pth \
        --source-a immuvis --source-b kronos \
        --output-dir results/cka_vit_vs_kronos/
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cross_sae.sparse_autoencoder import build_sae
from cross_sae.train_sae import (
    extract_immuvis_latents,
    extract_kronos_latents,
)


# ===================================================================
# CKA computation
# ===================================================================

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two representation matrices.

    Args:
        X: (N, D_x) feature activations from model A.
        Y: (N, D_y) feature activations from model B.

    Returns:
        CKA similarity score in [0, 1].
    """
    # Center columns
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Gram matrices via dot products
    XtX = X.T @ X  # (D_x, D_x)
    YtY = Y.T @ Y  # (D_y, D_y)
    XtY = X.T @ Y  # (D_x, D_y)

    # HSIC estimators (linear kernel)
    hsic_xy = np.sum(XtY ** 2)
    hsic_xx = np.sum(XtX ** 2)
    hsic_yy = np.sum(YtY ** 2)

    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10))


def rbf_cka(X: np.ndarray, Y: np.ndarray, sigma: float | None = None) -> float:
    """Compute RBF-kernel CKA between two representation matrices.

    Args:
        X: (N, D_x) feature activations.
        Y: (N, D_y) feature activations.
        sigma: RBF bandwidth. If None, uses median heuristic.

    Returns:
        CKA similarity score in [0, 1].
    """
    def _rbf_gram(Z, sigma):
        sq_dists = np.sum((Z[:, None] - Z[None, :]) ** 2, axis=-1)
        if sigma is None:
            sigma = np.sqrt(np.median(sq_dists) / 2 + 1e-10)
        return np.exp(-sq_dists / (2 * sigma ** 2)), sigma

    K_x, sigma_x = _rbf_gram(X, sigma)
    K_y, sigma_y = _rbf_gram(Y, sigma)

    # Center Gram matrices
    N = K_x.shape[0]
    H = np.eye(N) - 1.0 / N
    K_x = H @ K_x @ H
    K_y = H @ K_y @ H

    hsic_xy = np.trace(K_x @ K_y) / (N - 1) ** 2
    hsic_xx = np.trace(K_x @ K_x) / (N - 1) ** 2
    hsic_yy = np.trace(K_y @ K_y) / (N - 1) ** 2

    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10))


# ===================================================================
# SAE feature extraction
# ===================================================================

@torch.no_grad()
def get_sae_features(
    sae: torch.nn.Module,
    latents: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    device: str = "cuda",
    batch_size: int = 4096,
) -> np.ndarray:
    """Get SAE hidden feature activations for a set of latents.

    Returns:
        features: (N, hidden_dim) numpy array of sparse activations.
    """
    sae = sae.to(device).eval()
    all_features = []

    for start in range(0, latents.shape[0], batch_size):
        batch = latents[start:start + batch_size].to(device)
        batch = (batch - latent_mean.to(device)) / latent_std.to(device)
        h = sae.encode(batch)
        all_features.append(h.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def load_sae_from_checkpoint(checkpoint_path: str) -> tuple:
    """Load a trained SAE and its normalization stats from a checkpoint.

    Returns:
        sae, latent_mean, latent_std, input_dim, hidden_dim
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sae_kwargs = ckpt.get("sae_kwargs", {})
    sae = build_sae(
        ckpt["sae_variant"],
        ckpt["input_dim"],
        ckpt["hidden_dim"],
        **sae_kwargs,
    )
    sae.load_state_dict(ckpt["sae_state_dict"])
    latent_mean = ckpt["latent_mean"]
    latent_std = ckpt["latent_std"]
    return sae, latent_mean, latent_std, ckpt["input_dim"], ckpt["hidden_dim"]


# ===================================================================
# Plotting
# ===================================================================

def plot_cka_results(
    results: dict,
    output_dir: str,
    label_a: str = "Model A",
    label_b: str = "Model B",
):
    """Plot CKA comparison results."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. CKA scores bar chart
    ax = axes[0]
    keys = ["linear_cka_raw", "linear_cka_sae", "rbf_cka_raw", "rbf_cka_sae"]
    labels = ["Linear\n(raw latents)", "Linear\n(SAE features)", "RBF\n(raw latents)", "RBF\n(SAE features)"]
    values = [results.get(k, 0) for k in keys]
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]
    bars = ax.bar(range(len(keys)), values, color=colors, width=0.6)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("CKA Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"CKA: {label_a} vs {label_b}")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")

    # 2. Feature overlap histogram
    ax = axes[1]
    if "cosine_sim_max_a" in results:
        ax.hist(results["cosine_sim_max_a"], bins=50, alpha=0.7, label=f"{label_a} → {label_b}",
                color="#4c72b0")
        ax.hist(results["cosine_sim_max_b"], bins=50, alpha=0.7, label=f"{label_b} → {label_a}",
                color="#c44e52")
        ax.set_xlabel("Max cosine similarity to nearest feature")
        ax.set_ylabel("Count")
        ax.set_title("SAE Feature Alignment")
        ax.legend(fontsize=8)

    # 3. Sparsity comparison
    ax = axes[2]
    if "l0_a" in results and "l0_b" in results:
        ax.hist(results["l0_a"], bins=50, alpha=0.7, label=f"{label_a} (mean L0={results['l0_a'].mean():.1f})",
                color="#4c72b0")
        ax.hist(results["l0_b"], bins=50, alpha=0.7, label=f"{label_b} (mean L0={results['l0_b'].mean():.1f})",
                color="#c44e52")
        ax.set_xlabel("L0 (active features per sample)")
        ax.set_ylabel("Count")
        ax.set_title("SAE Sparsity Distribution")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cka_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved CKA plot to {output_dir}/cka_comparison.png")


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CKA comparison between two SAE feature spaces"
    )

    # SAE checkpoints
    parser.add_argument("--sae-a", required=True, help="Path to SAE checkpoint for model A")
    parser.add_argument("--sae-b", required=True, help="Path to SAE checkpoint for model B")

    # Model configs for latent extraction
    parser.add_argument("--source-a", default="immuvis", choices=["immuvis", "kronos"])
    parser.add_argument("--config-a", required=True, help="Training config for model A")
    parser.add_argument("--checkpoint-a", required=True, help="Model checkpoint for model A")
    parser.add_argument("--source-b", default="kronos", choices=["immuvis", "kronos"])
    parser.add_argument("--config-b", required=True, help="Training config for model B")
    parser.add_argument("--checkpoint-b", required=True, help="Model checkpoint for model B")

    parser.add_argument("--label-a", default="ViT-Baseline")
    parser.add_argument("--label-b", default="ImmunoKRONOS")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=10000,
                        help="Max samples for CKA computation (subsampled if needed)")
    parser.add_argument("--output-dir", default="results/cka_comparison/")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    # Load SAEs
    print("Loading SAE A...")
    sae_a, mean_a, std_a, dim_a, hdim_a = load_sae_from_checkpoint(args.sae_a)
    print(f"  SAE A: input_dim={dim_a}, hidden_dim={hdim_a}")

    print("Loading SAE B...")
    sae_b, mean_b, std_b, dim_b, hdim_b = load_sae_from_checkpoint(args.sae_b)
    print(f"  SAE B: input_dim={dim_b}, hidden_dim={hdim_b}")

    # Extract latents
    extractors = {
        "immuvis": extract_immuvis_latents,
        "kronos": extract_kronos_latents,
    }

    print(f"\nExtracting latents from model A ({args.source_a})...")
    latents_a, _ = extractors[args.source_a](
        args.config_a, args.checkpoint_a, device=device,
        max_batches=args.max_batches,
    )

    print(f"Extracting latents from model B ({args.source_b})...")
    latents_b, _ = extractors[args.source_b](
        args.config_b, args.checkpoint_b, device=device,
        max_batches=args.max_batches,
    )

    # Subsample to match sizes for CKA
    n = min(latents_a.shape[0], latents_b.shape[0], args.max_samples)
    print(f"\nUsing {n} samples for CKA computation")
    idx_a = torch.randperm(latents_a.shape[0])[:n]
    idx_b = torch.randperm(latents_b.shape[0])[:n]
    latents_a_sub = latents_a[idx_a]
    latents_b_sub = latents_b[idx_b]

    # Get SAE features
    print("Computing SAE features...")
    feat_a = get_sae_features(sae_a, latents_a_sub, mean_a, std_a, device=device)
    feat_b = get_sae_features(sae_b, latents_b_sub, mean_b, std_b, device=device)

    # Compute CKA scores
    print("\nComputing CKA...")
    results = {}

    # CKA on raw latents (project to same dim if needed)
    lat_a_np = latents_a_sub.numpy()
    lat_b_np = latents_b_sub.numpy()
    results["linear_cka_raw"] = linear_cka(lat_a_np, lat_b_np)
    print(f"  Linear CKA (raw latents):    {results['linear_cka_raw']:.4f}")

    results["rbf_cka_raw"] = rbf_cka(lat_a_np, lat_b_np)
    print(f"  RBF CKA (raw latents):       {results['rbf_cka_raw']:.4f}")

    # CKA on SAE features
    results["linear_cka_sae"] = linear_cka(feat_a, feat_b)
    print(f"  Linear CKA (SAE features):   {results['linear_cka_sae']:.4f}")

    results["rbf_cka_sae"] = rbf_cka(feat_a, feat_b)
    print(f"  RBF CKA (SAE features):      {results['rbf_cka_sae']:.4f}")

    # Feature alignment via decoder cosine similarity
    W_a = sae_a.W_dec if hasattr(sae_a, "W_dec") else sae_a.decoder.weight
    W_b = sae_b.W_dec if hasattr(sae_b, "W_dec") else sae_b.decoder.weight
    w_a = F.normalize(W_a.detach().float(), dim=0).numpy()
    w_b = F.normalize(W_b.detach().float(), dim=0).numpy()

    if w_a.shape[0] == w_b.shape[0]:
        sim = w_a.T @ w_b
        results["cosine_sim_max_a"] = np.abs(sim).max(axis=1)
        results["cosine_sim_max_b"] = np.abs(sim).max(axis=0)
        results["mean_max_cosine_a_to_b"] = float(results["cosine_sim_max_a"].mean())
        results["mean_max_cosine_b_to_a"] = float(results["cosine_sim_max_b"].mean())
        print(f"  Mean max|cos| A→B:           {results['mean_max_cosine_a_to_b']:.4f}")
        print(f"  Mean max|cos| B→A:           {results['mean_max_cosine_b_to_a']:.4f}")
    else:
        print(f"  [SKIP] Decoder cosine similarity: dim mismatch ({w_a.shape[0]} vs {w_b.shape[0]})")

    # L0 sparsity
    results["l0_a"] = (feat_a > 0).sum(axis=1).astype(float)
    results["l0_b"] = (feat_b > 0).sum(axis=1).astype(float)

    # Save
    save_data = {k: v for k, v in results.items() if isinstance(v, (float, int))}
    np.savez(
        os.path.join(args.output_dir, "cka_results.npz"),
        **{k: v for k, v in results.items() if isinstance(v, np.ndarray)},
        **{k: np.array(v) for k, v in save_data.items()},
    )

    import json
    with open(os.path.join(args.output_dir, "cka_metrics.json"), "w") as f:
        json.dump(save_data, f, indent=2)

    plot_cka_results(results, args.output_dir, label_a=args.label_a, label_b=args.label_b)
    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
