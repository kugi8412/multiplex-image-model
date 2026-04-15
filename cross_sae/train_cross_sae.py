"""Cross-sparse autoencoder: detect representational differences between models.

Workflow:
  1. Extract latents from both Model A and Model B on the same images.
  2. Train SAE_A on model A's latents and SAE_B on model B's latents.
  3. Cross-activation analysis:
     - Feed model A's latents through SAE_B → which SAE_B features fire?
     - Feed model B's latents through SAE_A → which SAE_A features fire?
  4. Identify:
     - **Shared features**: SAE features that activate similarly for both models.
     - **Unique-A features**: SAE_A features that don't transfer to model B.
     - **Unique-B features**: SAE_B features that don't transfer to model A.
  5. Visualise feature activation distributions, cosine similarity matrices,
     and spatial activation maps.

Usage::

    python -m cross_sae.train_cross_sae \\
        --config-a configs/train_vit_config.yaml \\
        --checkpoint-a checkpoints/vit_final.pth \\
        --config-b configs/train_mambaswin_config.yaml \\
        --checkpoint-b checkpoints/mambaswin_final.pth \\
        --sae-variant topk --k 32 --expansion 8 \\
        --epochs 30 --output-dir cross_sae_results/

    # Or compare ImmuVis vs VirTues:
    python -m cross_sae.train_cross_sae \\
        --config-a configs/train_vit_config.yaml \\
        --checkpoint-a checkpoints/vit_final.pth \\
        --source-b virtues \\
        --virtues-checkpoint path/to/virtues.pth \\
        --virtues-embeddings path/to/esm_embeddings/ \\
        --data-dir path/to/dataset/ \\
        --output-dir cross_sae_immuvis_vs_virtues/
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cross_sae.sparse_autoencoder import SAEOutput, build_sae
from cross_sae.train_sae import (
    extract_immuvis_latents,
    extract_virtues_latents,
    train_sae,
)


# ===================================================================
# Cross-activation analysis
# ===================================================================

@dataclass
class CrossActivationResult:
    """Results from cross-model SAE activation analysis."""

    # Per-feature stats: (hidden_dim,)
    mean_act_self: np.ndarray       # Mean activation when features from own model
    mean_act_cross: np.ndarray      # Mean activation on the other model's latents
    frac_active_self: np.ndarray    # Fraction of samples where feature fires (self)
    frac_active_cross: np.ndarray   # Fraction of samples where feature fires (cross)

    # Aggregate metrics
    shared_features: np.ndarray     # Indices of features active in both
    unique_features: np.ndarray     # Indices of features unique to self model


@torch.no_grad()
def cross_activate(
    sae: torch.nn.Module,
    latents_self: torch.Tensor,
    latents_cross: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    device: str = "cuda",
    batch_size: int = 4096,
    active_threshold: float = 0.01,
    shared_ratio_threshold: float = 0.3,
) -> CrossActivationResult:
    """Run cross-activation analysis for one SAE.

    Args:
        sae: Trained SAE module.
        latents_self: Latents from the model the SAE was trained on.
        latents_cross: Latents from the *other* model.
        latent_mean, latent_std: Normalisation stats from SAE training.
        active_threshold: Minimum activation to count as "active".
        shared_ratio_threshold: If cross_frac / self_frac > threshold,
            feature is considered "shared".

    Returns:
        CrossActivationResult with per-feature and aggregate stats.
    """
    sae = sae.to(device).eval()

    def _collect_stats(latents: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean_activation, frac_active) per feature."""
        N = latents.shape[0]
        hidden_dim = sae.hidden_dim if hasattr(sae, "hidden_dim") else sae.encoder.out_features
        sum_act = torch.zeros(hidden_dim, device=device)
        count_active = torch.zeros(hidden_dim, device=device)

        for start in range(0, N, batch_size):
            batch = latents[start : start + batch_size].to(device)
            batch = (batch - latent_mean.to(device)) / latent_std.to(device)
            h = sae.encode(batch)
            sum_act += h.sum(dim=0)
            count_active += (h > active_threshold).float().sum(dim=0)

        mean_act = (sum_act / N).cpu().numpy()
        frac_active = (count_active / N).cpu().numpy()
        return mean_act, frac_active

    mean_self, frac_self = _collect_stats(latents_self)
    mean_cross, frac_cross = _collect_stats(latents_cross)

    # Classify features
    # A feature is "shared" if it fires at least shared_ratio_threshold as
    # often on the cross model as it does on the self model
    self_active = frac_self > active_threshold
    ratio = np.where(frac_self > 0, frac_cross / (frac_self + 1e-8), 0.0)

    shared_mask = self_active & (ratio >= shared_ratio_threshold)
    unique_mask = self_active & (ratio < shared_ratio_threshold)

    return CrossActivationResult(
        mean_act_self=mean_self,
        mean_act_cross=mean_cross,
        frac_active_self=frac_self,
        frac_active_cross=frac_cross,
        shared_features=np.where(shared_mask)[0],
        unique_features=np.where(unique_mask)[0],
    )


# ===================================================================
# Decoder cosine similarity between two SAEs
# ===================================================================

def decoder_cosine_similarity(sae_a: torch.nn.Module, sae_b: torch.nn.Module) -> np.ndarray:
    """Compute cosine similarity between decoder columns of two SAEs.

    Returns:
        (hidden_dim_A, hidden_dim_B) cosine similarity matrix.
    """
    W_a = sae_a.W_dec if hasattr(sae_a, "W_dec") else sae_a.decoder.weight
    W_b = sae_b.W_dec if hasattr(sae_b, "W_dec") else sae_b.decoder.weight

    # Decoder weight shape: (input_dim, hidden_dim) — columns are feature directions
    w_a = F.normalize(W_a.detach().float(), dim=0)  # (input_dim, hidden_A)
    w_b = F.normalize(W_b.detach().float(), dim=0)  # (input_dim, hidden_B)

    sim = (w_a.t() @ w_b).cpu().numpy()  # (hidden_A, hidden_B)
    return sim


# ===================================================================
# Visualisation helpers
# ===================================================================

def plot_cross_activation_summary(
    result_a: CrossActivationResult,
    result_b: CrossActivationResult,
    output_dir: str,
    label_a: str = "Model A",
    label_b: str = "Model B",
):
    """Generate summary plots for cross-SAE analysis."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Feature activation frequency: self vs cross (SAE_A)
    ax = axes[0, 0]
    ax.scatter(result_a.frac_active_self, result_a.frac_active_cross,
               alpha=0.3, s=5, c="tab:blue")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel(f"Frac active on {label_a} latents")
    ax.set_ylabel(f"Frac active on {label_b} latents")
    ax.set_title(f"SAE trained on {label_a}: cross-activation")

    # 2. Feature activation frequency: self vs cross (SAE_B)
    ax = axes[0, 1]
    ax.scatter(result_b.frac_active_self, result_b.frac_active_cross,
               alpha=0.3, s=5, c="tab:orange")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel(f"Frac active on {label_b} latents")
    ax.set_ylabel(f"Frac active on {label_a} latents")
    ax.set_title(f"SAE trained on {label_b}: cross-activation")

    # 3. Shared vs unique features bar chart
    ax = axes[1, 0]
    labels = [label_a, label_b]
    shared = [len(result_a.shared_features), len(result_b.shared_features)]
    unique = [len(result_a.unique_features), len(result_b.unique_features)]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, shared, width, label="Shared", color="tab:green")
    ax.bar(x + width / 2, unique, width, label="Unique", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of features")
    ax.set_title("Shared vs. Unique SAE features")
    ax.legend()

    # 4. Mean activation ratio histogram
    ax = axes[1, 1]
    ratio_a = np.where(
        result_a.frac_active_self > 0,
        result_a.mean_act_cross / (result_a.mean_act_self + 1e-8),
        0.0,
    )
    ratio_b = np.where(
        result_b.frac_active_self > 0,
        result_b.mean_act_cross / (result_b.mean_act_self + 1e-8),
        0.0,
    )
    ax.hist(ratio_a[ratio_a > 0], bins=50, alpha=0.6, label=f"SAE_{label_a}", color="tab:blue")
    ax.hist(ratio_b[ratio_b > 0], bins=50, alpha=0.6, label=f"SAE_{label_b}", color="tab:orange")
    ax.axvline(1.0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Cross/Self mean activation ratio")
    ax.set_ylabel("Count")
    ax.set_title("Activation transfer ratio distribution")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cross_sae_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved summary plot to {output_dir}/cross_sae_summary.png")


def plot_decoder_similarity(sim_matrix: np.ndarray, output_dir: str,
                            label_a: str = "Model A", label_b: str = "Model B"):
    """Plot cosine similarity heatmap between two SAE decoder dictionaries."""
    os.makedirs(output_dir, exist_ok=True)

    # Sort by max similarity for visual clarity
    max_sim_per_row = np.abs(sim_matrix).max(axis=1)
    row_order = np.argsort(-max_sim_per_row)
    max_sim_per_col = np.abs(sim_matrix).max(axis=0)
    col_order = np.argsort(-max_sim_per_col)

    sorted_sim = sim_matrix[row_order][:, col_order]

    # Show top 200×200 for readability
    n = min(200, sorted_sim.shape[0], sorted_sim.shape[1])
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(sorted_sim[:n, :n], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel(f"SAE_{label_b} features (sorted)")
    ax.set_ylabel(f"SAE_{label_a} features (sorted)")
    ax.set_title(f"Decoder cosine similarity (top {n} features)")

    plt.savefig(os.path.join(output_dir, "decoder_cosine_similarity.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved decoder similarity plot to {output_dir}/decoder_cosine_similarity.png")


# ===================================================================
# CLI entry point
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train cross-sparse autoencoders and analyse model differences"
    )

    # Model A (always ImmuVis)
    parser.add_argument("--config-a", required=True, help="Config YAML for model A")
    parser.add_argument("--checkpoint-a", required=True, help="Checkpoint for model A")
    parser.add_argument("--label-a", default="ImmuVis-A", help="Label for model A in plots")

    # Model B — either ImmuVis or VirTues
    parser.add_argument(
        "--source-b", default="immuvis", choices=["immuvis", "virtues"],
        help="Source type for model B",
    )
    parser.add_argument("--config-b", default=None, help="Config YAML for model B (ImmuVis)")
    parser.add_argument("--checkpoint-b", default=None, help="Checkpoint for model B (ImmuVis)")
    parser.add_argument("--label-b", default="ImmuVis-B", help="Label for model B in plots")

    # VirTues-specific B args
    parser.add_argument("--virtues-checkpoint", default=None, help="VirTues checkpoint")
    parser.add_argument("--virtues-embeddings", default=None, help="ESM-2 embeddings dir")
    parser.add_argument("--data-dir", default=None, help="VirTues dataset directory")

    # SAE args
    parser.add_argument("--sae-variant", default="topk", choices=["vanilla", "topk", "gated"])
    parser.add_argument("--expansion", type=int, default=8)
    parser.add_argument("--l1-coeff", type=float, default=1e-3)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--aux-k", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-dir", default="cross_sae_results/")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Extract latents ----
    print("=" * 60)
    print("Extracting latents from Model A...")
    print("=" * 60)
    latents_a, dim_a = extract_immuvis_latents(
        args.config_a, args.checkpoint_a, device=device,
        max_batches=args.max_batches,
    )

    print("\n" + "=" * 60)
    print("Extracting latents from Model B...")
    print("=" * 60)
    if args.source_b == "immuvis":
        if not args.config_b or not args.checkpoint_b:
            parser.error("--config-b and --checkpoint-b required for ImmuVis model B")
        latents_b, dim_b = extract_immuvis_latents(
            args.config_b, args.checkpoint_b, device=device,
            max_batches=args.max_batches,
        )
    else:
        if not args.virtues_checkpoint or not args.virtues_embeddings or not args.data_dir:
            parser.error("VirTues model B requires --virtues-checkpoint, --virtues-embeddings, --data-dir")
        latents_b, dim_b = extract_virtues_latents(
            args.virtues_checkpoint, args.virtues_embeddings,
            args.data_dir, device=device, max_batches=args.max_batches,
        )
        args.label_b = "VirTues"

    # Dimensionality alignment check
    if dim_a != dim_b:
        print(f"Warning: latent dims differ (A={dim_a}, B={dim_b}). "
              f"Using projection to align B→A dimension.")
        # Simple linear projection trained on model B latents
        proj = torch.nn.Linear(dim_b, dim_a, bias=False)
        torch.nn.init.orthogonal_(proj.weight)
        latents_b = proj(latents_b).detach()
        dim_b = dim_a

    # ---- Build and train SAEs ----
    hidden_dim = dim_a * args.expansion
    sae_kwargs = {}
    if args.sae_variant in ("vanilla", "gated"):
        sae_kwargs["l1_coeff"] = args.l1_coeff
    if args.sae_variant == "topk":
        sae_kwargs["k"] = args.k
        sae_kwargs["aux_k"] = args.aux_k

    print(f"\nSAE config: {args.sae_variant} | dim={dim_a} | hidden={hidden_dim}")

    print("\n" + "=" * 60)
    print(f"Training SAE_A on {args.label_a} latents...")
    print("=" * 60)
    sae_a = build_sae(args.sae_variant, dim_a, hidden_dim, **sae_kwargs)
    history_a = train_sae(
        latents_a, sae_a, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, device=device,
    )

    print("\n" + "=" * 60)
    print(f"Training SAE_B on {args.label_b} latents...")
    print("=" * 60)
    sae_b = build_sae(args.sae_variant, dim_b, hidden_dim, **sae_kwargs)
    history_b = train_sae(
        latents_b, sae_b, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, device=device,
    )

    # ---- Cross-activation analysis ----
    print("\n" + "=" * 60)
    print("Running cross-activation analysis...")
    print("=" * 60)

    result_a = cross_activate(
        sae_a, latents_a, latents_b,
        history_a["latent_mean"], history_a["latent_std"],
        device=device,
    )
    result_b = cross_activate(
        sae_b, latents_b, latents_a,
        history_b["latent_mean"], history_b["latent_std"],
        device=device,
    )

    print(f"\nSAE_A ({args.label_a}):")
    print(f"  Shared features:  {len(result_a.shared_features)}")
    print(f"  Unique features:  {len(result_a.unique_features)}")
    print(f"SAE_B ({args.label_b}):")
    print(f"  Shared features:  {len(result_b.shared_features)}")
    print(f"  Unique features:  {len(result_b.unique_features)}")

    # ---- Decoder similarity ----
    sim_matrix = decoder_cosine_similarity(sae_a, sae_b)
    max_sim = np.abs(sim_matrix).max(axis=1)
    print(f"\nDecoder cosine similarity (SAE_A vs SAE_B):")
    print(f"  Mean max|cos|: {max_sim.mean():.4f}")
    print(f"  Features with |cos| > 0.8: {(max_sim > 0.8).sum()}")

    # ---- Save everything ----
    torch.save({
        "sae_a_state_dict": sae_a.state_dict(),
        "sae_b_state_dict": sae_b.state_dict(),
        "sae_variant": args.sae_variant,
        "input_dim": dim_a,
        "hidden_dim": hidden_dim,
        "sae_kwargs": sae_kwargs,
        "label_a": args.label_a,
        "label_b": args.label_b,
        "history_a": {k: v for k, v in history_a.items()},
        "history_b": {k: v for k, v in history_b.items()},
    }, os.path.join(args.output_dir, "cross_sae_checkpoint.pth"))

    np.savez(
        os.path.join(args.output_dir, "cross_activation_data.npz"),
        mean_act_self_a=result_a.mean_act_self,
        mean_act_cross_a=result_a.mean_act_cross,
        frac_active_self_a=result_a.frac_active_self,
        frac_active_cross_a=result_a.frac_active_cross,
        shared_a=result_a.shared_features,
        unique_a=result_a.unique_features,
        mean_act_self_b=result_b.mean_act_self,
        mean_act_cross_b=result_b.mean_act_cross,
        frac_active_self_b=result_b.frac_active_self,
        frac_active_cross_b=result_b.frac_active_cross,
        shared_b=result_b.shared_features,
        unique_b=result_b.unique_features,
        decoder_sim=sim_matrix,
    )

    # ---- Plots ----
    plot_cross_activation_summary(
        result_a, result_b, args.output_dir,
        label_a=args.label_a, label_b=args.label_b,
    )
    plot_decoder_similarity(
        sim_matrix, args.output_dir,
        label_a=args.label_a, label_b=args.label_b,
    )

    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
