#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare ImmukronosAutoencoder (finetuned decoder) vs ViT Baseline on HN dataset.

Runs leave-one-out validation for both models on the 'hn' dataset,
then produces a side-by-side comparison of per-marker MSE and Pearson R.

Usage:
  python compare_immukronos_vs_vit_hn.py \
      --immukronos-checkpoint checkpoints/immukronos_finetuned/best_model-<RUN>.pth \
      --vit-checkpoint checkpoints/exp7c_vit_baseline/best_model-<RUN>.pth \
      [--kronos-config configs/exp6a_immukronos_v2.yaml] \
      [--immukronos-decoder-config configs/finetune_immukronos_decoder.yaml] \
      [--vit-config configs/exp7c_vit_baseline.yaml] \
      [--output-dir results/immukronos_vs_vit_hn]
"""

import argparse
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_leave_one_out(
    model_type: str,
    config: str,
    checkpoint: str,
    model_label: str,
    results_dir: str,
    datasets: list[str],
    kronos_config: str | None = None,
    kronos_version: str = "v2",
    extra_args: list[str] | None = None,
):
    """Run run_validation_leave_one_out.py for a single model."""
    cmd = [
        sys.executable, "run_validation_leave_one_out.py",
        "--model-type", model_type,
        "--config", config,
        "--checkpoint", checkpoint,
        "--model-label", model_label,
        "--results-dir", results_dir,
        "--datasets", *datasets,
        "--save-reconstructions",
        "--recon-dir", os.path.join(results_dir, "recons"),
    ]
    if kronos_config:
        cmd.extend(["--kronos-config", kronos_config])
    if kronos_version != "v2":
        cmd.extend(["--kronos-version", kronos_version])
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"Running LOO validation: {model_label}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    subprocess.run(cmd, check=True)


def load_results(results_dir: str, model_label: str) -> pd.DataFrame:
    """Load LOO results CSV for a model."""
    safe_label = model_label.replace("/", "_").replace("\\", "_")
    csv_path = os.path.join(results_dir, f"{safe_label}_loo.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results not found: {csv_path}")
    return pd.read_csv(csv_path)


def plot_comparison(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str,
                    label_b: str, output_dir: str):
    """Create comparison plots: per-marker MSE and Pearson R."""

    # Per-marker aggregation
    stats_a = df_a.groupby("marker").agg(
        mse_mean=("mse", "mean"), mse_std=("mse", "std"),
        pearson_mean=("pearson", "mean"), pearson_std=("pearson", "std"),
    ).reset_index()
    stats_b = df_b.groupby("marker").agg(
        mse_mean=("mse", "mean"), mse_std=("mse", "std"),
        pearson_mean=("pearson", "mean"), pearson_std=("pearson", "std"),
    ).reset_index()

    # Merge on marker
    merged = stats_a.merge(stats_b, on="marker", suffixes=(f"_{label_a}", f"_{label_b}"))
    merged = merged.sort_values(f"mse_mean_{label_a}")

    markers = merged["marker"].values
    x = np.arange(len(markers))
    width = 0.35

    # --- MSE comparison ---
    fig, ax = plt.subplots(figsize=(max(14, len(markers) * 0.5), 6))
    ax.bar(x - width / 2, merged[f"mse_mean_{label_a}"], width,
           yerr=merged[f"mse_std_{label_a}"], label=label_a, alpha=0.85, capsize=2)
    ax.bar(x + width / 2, merged[f"mse_mean_{label_b}"], width,
           yerr=merged[f"mse_std_{label_b}"], label=label_b, alpha=0.85, capsize=2)
    ax.set_xlabel("Marker")
    ax.set_ylabel("MSE (leave-one-out)")
    ax.set_title(f"Per-Marker MSE — {label_a} vs {label_b} (HN dataset)")
    ax.set_xticks(x)
    ax.set_xticklabels(markers, rotation=60, ha="right", fontsize=8)
    ax.legend()
    fig.tight_layout()
    mse_path = os.path.join(output_dir, "comparison_mse_per_marker.png")
    fig.savefig(mse_path, dpi=150)
    print(f"Saved: {mse_path}")
    plt.close(fig)

    # --- Pearson R comparison ---
    fig, ax = plt.subplots(figsize=(max(14, len(markers) * 0.5), 6))
    merged_r = merged.sort_values(f"pearson_mean_{label_a}", ascending=False)
    markers_r = merged_r["marker"].values
    x_r = np.arange(len(markers_r))
    ax.bar(x_r - width / 2, merged_r[f"pearson_mean_{label_a}"], width,
           yerr=merged_r[f"pearson_std_{label_a}"], label=label_a, alpha=0.85, capsize=2)
    ax.bar(x_r + width / 2, merged_r[f"pearson_mean_{label_b}"], width,
           yerr=merged_r[f"pearson_std_{label_b}"], label=label_b, alpha=0.85, capsize=2)
    ax.set_xlabel("Marker")
    ax.set_ylabel("Pearson R (leave-one-out)")
    ax.set_title(f"Per-Marker Pearson R — {label_a} vs {label_b} (HN dataset)")
    ax.set_xticks(x_r)
    ax.set_xticklabels(markers_r, rotation=60, ha="right", fontsize=8)
    ax.legend()
    fig.tight_layout()
    pearson_path = os.path.join(output_dir, "comparison_pearson_per_marker.png")
    fig.savefig(pearson_path, dpi=150)
    print(f"Saved: {pearson_path}")
    plt.close(fig)

    # --- Summary table ---
    summary = pd.DataFrame({
        "model": [label_a, label_b],
        "mean_mse": [df_a["mse"].mean(), df_b["mse"].mean()],
        "std_mse": [df_a["mse"].std(), df_b["mse"].std()],
        "median_mse": [df_a["mse"].median(), df_b["mse"].median()],
        "mean_pearson": [df_a["pearson"].mean(), df_b["pearson"].mean()],
        "std_pearson": [df_a["pearson"].std(), df_b["pearson"].std()],
        "median_pearson": [df_a["pearson"].median(), df_b["pearson"].median()],
        "n_observations": [len(df_a), len(df_b)],
    })
    summary_path = os.path.join(output_dir, "comparison_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))

    # --- Per-marker detailed CSV ---
    detail_path = os.path.join(output_dir, "comparison_per_marker.csv")
    merged.to_csv(detail_path, index=False)
    print(f"\nSaved: {detail_path}")

    # --- Scatter: MSE of model A vs model B per marker ---
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(merged[f"mse_mean_{label_a}"], merged[f"mse_mean_{label_b}"],
               s=40, alpha=0.8)
    for i, marker in enumerate(merged["marker"]):
        ax.annotate(marker,
                    (merged[f"mse_mean_{label_a}"].iloc[i],
                     merged[f"mse_mean_{label_b}"].iloc[i]),
                    fontsize=6, alpha=0.7)
    lim = max(merged[f"mse_mean_{label_a}"].max(), merged[f"mse_mean_{label_b}"].max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel(f"MSE — {label_a}")
    ax.set_ylabel(f"MSE — {label_b}")
    ax.set_title("Per-Marker MSE Scatter")
    ax.legend()
    fig.tight_layout()
    scatter_path = os.path.join(output_dir, "comparison_mse_scatter.png")
    fig.savefig(scatter_path, dpi=150)
    print(f"Saved: {scatter_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare ImmukronosAutoencoder vs ViT Baseline on HN (leave-one-out)."
    )
    # ImmukronosAutoencoder
    parser.add_argument("--immukronos-checkpoint", required=True,
                        help="Finetuned ImmukronosAutoencoder checkpoint (.pth)")
    parser.add_argument("--kronos-config", default="configs/exp6a_immukronos_v2.yaml",
                        help="Original KRONOS training config")
    parser.add_argument("--immukronos-decoder-config",
                        default="configs/finetune_immukronos_decoder.yaml",
                        help="Decoder/finetuning config YAML")
    parser.add_argument("--kronos-version", default="v2", choices=["v2", "v3"])

    # ViT Baseline
    parser.add_argument("--vit-checkpoint", required=True,
                        help="ViT Baseline checkpoint (.pth)")
    parser.add_argument("--vit-config", default="configs/exp7c_vit_baseline.yaml",
                        help="ViT Baseline training config YAML")

    # General
    parser.add_argument("--output-dir", default="results/immukronos_vs_vit_hn")
    parser.add_argument("--datasets", nargs="+", default=["hn"],
                        help="Dataset(s) to evaluate on (default: hn)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation, only plot from existing CSVs")
    parser.add_argument("--immukronos-label", default="ImmuKronos_finetuned")
    parser.add_argument("--vit-label", default="ViT_baseline")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.skip_eval:
        # Run LOO for ImmukronosAutoencoder
        run_leave_one_out(
            model_type="immukronos_finetuned",
            config=args.immukronos_decoder_config,
            checkpoint=args.immukronos_checkpoint,
            model_label=args.immukronos_label,
            results_dir=args.output_dir,
            datasets=args.datasets,
            kronos_config=args.kronos_config,
            kronos_version=args.kronos_version,
        )

        # Run LOO for ViT Baseline
        run_leave_one_out(
            model_type="autoencoder",
            config=args.vit_config,
            checkpoint=args.vit_checkpoint,
            model_label=args.vit_label,
            results_dir=args.output_dir,
            datasets=args.datasets,
        )

    # Load and compare
    df_immukronos = load_results(args.output_dir, args.immukronos_label)
    df_vit = load_results(args.output_dir, args.vit_label)

    print(f"\nImmukronos results: {len(df_immukronos)} observations")
    print(f"ViT baseline results: {len(df_vit)} observations")

    plot_comparison(df_immukronos, df_vit, args.immukronos_label,
                    args.vit_label, args.output_dir)

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
