#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare VirTues-preprocessed data from existing arcsinh(x/5) images.

Pipeline:
  1. Invert arcsinh: raw = 5 * sinh(arcsinh_data)
  2. Per-tissue 99th percentile quantile computation
  3. Quantile clip → log1p
  4. Per-tissue per-channel mean/std computation (in log1p space)
  5. z-standardize with computed mean/std
  6. Apply Gaussian blur (kernel=3, sigma=1.0)
  7. Save preprocessed images and statistics to new output folder

The output structure mirrors the input:
  {output_root}/{split}/{dataset}/imgs/*.npy
  {output_root}/stats/{dataset}/quantiles.csv
  {output_root}/stats/{dataset}/means.csv
  {output_root}/stats/{dataset}/stds.csv

Usage:
    python prepare_virtues_data.py \\
        --panel-config configs/all_panels_config.yaml \\
        --output-suffix _virtues

    This reads data from the paths in the panel config and writes to
    {original_path}_virtues/{split}/{dataset}/imgs/*.npy
"""

import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
from ruamel.yaml import YAML
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


def invert_arcsinh(data: np.ndarray) -> np.ndarray:
    """Invert arcsinh(x/5) transform: raw = 5 * sinh(data)."""
    return 5.0 * np.sinh(data)


def compute_tissue_quantiles(
    images: list[np.ndarray], quantile: float = 0.99
) -> np.ndarray:
    """Compute per-channel quantile across all images in a tissue/dataset.

    Args:
        images: List of arrays with shape (C, H, W).
        quantile: Quantile to compute (default 0.99 = 99th percentile).

    Returns:
        Array of shape (C,) with per-channel quantile values.
    """
    # Stack all pixels per channel
    all_pixels = np.concatenate(
        [img.reshape(img.shape[0], -1) for img in images], axis=1
    )  # (C, total_pixels)
    return np.quantile(all_pixels, quantile, axis=1)  # (C,)


def compute_log_stats(
    images: list[np.ndarray], quantiles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std in log1p space after quantile clipping.

    Args:
        images: List of raw intensity arrays (C, H, W).
        quantiles: Per-channel 99th percentile values (C,).

    Returns:
        (means, stds) each of shape (C,) in log1p space.
    """
    all_pixels = np.concatenate(
        [img.reshape(img.shape[0], -1) for img in images], axis=1
    )
    # Quantile clip
    q = quantiles[:, np.newaxis]
    clipped = np.clip(all_pixels, 0, q)
    # log1p transform
    logged = np.log1p(clipped)
    means = logged.mean(axis=1)
    stds = logged.std(axis=1)
    stds = np.where(stds < 1e-8, 1.0, stds)  # avoid division by zero
    return means, stds


def preprocess_image(
    raw: np.ndarray,
    quantiles: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    blur_sigma: float = 1.0,
) -> np.ndarray:
    """Apply full VirTues preprocessing to a single raw image.

    Pipeline: quantile clip → log1p → Gaussian blur → z-standardize.

    Args:
        raw: Raw intensity image (C, H, W).
        quantiles: Per-channel 99th percentile (C,).
        means: Per-channel log1p mean (C,).
        stds: Per-channel log1p std (C,).
        blur_sigma: Gaussian blur sigma.

    Returns:
        Preprocessed image (C, H, W) as float32.
    """
    C = raw.shape[0]
    q = quantiles[:, np.newaxis, np.newaxis]
    m = means[:, np.newaxis, np.newaxis]
    s = stds[:, np.newaxis, np.newaxis]

    # 1. Quantile clip
    img = np.clip(raw, 0, q)
    # 2. log1p
    img = np.log1p(img)
    # 3. Gaussian blur per channel
    for c in range(C):
        img[c] = gaussian_filter(img[c], sigma=blur_sigma)
    # 4. z-standardize
    img = (img - m) / (s + 1e-8)

    return img.astype(np.float32)


def process_dataset(
    input_dir: str,
    output_dir: str,
    stats_dir: str,
    file_extension: str = "npy",
    blur_sigma: float = 1.0,
) -> dict:
    """Process all images in a single dataset directory.

    Args:
        input_dir: Path to {split}/{dataset}/imgs/ with arcsinh .npy files.
        output_dir: Path to write preprocessed .npy files.
        stats_dir: Path to write quantiles/means/stds CSVs.
        file_extension: File extension (default 'npy').
        blur_sigma: Gaussian blur sigma.

    Returns:
        Dictionary with 'quantiles', 'means', 'stds' arrays.
    """
    pattern = os.path.join(input_dir, f"*.{file_extension}")
    file_paths = sorted(glob(pattern))
    if not file_paths:
        return None

    print(f"    Loading {len(file_paths)} images and inverting arcsinh...")
    raw_images = []
    for fp in tqdm(file_paths, desc="    Inverting", leave=False):
        arcsinh_data = np.load(fp)
        raw = invert_arcsinh(arcsinh_data)
        raw_images.append(raw)

    # Compute statistics on raw data
    print("    Computing 99th percentile quantiles...")
    quantiles = compute_tissue_quantiles(raw_images, quantile=0.99)

    print("    Computing log1p mean/std...")
    means, stds = compute_log_stats(raw_images, quantiles)

    # Save statistics
    os.makedirs(stats_dir, exist_ok=True)
    np.savetxt(os.path.join(stats_dir, "quantiles.csv"), quantiles, delimiter=",")
    np.savetxt(os.path.join(stats_dir, "means.csv"), means, delimiter=",")
    np.savetxt(os.path.join(stats_dir, "stds.csv"), stds, delimiter=",")

    # Preprocess and save each image
    os.makedirs(output_dir, exist_ok=True)
    print(f"    Preprocessing and saving to {output_dir}...")
    for fp, raw in tqdm(
        zip(file_paths, raw_images), total=len(file_paths),
        desc="    Processing", leave=False
    ):
        preprocessed = preprocess_image(raw, quantiles, means, stds, blur_sigma)
        out_name = os.path.basename(fp)
        np.save(os.path.join(output_dir, out_name), preprocessed)

    return {"quantiles": quantiles, "means": means, "stds": stds}


def main():
    parser = argparse.ArgumentParser(
        description="Prepare VirTues-preprocessed data from arcsinh images"
    )
    parser.add_argument(
        "--panel-config",
        default="configs/all_panels_config.yaml",
        help="Path to panel config YAML",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Absolute output root directory. If set, output is written to "
             "{output_root}/{split}/{dataset}/imgs/ instead of next to the source data.",
    )
    parser.add_argument(
        "--output-suffix",
        default="_virtues",
        help="Suffix to append to original data path for output (default: _virtues). "
             "Ignored when --output-root is set.",
    )
    parser.add_argument(
        "--file-extension",
        default="npy",
        help="File extension of input images (default: npy)",
    )
    parser.add_argument(
        "--blur-sigma",
        type=float,
        default=1.0,
        help="Gaussian blur sigma (default: 1.0)",
    )
    args = parser.parse_args()

    yaml = YAML(typ="safe")
    with open(args.panel_config, "r") as f:
        panel_config = yaml.load(f)

    datasets = panel_config["datasets"]
    paths = panel_config["paths"]

    for split, split_path in paths.items():
        if args.output_root is not None:
            output_root = os.path.join(args.output_root, split)
        else:
            output_root = split_path.rstrip("/") + args.output_suffix
        stats_root = os.path.join(output_root, "stats")
        print(f"\n=== Processing split: {split} ===")
        print(f"  Input:  {split_path}")
        print(f"  Output: {output_root}")

        for dataset in datasets:
            input_dir = os.path.join(split_path, dataset, "imgs")
            if not os.path.isdir(input_dir):
                print(f"  [{dataset}] SKIP — {input_dir} not found")
                continue

            output_dir = os.path.join(output_root, dataset, "imgs")
            stats_dir = os.path.join(stats_root, dataset)

            # Check if already preprocessed
            if os.path.isdir(output_dir) and len(glob(os.path.join(output_dir, f"*.{args.file_extension}"))) > 0:
                print(f"  [{dataset}] Already preprocessed — skipping")
                continue

            print(f"  [{dataset}] Processing...")
            result = process_dataset(
                input_dir, output_dir, stats_dir,
                file_extension=args.file_extension,
                blur_sigma=args.blur_sigma,
            )
            if result is None:
                print(f"  [{dataset}] No files found — skipping")


if __name__ == "__main__":
    main()
