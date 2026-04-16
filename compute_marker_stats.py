#!/usr/bin/env python
# -*- coding: utf-8 -*-
# compute_marker_stats.py

"""
Compute per-marker mean and std from the training data and update marker_metadata.csv.

Scans all panels and datasets defined in the panels config.  For every marker
in the tokenizer, aggregates pixel-level statistics across all images where
that marker appears, applying the same preprocessing used during training
(arcsinh → butterworth → clip normalization).

Usage::
    python compute_marker_stats.py
    python compute_marker_stats.py --panels-config configs/all_panels_config.yaml \
                                   --tokenizer-config configs/all_markers_tokenizer.yaml \
                                   --metadata-csv configs/marker_metadata.csv \
                                   --split train --max-images-per-dataset 0
"""

import argparse
import csv
import os
from glob import glob

import numpy as np
from ruamel.yaml import YAML
from skimage import filters
from tqdm import tqdm


def butterworth(img):
    filtered = [
        filters.butterworth(img[i], cutoff_frequency_ratio=0.2, high_pass=False)
        for i in range(img.shape[0])
    ]
    return np.stack(filtered)


def norm_clip(img, upper_bound):
    return np.clip(img, 0, upper_bound) / upper_bound


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-marker mean/std from training data and update marker_metadata.csv"
    )
    parser.add_argument(
        "--panels-config", default="configs/all_panels_config.yaml",
        help="Path to panels config YAML",
    )
    parser.add_argument(
        "--tokenizer-config", default="configs/all_markers_tokenizer.yaml",
        help="Path to the marker tokenizer YAML",
    )
    parser.add_argument(
        "--metadata-csv", default="configs/marker_metadata.csv",
        help="Path to marker_metadata.csv (will be updated in-place)",
    )
    parser.add_argument(
        "--split", default="train",
        help="Data split to compute statistics from (default: train)",
    )
    parser.add_argument(
        "--file-extension", default="npy", choices=["npy", "tiff"],
        help="Image file extension",
    )
    parser.add_argument(
        "--max-images-per-dataset", type=int, default=0,
        help="Max images to sample per dataset (0 = use all)",
    )
    parser.add_argument(
        "--use-butterworth", action="store_true", default=True,
        help="Apply butterworth filter (default: True)",
    )
    parser.add_argument(
        "--use-clip-normalization", action="store_true", default=True,
        help="Apply clip normalization (default: True)",
    )
    parser.add_argument(
        "--no-butterworth", action="store_false", dest="use_butterworth",
    )
    parser.add_argument(
        "--no-clip-normalization", action="store_false", dest="use_clip_normalization",
    )
    args = parser.parse_args()

    yaml = YAML(typ="safe")

    # Load configs
    with open(args.panels_config, "r") as f:
        panels_config = yaml.load(f)
    with open(args.tokenizer_config, "r") as f:
        tokenizer = yaml.load(f)

    img_path_root = panels_config["paths"][args.split]
    datasets = panels_config["datasets"]
    markers_per_dataset = panels_config["markers"]
    clip_limits = panels_config.get("clip_limits", {})
    global_upper_bound = 5.0

    read_func = np.load if args.file_extension == "npy" else __import__("tifffile").imread

    # --- Welford's online algorithm accumulators per marker ---
    # Using Welford's method to compute running mean and variance in a single
    # pass without storing all pixel values, keeping memory usage constant.
    marker_count = {}   # total number of pixels seen
    marker_mean = {}    # running mean
    marker_m2 = {}      # running sum of squared deviations

    all_marker_names = sorted(tokenizer.keys(), key=str.lower)
    for name in all_marker_names:
        marker_count[name] = 0
        marker_mean[name] = 0.0
        marker_m2[name] = 0.0

    print(f"Computing statistics from {len(datasets)} datasets in '{args.split}' split")
    print(f"Preprocessing: butterworth={args.use_butterworth}, clip_norm={args.use_clip_normalization}")

    for dataset in datasets:
        if dataset not in markers_per_dataset:
            print(f"  [SKIP] {dataset}: no markers defined in panels config")
            continue

        dataset_markers = markers_per_dataset[dataset]
        dataset_dir = os.path.join(img_path_root, dataset, "imgs")

        if not os.path.isdir(dataset_dir):
            print(f"  [SKIP] {dataset}: directory not found ({dataset_dir})")
            continue

        image_files = sorted(glob(os.path.join(dataset_dir, f"*.{args.file_extension}")))

        if args.max_images_per_dataset > 0 and len(image_files) > args.max_images_per_dataset:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(image_files), args.max_images_per_dataset, replace=False)
            image_files = [image_files[i] for i in sorted(indices)]

        if not image_files:
            print(f"  [SKIP] {dataset}: no images found in {dataset_dir}")
            continue

        clip_ub = clip_limits.get(dataset, global_upper_bound)

        print(f"  {dataset}: {len(image_files)} images, {len(dataset_markers)} markers, clip={clip_ub}")

        for img_file in tqdm(image_files, desc=f"    {dataset}", leave=False):
            img = read_func(img_file).astype(np.float32)

            # Same preprocessing as DatasetFromTIFF (use_preprocessing=False means
            # the data is already arcsinh-transformed from the folder name)
            if args.use_butterworth:
                img = butterworth(img)
            if args.use_clip_normalization:
                img = norm_clip(img, clip_ub)

            # img shape: (C, H, W) — C matches len(dataset_markers)
            if img.shape[0] != len(dataset_markers):
                print(f"    [WARN] {img_file}: channels={img.shape[0]}, "
                      f"expected={len(dataset_markers)}, skipping")
                continue

            for ch_idx, marker_name in enumerate(dataset_markers):
                if marker_name not in marker_count:
                    # Marker in panel but not in tokenizer — skip
                    continue

                channel_pixels = img[ch_idx].ravel().astype(np.float64)
                n_new = len(channel_pixels)
                if n_new == 0:
                    continue

                # Batch Welford update
                new_sum = channel_pixels.sum()
                new_mean = new_sum / n_new

                n_old = marker_count[marker_name]
                mean_old = marker_mean[marker_name]
                m2_old = marker_m2[marker_name]

                n_total = n_old + n_new
                delta = new_mean - mean_old
                new_mean_total = mean_old + delta * n_new / n_total

                # For M2: M2_new_batch = sum((x_i - new_mean_batch)^2)
                # = sum(x_i^2) - n_new * new_mean_batch^2
                new_sq_sum = (channel_pixels ** 2).sum()
                m2_new_batch = new_sq_sum - n_new * new_mean ** 2

                m2_total = m2_old + m2_new_batch + delta ** 2 * n_old * n_new / n_total

                marker_count[marker_name] = n_total
                marker_mean[marker_name] = new_mean_total
                marker_m2[marker_name] = m2_total

    # --- Compute final mean/std ---
    final_stats = {}
    for name in all_marker_names:
        n = marker_count[name]
        if n > 1:
            mean_val = marker_mean[name]
            std_val = np.sqrt(marker_m2[name] / (n - 1))
        elif n == 1:
            mean_val = marker_mean[name]
            std_val = 0.0
        else:
            mean_val = 0.0
            std_val = 0.0
        final_stats[name] = (mean_val, std_val)

    # --- Load existing metadata to preserve marker_id ordering ---
    existing_ids = {}
    if os.path.isfile(args.metadata_csv):
        with open(args.metadata_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mname = row["marker_name"].strip().strip('"')
                mid = int(row["marker_id"])
                existing_ids[mname] = mid
        print(f"\nLoaded {len(existing_ids)} existing marker IDs from {args.metadata_csv}")

    # For markers not in the existing CSV, assign sequential IDs from tokenizer
    for name in all_marker_names:
        if name not in existing_ids:
            existing_ids[name] = tokenizer[name]

    # --- Write updated CSV ---
    os.makedirs(os.path.dirname(args.metadata_csv) or ".", exist_ok=True)
    with open(args.metadata_csv, "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(["marker_name", "marker_id", "marker_mean", "marker_std"])
        for name in sorted(existing_ids.keys(), key=lambda k: existing_ids[k]):
            mid = existing_ids[name]
            mean_val, std_val = final_stats.get(name, (0.0, 0.0))
            writer.writerow([name, mid, f"{mean_val:.15g}", f"{std_val:.15g}"])

    # --- Summary ---
    markers_with_data = sum(1 for n in all_marker_names if marker_count[n] > 0)
    markers_no_data = sum(1 for n in all_marker_names if marker_count[n] == 0)
    total_pixels = sum(marker_count.values())

    print(f"\nDone! Updated {args.metadata_csv}")
    print(f"  Markers with data: {markers_with_data}")
    print(f"  Markers with no data: {markers_no_data}")
    print(f"  Total pixels processed: {total_pixels:,.0f}")

    if markers_no_data > 0:
        no_data = [n for n in all_marker_names if marker_count[n] == 0]
        print(f"  Markers missing data (first 20): {no_data[:20]}")


if __name__ == "__main__":
    main()
