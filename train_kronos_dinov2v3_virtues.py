#!/usr/bin/env python
# -*- coding: utf-8 -*-
# train_kronos_dinov2v3_virtues.py


"""
Train KRONOS ViT backbone with DINOv2/DINOv3 self-distillation using
**VirTues-style preprocessing** (Wenckstern et al., 2025).

Preprocessing pipeline (per VirTues):
  1. Quantile clipping at the 99th tissue percentile
  2. log1p transform
  3. Gaussian blur (kernel=3, sigma=1.0)
  4. Channel-wise z-standardization (per-tissue log-space mean/std)
  5. Channel dropout augmentation (randomly drop a fraction of channels)

Data layout follows VirTues conventions:
  - Pre-cropped .npy files in a crops/ directory
  - Per-dataset CSV files: channels.csv, quantiles.csv, means.csv, stds.csv,
    tissue_index.csv, crop_index.csv

Falls back to the existing ImmuVis DatasetFromTIFF pipeline when
``use_virtues_preprocessing: false`` in the config.

Usage::
    python train_kronos_dinov2v3_virtues.py configs/train_kronos_dinov2_virtues_config.yaml
    python train_kronos_dinov2v3_virtues.py configs/train_kronos_dinov3_virtues_config.yaml
"""

import argparse
import csv
import functools
import math
import os
import random
from glob import glob

import comet_ml  # noqa: F401
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ruamel.yaml import YAML
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.transforms import v2 as T
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomVerticalFlip,
)
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from multiplex_model.utils import init_experiment, finish_experiment, get_run_name
from multiplex_model.kronos.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from multiplex_model.kronos.dino_head import DINOHead

# Re-use marker embedding resolution from the original script
from train_kronos_dinov2v3 import (
    resolve_marker_embeddings,
    build_tokenizer_from_marker_ids,
    load_marker_metadata,
    KronosDINOv2v3,
    DINOLoss,
    iBOTLoss,
    KoLeoLoss,
    generate_spatial_ibot_mask,
    SpatialProteomicsMultiCrop,
    cosine_scheduler,
    MODEL_CONFIGS,
    build_backbone,
)


# -------------------------------------------
# 1. VIRTUES-STYLE AUGMENTATIONS
# -------------------------------------------

class MultiplexRandomCrop:
    """Random spatial crop for (C, H, W) multiplex images."""

    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, image):
        _, H, W = image.shape
        th, tw = self.size
        r = random.randint(0, max(H - th, 0))
        c = random.randint(0, max(W - tw, 0))
        return image[:, r:r + th, c:c + tw]


class MultiplexRandomSymmetry:
    """Random horizontal/vertical flip + 90-degree rotation for multiplex images."""

    def __call__(self, image):
        if random.random() > 0.5:
            image = torch.flip(image, dims=[-1])
        if random.random() > 0.5:
            image = torch.flip(image, dims=[-2])
        if random.random() > 0.5:
            image = image.permute(0, 2, 1)
        return image


class ChannelDropout:
    """Randomly drop a fraction of channels, setting them to zero.

    Args:
        channel_fraction: (min_keep, max_keep) — fraction of channels to keep.
    """

    def __init__(self, channel_fraction=(0.75, 1.0)):
        self.channel_fraction = channel_fraction

    def __call__(self, multiplex, marker_indices):
        C = multiplex.shape[0]
        frac = random.uniform(*self.channel_fraction)
        n_keep = max(1, int(C * frac))
        if n_keep >= C:
            return multiplex, marker_indices
        perm = torch.randperm(C)[:n_keep].sort().values
        return multiplex[perm], marker_indices[perm]


# -------------------------------------------
# 2. VIRTUES-STYLE DATASET
# -------------------------------------------

class VirtuesMultiplexDataset(Dataset):
    """Load pre-cropped multiplex images with VirTues-style preprocessing.

    Preprocessing:
      1. Quantile clip at 99th percentile per tissue
      2. log1p
      3. Gaussian blur (k=3, sigma=1)
      4. z-standardization with per-tissue log-space mean/std
    """

    def __init__(
        self,
        crop_dir,
        tissue_index_csv,
        crop_index_csv,
        channels_csv,
        quantiles_csv,
        means_csv,
        stds_csv,
        marker_tokenizer,
        split="train",
        crop_size=128,
        channel_fraction=(0.75, 1.0),
    ):
        self.crop_dir = crop_dir
        self.tissue_index = pd.read_csv(tissue_index_csv)
        self.crop_index = pd.read_csv(crop_index_csv)
        self.channels = pd.read_csv(channels_csv)
        self.quantiles = pd.read_csv(quantiles_csv, index_col=0)
        self.means = pd.read_csv(means_csv, index_col=0)
        self.stds = pd.read_csv(stds_csv, index_col=0)

        # Filter by split
        if split != "all":
            self.tissue_index = self.tissue_index.query(f'split == "{split}"')
            self.crop_index = self.crop_index[
                self.crop_index["tissue_id"].isin(self.tissue_index["tissue_id"])
            ]

        # Build channel mask and marker indices from tokenizer
        self.channel_mask = []
        self.marker_indices = []
        for _, row in self.channels.iterrows():
            marker_name = row.get("marker_name", row.get("protein_id", ""))
            if marker_name in marker_tokenizer:
                self.channel_mask.append(True)
                self.marker_indices.append(marker_tokenizer[marker_name])
            else:
                self.channel_mask.append(False)
        self.channel_mask = torch.tensor(self.channel_mask, dtype=torch.bool)
        self.marker_indices = torch.tensor(self.marker_indices, dtype=torch.long)

        self.crop_size = crop_size
        self.random_crop = MultiplexRandomCrop(size=(crop_size, crop_size))
        self.random_symmetry = MultiplexRandomSymmetry()
        self.gaussian_blur = T.GaussianBlur(kernel_size=3, sigma=1.0)
        self.drop_channels = ChannelDropout(channel_fraction=channel_fraction)

    def __len__(self):
        return len(self.crop_index)

    def __getitem__(self, idx):
        row = self.crop_index.iloc[idx]
        tissue_id = row["tissue_id"]
        crop_id = row["crop_id"]

        path = os.path.join(self.crop_dir, f"{tissue_id}_{crop_id}.npy")
        multiplex = np.load(path)
        multiplex = torch.tensor(multiplex, dtype=torch.float32)
        multiplex = multiplex[self.channel_mask]

        multiplex = self._preprocess(tissue_id, multiplex)
        marker_indices = self.marker_indices.clone()

        multiplex, marker_indices = self._augment(multiplex, marker_indices)

        return multiplex, marker_indices

    def _preprocess(self, tissue_id, multiplex):
        # 1. Quantile clipping
        q_vals = self.quantiles.loc[tissue_id].values[self.channel_mask.numpy()]
        q_vals = torch.from_numpy(q_vals).float()[:, None, None]
        multiplex = torch.clamp(multiplex, min=torch.zeros_like(q_vals), max=q_vals)

        # 2. log1p
        multiplex = torch.log1p(multiplex)

        # 3. Gaussian blur
        multiplex = self.gaussian_blur(multiplex)

        # 4. z-standardization in log-space
        log_mean = self.means.loc[tissue_id].values[self.channel_mask.numpy()]
        log_std = self.stds.loc[tissue_id].values[self.channel_mask.numpy()]
        log_mean = torch.from_numpy(log_mean).float()[:, None, None]
        log_std = torch.from_numpy(log_std).float()[:, None, None]
        multiplex = (multiplex - log_mean) / (log_std + 1e-8)

        return multiplex

    def _augment(self, multiplex, marker_indices):
        multiplex = self.random_crop(multiplex)
        multiplex = self.random_symmetry(multiplex)
        # Channel dropout is applied per-batch in the collate function,
        # NOT per-sample, to keep channel counts consistent within a batch.
        return multiplex, marker_indices


# -------------------------------------------
# 3. MULTI-CROP WRAPPER FOR VIRTUES DATASET
# -------------------------------------------

class VirtuesDINODatasetWrapper(Dataset):
    """Wraps VirtuesMultiplexDataset with multi-crop augmentation for DINO training."""

    def __init__(self, base_dataset, multi_crop_transform):
        self.base_dataset = base_dataset
        self.multi_crop = multi_crop_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, marker_indices = self.base_dataset[idx]
        crops = self.multi_crop(img)
        return crops, marker_indices


def virtues_dino_collate_fn(batch, channel_fraction=(0.75, 1.0)):
    """Collate for VirTues DINO dataset with batch-level channel dropout.

    All items in a batch get the same random channel subset so that
    ``torch.stack`` on crops and channel_ids works without padding.

    Note: assumes all items in a batch have the same channel count
    (e.g., from same dataset/panel). Use PanelBatchSampler or single-panel
    ConcatDataset to ensure this.
    """
    num_crops = len(batch[0][0])
    C = batch[0][1].shape[0]

    # Batch-level channel dropout: sample one fraction for the whole batch
    frac = random.uniform(*channel_fraction)
    n_keep = max(1, int(C * frac))
    if n_keep < C:
        perm = torch.randperm(C)[:n_keep].sort().values
    else:
        perm = None  # keep all

    collated_crops = []
    for i in range(num_crops):
        crops = torch.stack([item[0][i] for item in batch])  # (B, C, H, W)
        if perm is not None:
            crops = crops[:, perm]
        collated_crops.append(crops)

    channel_ids = torch.stack([item[1] for item in batch])    # (B, C)
    if perm is not None:
        channel_ids = channel_ids[:, perm]

    return collated_crops, channel_ids


# -------------------------------------------
# 4. IMMUVIS FALLBACK DATASET
# -------------------------------------------

def build_immuvis_dataloader(config, PANEL_CONFIG, TOKENIZER, train_transform):
    """Build DataLoader using the existing ImmuVis DatasetFromTIFF pipeline."""
    from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler

    class DINODatasetWrapper(Dataset):
        def __init__(self, base_dataset, transform):
            self.base_dataset = base_dataset
            self.transform = transform

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, idx):
            img, channel_ids, dataset_name, img_path = self.base_dataset[idx]
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()
            elif img.dtype != torch.float32:
                img = img.float()
            crops = self.transform(img)
            return crops, channel_ids, dataset_name, img_path

    def dino_collate_fn(batch):
        num_crops = len(batch[0][0])
        collated_crops = [torch.stack([item[0][i] for item in batch]) for i in range(num_crops)]
        channel_ids = torch.stack([item[1] for item in batch])
        dataset_names = [item[2] for item in batch]
        img_paths = [item[3] for item in batch]
        return collated_crops, channel_ids, dataset_names, img_paths

    train_dataset_base = DatasetFromTIFF(
        panels_config=PANEL_CONFIG,
        split="train",
        marker_tokenizer=TOKENIZER,
        transform=None,
        use_preprocessing=False,
        use_butterworth_filter=True,
        use_clip_normalization=True,
        file_extension=config.get("file_extension", "npy"),
    )

    train_dataset = DINODatasetWrapper(train_dataset_base, train_transform)
    train_batch_sampler = PanelBatchSampler(train_dataset_base, config["batch_size"])

    return DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=config.get("num_workers", 4),
        collate_fn=dino_collate_fn,
        pin_memory=True,
        persistent_workers=config.get("num_workers", 4) > 0,
        prefetch_factor=4 if config.get("num_workers", 4) > 0 else None,
    ), True  # True = immuvis-style collate (4-tuples)


# -------------------------------------------
# 4b. VIRTUES-LITE ON ARCSINH DATA
# -------------------------------------------

def _check_virtues_precomputed(panel_config: dict, suffix: str = "_virtues",
                               explicit_path: str | None = None) -> str | None:
    """Check if VirTues-preprocessed data already exists.

    Looks for data at either an explicit path or {original_train_path}{suffix}/
    with the same dataset subdirectories containing .npy files.

    Args:
        panel_config: Panel configuration dict with 'paths' and 'datasets'.
        suffix: Suffix to append to original data path.
        explicit_path: If set, check this path directly (for --output-root style).

    Returns:
        The path to precomputed data root if found, otherwise None.
    """
    candidates = []
    if explicit_path is not None:
        # Check explicit_path/train/ structure
        candidates.append(os.path.join(explicit_path, "train"))
        candidates.append(explicit_path)
    train_path = panel_config["paths"].get("train", "")
    candidates.append(train_path.rstrip("/") + suffix)

    for virtues_path in candidates:
        if not os.path.isdir(virtues_path):
            continue
        for dataset in panel_config.get("datasets", []):
            imgs_dir = os.path.join(virtues_path, dataset, "imgs")
            if os.path.isdir(imgs_dir) and len(glob(os.path.join(imgs_dir, "*.npy"))) > 0:
                return virtues_path
    return None


def build_arcsinh_virtues_dataloader(config, PANEL_CONFIG, TOKENIZER,
                                     marker_id_map, train_transform):
    """Load arcsinh data via DatasetFromTIFF, apply VirTues-style augmentations.

    If pre-computed VirTues data exists (created by prepare_virtues_data.py),
    loads it directly with no additional preprocessing.
    Otherwise falls back to runtime preprocessing:
    load arcsinh .npy → Gaussian blur → z-standardize (using
    marker_metadata.csv stats) → channel dropout → multi-crop.
    """
    from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler

    virtues_suffix = config.get("virtues_data_suffix", "_virtues")
    virtues_data_root = config.get("virtues_data_root", None)
    precomputed_path = _check_virtues_precomputed(
        PANEL_CONFIG, suffix=virtues_suffix, explicit_path=virtues_data_root
    )

    if precomputed_path is not None:
        print(f"[VirTues] Found pre-computed data at: {precomputed_path}")
        print("[VirTues] Loading directly — no runtime preprocessing needed.")

        # Build a modified panel config pointing to the precomputed path
        virtues_panel_config = dict(PANEL_CONFIG)
        virtues_panel_config["paths"] = {}
        for split, original_path in PANEL_CONFIG["paths"].items():
            virtues_panel_config["paths"][split] = original_path.rstrip("/") + virtues_suffix

        channel_fraction = tuple(config.get("channel_fraction", [0.75, 1.0]))

        class PrecomputedVirtuesDINOWrapper(Dataset):
            """Load pre-computed VirTues data — no additional preprocessing."""

            def __init__(self, base_dataset, transform):
                self.base_dataset = base_dataset
                self.transform = transform

            def __len__(self):
                return len(self.base_dataset)

            def __getitem__(self, idx):
                img, channel_ids, dataset_name, img_path = self.base_dataset[idx]
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img).float()
                elif img.dtype != torch.float32:
                    img = img.float()
                crops = self.transform(img)
                return crops, channel_ids, dataset_name, img_path

        def dino_collate_fn(batch):
            num_crops = len(batch[0][0])
            C = batch[0][1].shape[0]
            frac = random.uniform(*channel_fraction)
            n_keep = max(1, int(C * frac))
            perm = torch.randperm(C)[:n_keep].sort().values if n_keep < C else None

            collated_crops = []
            for i in range(num_crops):
                crops = torch.stack([item[0][i] for item in batch])
                if perm is not None:
                    crops = crops[:, perm]
                collated_crops.append(crops)

            channel_ids = torch.stack([item[1] for item in batch])
            if perm is not None:
                channel_ids = channel_ids[:, perm]

            dataset_names = [item[2] for item in batch]
            img_paths = [item[3] for item in batch]
            return collated_crops, channel_ids, dataset_names, img_paths

        train_dataset_base = DatasetFromTIFF(
            panels_config=virtues_panel_config,
            split="train",
            marker_tokenizer=TOKENIZER,
            transform=None,
            use_preprocessing=False,       # already preprocessed
            use_butterworth_filter=False,   # already done
            use_clip_normalization=False,   # already done
            file_extension=config.get("file_extension", "npy"),
        )

        train_dataset = PrecomputedVirtuesDINOWrapper(train_dataset_base, train_transform)
        train_batch_sampler = PanelBatchSampler(train_dataset_base, config["batch_size"])

        return DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=config.get("num_workers", 4),
            collate_fn=dino_collate_fn,
            pin_memory=True,
            persistent_workers=config.get("num_workers", 4) > 0,
            prefetch_factor=4 if config.get("num_workers", 4) > 0 else None,
        ), True

    # --- Fallback: runtime preprocessing from arcsinh data ---
    print("[VirTues] No pre-computed data found — applying runtime preprocessing.")
    print("[VirTues] Hint: Run 'python prepare_virtues_data.py --panel-config "
          f"{config.get('panel_config', 'configs/all_panels_config.yaml')}' to pre-compute.")

    # Load per-marker mean/std from marker_metadata.csv
    metadata_path = config.get("marker_metadata_csv", "configs/marker_metadata.csv")
    marker_stats = {}  # marker_name -> (mean, std)
    if os.path.isfile(metadata_path):
        import csv as csv_mod
        with open(metadata_path, "r", newline="") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                name = row["marker_name"].strip().strip('"')
                marker_stats[name] = (
                    float(row["marker_mean"]),
                    float(row["marker_std"]),
                )
    else:
        print(f"[WARNING] {metadata_path} not found — z-standardization disabled")

    gaussian_blur = T.GaussianBlur(kernel_size=3, sigma=1.0)
    channel_fraction = tuple(config.get("channel_fraction", [0.75, 1.0]))

    # Reverse tokenizer to get marker names from IDs
    inv_tokenizer = {v: k for k, v in TOKENIZER.items()}

    class ArcsinhVirtuesDINOWrapper(Dataset):
        """DatasetFromTIFF + Gaussian blur + z-standardize (no per-sample dropout)."""

        def __init__(self, base_dataset, transform):
            self.base_dataset = base_dataset
            self.transform = transform

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, idx):
            img, channel_ids, dataset_name, img_path = self.base_dataset[idx]
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()
            elif img.dtype != torch.float32:
                img = img.float()

            # 1. Gaussian blur (VirTues-style)
            img = gaussian_blur(img)

            # 2. Per-channel z-standardization using marker_metadata.csv stats
            if marker_stats:
                for c_idx in range(img.shape[0]):
                    mid = channel_ids[c_idx].item()
                    mname = inv_tokenizer.get(mid)
                    if mname and mname in marker_stats:
                        m, s = marker_stats[mname]
                        if s > 1e-8:
                            img[c_idx] = (img[c_idx] - m) / s

            # Channel dropout applied per-batch in collate, not per-sample
            crops = self.transform(img)
            return crops, channel_ids, dataset_name, img_path

    def dino_collate_fn(batch):
        """Collate with batch-level channel dropout for arcsinh VirTues path."""
        num_crops = len(batch[0][0])
        C = batch[0][1].shape[0]

        # Batch-level channel dropout
        frac = random.uniform(*channel_fraction)
        n_keep = max(1, int(C * frac))
        if n_keep < C:
            perm = torch.randperm(C)[:n_keep].sort().values
        else:
            perm = None

        collated_crops = []
        for i in range(num_crops):
            crops = torch.stack([item[0][i] for item in batch])
            if perm is not None:
                crops = crops[:, perm]
            collated_crops.append(crops)

        channel_ids = torch.stack([item[1] for item in batch])
        if perm is not None:
            channel_ids = channel_ids[:, perm]

        dataset_names = [item[2] for item in batch]
        img_paths = [item[3] for item in batch]
        return collated_crops, channel_ids, dataset_names, img_paths

    train_dataset_base = DatasetFromTIFF(
        panels_config=PANEL_CONFIG,
        split="train",
        marker_tokenizer=TOKENIZER,
        transform=None,
        use_preprocessing=False,
        use_butterworth_filter=False,   # no butterworth — VirTues uses Gaussian blur
        use_clip_normalization=False,    # no clip — z-standardize instead
        file_extension=config.get("file_extension", "npy"),
    )

    train_dataset = ArcsinhVirtuesDINOWrapper(train_dataset_base, train_transform)
    train_batch_sampler = PanelBatchSampler(train_dataset_base, config["batch_size"])

    return DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=config.get("num_workers", 4),
        collate_fn=dino_collate_fn,
        pin_memory=True,
        persistent_workers=config.get("num_workers", 4) > 0,
        prefetch_factor=4 if config.get("num_workers", 4) > 0 else None,
    ), True  # True = immuvis-style collate (4-tuples)


def build_virtues_dataloader(config, TOKENIZER, train_transform):
    """Build DataLoader using VirTues-style preprocessing from dataset configs."""
    datasets_config_path = config.get("virtues_datasets_config")
    if not datasets_config_path:
        raise ValueError("virtues_datasets_config must be set in the config YAML "
                         "when use_virtues_preprocessing is enabled.")

    yaml = YAML(typ="safe")
    with open(datasets_config_path, "r") as f:
        ds_conf = yaml.load(f)

    all_datasets = []
    for ds_name, ds_cfg in ds_conf["datasets"].items():
        dataset = VirtuesMultiplexDataset(
            crop_dir=ds_cfg["crop_dir"],
            tissue_index_csv=ds_cfg["tissue_index"],
            crop_index_csv=ds_cfg["crop_index"],
            channels_csv=ds_cfg["channels_file"],
            quantiles_csv=ds_cfg["quantiles_file"],
            means_csv=ds_cfg["means_file"],
            stds_csv=ds_cfg["stds_file"],
            marker_tokenizer=TOKENIZER,
            split="train",
            crop_size=config.get("global_crops_size", [128, 128])[0]
            if isinstance(config.get("global_crops_size", 128), list)
            else config.get("global_crops_size", 128),
            channel_fraction=tuple(config.get("channel_fraction", [0.75, 1.0])),
        )
        all_datasets.append(dataset)
        print(f"  [VirTues] {ds_name}: {len(dataset)} crops")

    merged = ConcatDataset(all_datasets) if len(all_datasets) > 1 else all_datasets[0]
    wrapped = VirtuesDINODatasetWrapper(merged, train_transform)

    channel_fraction = tuple(config.get("channel_fraction", [0.75, 1.0]))
    collate_fn = functools.partial(
        virtues_dino_collate_fn, channel_fraction=channel_fraction
    )

    return DataLoader(
        wrapped,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.get("num_workers", 4) > 0,
        prefetch_factor=4 if config.get("num_workers", 4) > 0 else None,
    ), False  # False = virtues mode (collate returns 2-tuples)


# -------------------------------------------
# 5. MAIN TRAINING LOOP
# -------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KRONOS DINOv2/v3 training with VirTues-style preprocessing"
    )
    parser.add_argument("config", help="Path to config YAML")
    parser.add_argument("--from-checkpoint", default=None, help="Resume from checkpoint path")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    yaml = YAML(typ="safe")
    with open(args.config, "r") as f:
        config = yaml.load(f)

    # Validate this is a KRONOS DINOv2/v3 config, not a masked-model config
    REQUIRED_KEYS = ["global_crops_scale", "global_crops_size", "local_crops_scale",
                     "local_crops_size", "model_name", "out_dim",
                     "teacher_temp", "teacher_momentum"]
    missing = [k for k in REQUIRED_KEYS if k not in config]
    if missing:
        if "encoder" in config or "decoder" in config:
            raise SystemExit(
                f"ERROR: '{args.config}' is a masked-model config (has encoder/decoder),\n"
                f"but you ran train_kronos_dinov2v3_virtues.py which needs DINO config.\n"
                f"Use 'python train_masked_model.py {args.config}' instead.\n"
                f"Missing keys: {missing}"
            )
        raise SystemExit(
            f"ERROR: Config '{args.config}' is missing required keys: {missing}\n"
            f"See configs/train_kronos_dinov2_virtues_config.yaml for an example."
        )

    device = torch.device(args.device or config.get("device", "cuda"))
    training_mode = config.get("training_mode", "dinov2")
    assert training_mode in ("dinov2", "dinov3"), (
        f"training_mode must be 'dinov2' or 'dinov3', got '{training_mode}'"
    )
    preprocessing_mode = config.get("preprocessing_mode", None)
    if preprocessing_mode is None:
        # Legacy fallback: use_virtues_preprocessing boolean
        use_virtues = config.get("use_virtues_preprocessing", True)
        preprocessing_mode = "virtues" if use_virtues else "immuvis"
    assert preprocessing_mode in ("virtues", "immuvis", "virtues_arcsinh"), (
        f"preprocessing_mode must be 'virtues', 'immuvis', or 'virtues_arcsinh', "
        f"got '{preprocessing_mode}'"
    )
    print(f"Training mode: {training_mode.upper()} | Device: {device} "
          f"| Preprocessing: {preprocessing_mode}")

    # ---- Marker embedding resolution ----
    TOKENIZER_RAW = YAML().load(open(config["tokenizer_config"]))
    marker_id_map, num_markers = resolve_marker_embeddings(config, TOKENIZER_RAW)
    TOKENIZER, num_markers = build_tokenizer_from_marker_ids(TOKENIZER_RAW, marker_id_map)

    n_global = config.get("global_crops_number", 2)
    n_local = config.get("local_crops_number", 8)
    ncrops = n_global + n_local

    train_transform = SpatialProteomicsMultiCrop(
        global_scale=config["global_crops_scale"],
        local_scale=config["local_crops_scale"],
        global_size=config["global_crops_size"],
        local_size=config["local_crops_size"],
        local_crops_num=n_local,
        global_crops_num=n_global,
    )

    # ---- Data (VirTues / ImmuVis / VirTues-Lite on arcsinh) ----
    if preprocessing_mode == "virtues":
        train_dataloader, immuvis_mode = build_virtues_dataloader(
            config, TOKENIZER, train_transform
        )
    elif preprocessing_mode == "virtues_arcsinh":
        PANEL_CONFIG = YAML().load(open(config["panel_config"]))
        train_dataloader, immuvis_mode = build_arcsinh_virtues_dataloader(
            config, PANEL_CONFIG, TOKENIZER, marker_id_map, train_transform
        )
    else:
        PANEL_CONFIG = YAML().load(open(config["panel_config"]))
        train_dataloader, immuvis_mode = build_immuvis_dataloader(
            config, PANEL_CONFIG, TOKENIZER, train_transform
        )

    # ---- Model ----
    model_name = config["model_name"]
    patch_size = config.get("patch_size", 16)
    img_size = config.get("global_crops_size", [128, 128])
    if isinstance(img_size, list):
        img_size = img_size[0]
    num_register_tokens = config.get("num_register_tokens", 0)
    ffn_layer = config.get("ffn_layer", "mlp")
    init_values = config.get("init_values", None)

    ibot_out_dim = config.get("ibot_out_dim", None) if training_mode == "dinov3" else None

    backbone_student, embed_dim = build_backbone(
        model_name, num_markers, img_size, patch_size,
        drop_path_rate=config.get("drop_path_rate", 0.1),
        num_register_tokens=num_register_tokens,
        ffn_layer=ffn_layer,
        init_values=init_values,
    )
    backbone_teacher, _ = build_backbone(
        model_name, num_markers, img_size, patch_size,
        drop_path_rate=0.0,
        num_register_tokens=num_register_tokens,
        ffn_layer=ffn_layer,
        init_values=init_values,
    )

    student = KronosDINOv2v3(backbone_student, embed_dim, config["out_dim"], ibot_out_dim).to(device)
    teacher = KronosDINOv2v3(backbone_teacher, embed_dim, config["out_dim"], ibot_out_dim).to(device)

    teacher.load_state_dict(student.state_dict(), strict=False)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    # ---- Losses ----
    dino_loss_fn = DINOLoss(
        out_dim=config["out_dim"],
        ncrops=ncrops,
        warmup_teacher_temp=config["warmup_teacher_temp"],
        teacher_temp=config["teacher_temp"],
        warmup_teacher_temp_epochs=config["warmup_teacher_temp_epochs"],
        nepochs=config["epochs"],
        nglobal_crops=n_global,
    ).to(device)

    ibot_loss_fn = None
    koleo_loss_fn = None

    if training_mode == "dinov3":
        ibot_loss_fn = iBOTLoss(
            out_dim=ibot_out_dim,
            warmup_teacher_temp=config.get("ibot_warmup_teacher_temp",
                                           config["warmup_teacher_temp"]),
            teacher_temp=config.get("ibot_teacher_temp", config["teacher_temp"]),
            warmup_teacher_temp_epochs=config["warmup_teacher_temp_epochs"],
            nepochs=config["epochs"],
        ).to(device)
        koleo_loss_fn = KoLeoLoss()

    # ---- Optimizer ----
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    niter_per_ep = len(train_dataloader)
    lr_schedule = cosine_scheduler(
        config["lr"], config["final_lr"], config["epochs"],
        niter_per_ep, config["warmup_epochs"],
    )
    wd_schedule = cosine_scheduler(
        config["weight_decay"], config["weight_decay_final"],
        config["epochs"], niter_per_ep,
    )
    momentum_schedule = cosine_scheduler(
        config["teacher_momentum"], 1.0, config["epochs"], niter_per_ep,
    )

    # ---- Checkpoint resume ----
    start_epoch = 0
    ckpt_path = args.from_checkpoint or config.get("from_checkpoint")

    use_fp16 = config.get("use_fp16", False)
    autocast_dtype = torch.float16 if use_fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        student.load_state_dict(ckpt["student_state_dict"])
        teacher.load_state_dict(ckpt["teacher_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "dino_loss_state_dict" in ckpt:
            dino_loss_fn.load_state_dict(ckpt["dino_loss_state_dict"])
        if ibot_loss_fn and "ibot_loss_state_dict" in ckpt:
            ibot_loss_fn.load_state_dict(ckpt["ibot_loss_state_dict"])
        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"  Resumed at epoch {start_epoch}")

    # ---- Logging ----
    init_experiment(config)
    run_name = get_run_name()

    checkpoints_path = config.get("checkpoints_dir", "checkpoints")
    os.makedirs(checkpoints_path, exist_ok=True)

    grad_accum_steps = config.get("gradient_accumulation_steps", 1)
    clip_grad = config.get("clip_grad", 3.0)
    ibot_weight = config.get("ibot_loss_weight", 1.0)
    koleo_weight = config.get("koleo_loss_weight", 0.1)
    ibot_mask_ratio = config.get("ibot_mask_ratio", 0.3)

    print(f"KRONOS {training_mode.upper()} (VirTues preprocessing) | "
          f"{model_name} | embed={embed_dim} | patch={patch_size} | "
          f"markers={num_markers} | registers={num_register_tokens} | ffn={ffn_layer}")
    print(f"Crops: {n_global} global + {n_local} local | "
          f"Batch: {config['batch_size']} x {grad_accum_steps} accum")
    print(f"Training: epochs {start_epoch}..{config['epochs']-1} | "
          f"{niter_per_ep} iters/epoch")

    # -------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------
    for epoch in range(start_epoch, config["epochs"]):
        student.train()
        teacher.eval()
        running_loss = 0.0
        running_dino = 0.0
        running_ibot = 0.0
        running_koleo = 0.0

        for batch_idx, batch_data in enumerate(
            tqdm(train_dataloader, desc=f"Epoch {epoch}")
        ):
            # Unpack batch — ImmuVis returns 4-tuples, VirTues returns 2-tuples
            if immuvis_mode:
                crops, channel_ids, _, _ = batch_data
            else:
                crops, channel_ids = batch_data

            global_step = niter_per_ep * epoch + batch_idx
            sched_idx = min(global_step, len(lr_schedule) - 1)

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule[sched_idx]
                param_group["weight_decay"] = wd_schedule[sched_idx]

            crops = [c.to(device, dtype=torch.float32, non_blocking=True) for c in crops]
            channel_ids = channel_ids.to(device, dtype=torch.long, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=autocast_dtype):

                if training_mode == "dinov2":
                    student_output = student.forward_dino_multicrop(crops, channel_ids)
                    with torch.no_grad():
                        teacher_output = teacher.forward_dino_multicrop(
                            crops[:n_global], channel_ids
                        )
                    loss_dino = dino_loss_fn(student_output, teacher_output, epoch)
                    total_loss = loss_dino

                else:
                    # DINOv3: CLS + iBOT + KoLeo
                    B = crops[0].shape[0]
                    C_markers = channel_ids.shape[1]
                    # Use backbone's actual patch count per marker (accounts for stride)
                    n_spatial = student.backbone.patch_embed.num_patches

                    ibot_masks_per_crop = [
                        generate_spatial_ibot_mask(
                            B, C_markers, n_spatial, ibot_mask_ratio, device
                        )
                        for _ in range(n_global)
                    ]

                    with torch.no_grad():
                        teacher_cls_all = []
                        teacher_patch_masked_all = []
                        for gi in range(n_global):
                            t_cls, t_patches = teacher.forward_features_single(
                                crops[gi], channel_ids, masks=None
                            )
                            teacher_cls_all.append(t_cls)
                            teacher_patch_masked_all.append(
                                teacher.ibot_head(t_patches[ibot_masks_per_crop[gi]])
                            )
                        teacher_dino_out = teacher.dino_head(
                            torch.cat(teacher_cls_all, dim=0)
                        )
                        teacher_ibot_out = torch.cat(teacher_patch_masked_all, dim=0)

                    student_cls_global = []
                    student_ibot_all = []
                    student_cls_features_for_koleo = []

                    for gi in range(n_global):
                        s_cls, s_patches = student.forward_features_single(
                            crops[gi], channel_ids, masks=ibot_masks_per_crop[gi]
                        )
                        student_cls_global.append(s_cls)
                        student_cls_features_for_koleo.append(s_cls)
                        student_ibot_all.append(
                            student.ibot_head(s_patches[ibot_masks_per_crop[gi]])
                        )

                    student_cls_local = []
                    for li in range(n_global, ncrops):
                        s_cls, _ = student.forward_features_single(
                            crops[li], channel_ids, masks=None
                        )
                        student_cls_local.append(s_cls)

                    student_dino_out = student.dino_head(
                        torch.cat(student_cls_global + student_cls_local, dim=0)
                    )
                    student_ibot_out = torch.cat(student_ibot_all, dim=0)

                    loss_dino = dino_loss_fn(student_dino_out, teacher_dino_out, epoch)
                    loss_ibot = ibot_loss_fn(student_ibot_out, teacher_ibot_out, epoch)
                    loss_koleo = koleo_loss_fn(
                        torch.cat(student_cls_features_for_koleo, dim=0)
                    )

                    total_loss = (loss_dino
                                  + ibot_weight * loss_ibot
                                  + koleo_weight * loss_koleo)

                    running_dino += loss_dino.item()
                    running_ibot += loss_ibot.item()
                    running_koleo += loss_koleo.item()

            accumulated_loss = total_loss / grad_accum_steps
            scaler.scale(accumulated_loss).backward()

            if ((batch_idx + 1) % grad_accum_steps == 0) or (
                (batch_idx + 1) == len(train_dataloader)
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # EMA teacher update
            m = momentum_schedule[sched_idx]
            with torch.no_grad():
                for param_s, param_t in zip(
                    student.parameters(), teacher.parameters()
                ):
                    param_t.data.mul_(m).add_((1 - m) * param_s.detach().data)

            running_loss += total_loss.item()

            if (batch_idx + 1) % 10 == 0:
                exp = comet_ml.get_global_experiment()
                if exp is not None:
                    log_dict = {
                        "train/total_loss": total_loss.item(),
                        "train/lr": lr_schedule[sched_idx],
                        "train/teacher_momentum": m,
                    }
                    if training_mode == "dinov2":
                        log_dict["train/dino_loss"] = loss_dino.item()
                    else:
                        log_dict["train/dino_loss"] = loss_dino.item()
                        log_dict["train/ibot_loss"] = loss_ibot.item()
                        log_dict["train/koleo_loss"] = loss_koleo.item()
                    exp.log_metrics(log_dict, step=global_step)

        # End of epoch
        epoch_loss = running_loss / max(len(train_dataloader), 1)
        if training_mode == "dinov2":
            print(f"Epoch {epoch} | DINO Loss: {epoch_loss:.4f} | "
                  f"LR: {lr_schedule[min(niter_per_ep * (epoch + 1) - 1, len(lr_schedule) - 1)]:.2e}")
        else:
            ep_dino = running_dino / max(len(train_dataloader), 1)
            ep_ibot = running_ibot / max(len(train_dataloader), 1)
            ep_koleo = running_koleo / max(len(train_dataloader), 1)
            print(f"Epoch {epoch} | Total: {epoch_loss:.4f} "
                  f"(DINO: {ep_dino:.4f}, iBOT: {ep_ibot:.4f}, KoLeo: {ep_koleo:.4f})")

        exp = comet_ml.get_global_experiment()
        if exp is not None:
            log_epoch = {"epoch/total_loss": epoch_loss}
            if training_mode == "dinov3":
                log_epoch["epoch/dino_loss"] = running_dino / max(len(train_dataloader), 1)
                log_epoch["epoch/ibot_loss"] = running_ibot / max(len(train_dataloader), 1)
                log_epoch["epoch/koleo_loss"] = running_koleo / max(len(train_dataloader), 1)
            exp.log_metrics(log_epoch, epoch=epoch)

        # Save checkpoint
        if (epoch + 1) % config.get("save_checkpoint_freq", 5) == 0:
            save_dict = {
                "student_state_dict": student.state_dict(),
                "teacher_state_dict": teacher.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "dino_loss_state_dict": dino_loss_fn.state_dict(),
                "epoch": epoch,
                "config": config,
            }
            if ibot_loss_fn is not None:
                save_dict["ibot_loss_state_dict"] = ibot_loss_fn.state_dict()
            torch.save(
                save_dict,
                os.path.join(
                    checkpoints_path,
                    f"kronos_{training_mode}_virtues-{run_name}-epoch_{epoch}.pth",
                ),
            )

    # Final save
    final_dict = {
        "student_state_dict": student.state_dict(),
        "teacher_state_dict": teacher.state_dict(),
        "epoch": config["epochs"] - 1,
        "config": config,
    }
    torch.save(
        final_dict,
        os.path.join(
            checkpoints_path,
            f"kronos_{training_mode}_virtues-{run_name}-final.pth",
        ),
    )

    print("Training finished!")
    finish_experiment()


if __name__ == "__main__":
    main()
