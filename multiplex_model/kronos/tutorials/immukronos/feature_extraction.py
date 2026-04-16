"""
Patch Extraction and Feature Extraction utilities for ImmuKronos (DINOv3).

Supports two data formats:
    1. HDF5 patches (original KRONOS tutorial format) — PatchDataset
    2. ImmuVis .npy format (multi-panel, variable markers) — ImmuVisDataset

For ImmuVis data, use ImmuVisFeatureExtractor which handles:
    - Panel-aware batching (PanelBatchSampler)
    - Variable marker counts across panels
    - Fixed-size marker features (padded to full tokenizer size)
"""

import os
import sys
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import skimage.io as skio
import scanpy as sc
from glob import glob

from .inference import load_immukronos_model, ImmuKronosWrapper


class PatchDataset(Dataset):
    """
    Loads multiplex patches (HDF5 format) for feature extraction.

    Each HDF5 file contains per-marker spatial data.

    Attributes:
        patch_dir: Directory with .h5 patch files.
        marker_list: Ordered list of marker names.
        marker_metadata: DataFrame with marker_id, marker_mean, marker_std.
        marker_max_values: Max intensity for initial scaling.
    """

    def __init__(self, config):
        self.config = config
        self.patch_dir = config["patch_dir"]
        self.patch_list = sorted([f for f in os.listdir(self.patch_dir) if f.endswith(".h5")])
        self.marker_list = config["marker_list"]
        self.marker_max_values = config["marker_max_values"]
        self.marker_metadata = pd.read_csv(config["marker_info_with_metadata_csv_path"])
        self.marker_metadata.set_index("marker_name", inplace=True)

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        patch_name = self.patch_list[idx]
        patch_path = os.path.join(self.patch_dir, patch_name)

        with h5py.File(patch_path, "r") as f:
            patch_markers = []
            marker_ids = []

            for marker_name in self.marker_list:
                marker_id = self.marker_metadata.loc[marker_name, "marker_id"]
                marker_mean = self.marker_metadata.loc[marker_name, "marker_mean"]
                marker_std = self.marker_metadata.loc[marker_name, "marker_std"]

                marker = f[marker_name][:] / self.marker_max_values
                marker = (marker - marker_mean) / marker_std

                patch_markers.append(torch.tensor(marker, dtype=torch.float32))
                marker_ids.append(int(marker_id))

            patch_markers = torch.stack(patch_markers, dim=0)
            marker_ids = torch.tensor(marker_ids, dtype=torch.long)

        return patch_markers, marker_ids, patch_name


class CellPatchDataset(PatchDataset):
    """PatchDataset that also loads a cell mask from each HDF5 file."""

    def __getitem__(self, idx):
        patch_markers, marker_ids, patch_name = super().__getitem__(idx)

        patch_path = os.path.join(self.patch_dir, self.patch_list[idx])
        with h5py.File(patch_path, "r") as f:
            cell_mask = np.uint8(f["mask"][:]) if "mask" in f else np.ones(
                patch_markers.shape[1:], dtype=np.uint8
            )

        return patch_markers, marker_ids, cell_mask, patch_name


class FeatureExtractor:
    """
    Extracts features from patches using a trained ImmuKronos-DINOv3 model.

    Produces two types of outputs:
        - CLS features:    (embed_dim,) per patch → saved to cls_features/
        - Marker features: (num_markers * embed_dim,) per patch → saved to marker_features/

    Config keys:
        checkpoint_path: Path to trained model checkpoint.
        patch_dir: Directory with .h5 patch files.
        output_dir: Directory to save extracted features.
        marker_list: List of marker names.
        marker_max_values: Max intensity value.
        marker_info_with_metadata_csv_path: Path to marker metadata CSV.
        batch_size: Batch size for feature extraction.
        device: 'cuda' or 'cpu'.
        model_config: dict with embed_dim, depth, num_heads, patch_size, etc.
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get("device", "cuda"))

        model_cfg = config.get("model_config", {})
        model, self.embed_dim = load_immukronos_model(
            checkpoint_path=config["checkpoint_path"],
            num_markers=config.get("num_markers", 512),
            embed_dim=model_cfg.get("embed_dim", 384),
            depth=model_cfg.get("depth", 12),
            num_heads=model_cfg.get("num_heads", 6),
            patch_size=model_cfg.get("patch_size", 8),
            out_dim=model_cfg.get("out_dim", 65536),
            num_register_tokens=model_cfg.get("num_register_tokens", 4),
            ibot_out_dim=model_cfg.get("ibot_out_dim", 8192),
            device=str(self.device),
        )
        self.model = ImmuKronosWrapper(model, model_cfg.get("patch_size", 8))
        self.model.eval()

    def extract_features(self, dataset=None, save_spatial=False):
        """
        Extract and save features for all patches.

        Args:
            dataset: Optional Dataset instance. If None, creates PatchDataset from config.
            save_spatial: If True, also save full spatial token features.
        """
        if dataset is None:
            dataset = PatchDataset(self.config)

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get("batch_size", 16),
            shuffle=False,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True,
        )

        output_dir = self.config["output_dir"]
        cls_dir = os.path.join(output_dir, "cls_features")
        marker_dir = os.path.join(output_dir, "marker_features")
        os.makedirs(cls_dir, exist_ok=True)
        os.makedirs(marker_dir, exist_ok=True)
        if save_spatial:
            spatial_dir = os.path.join(output_dir, "spatial_features")
            os.makedirs(spatial_dir, exist_ok=True)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                # Handle both PatchDataset (3-tuple) and CellPatchDataset (4-tuple)
                if len(batch) == 4:
                    patches, marker_ids, cell_masks, patch_names = batch
                    # Apply cell mask
                    mask_tensor = torch.tensor(cell_masks, dtype=torch.float32).unsqueeze(1)
                    patches = patches * mask_tensor
                elif len(batch) == 3:
                    patches, marker_ids, patch_names = batch
                else:
                    raise ValueError(f"Unexpected batch length: {len(batch)}")

                patches = patches.to(self.device, dtype=torch.float32)
                marker_ids = marker_ids.to(self.device)

                patch_features, marker_features, spatial_features = self.model(
                    patches, marker_ids=marker_ids
                )

                patch_features_np = patch_features.cpu().numpy()
                marker_features_np = marker_features.cpu().numpy()

                for j, name in enumerate(patch_names):
                    stem = name.replace(".h5", "")

                    # CLS feature: (embed_dim,)
                    np.save(
                        os.path.join(cls_dir, f"{stem}.npy"),
                        patch_features_np[j],
                    )

                    # Marker features: (num_markers, embed_dim) -> flattened
                    np.save(
                        os.path.join(marker_dir, f"{stem}.npy"),
                        marker_features_np[j].flatten(),
                    )

                    if save_spatial:
                        spatial_np = spatial_features[j].cpu().numpy()
                        np.save(
                            os.path.join(spatial_dir, f"{stem}.npy"),
                            spatial_np,
                        )

        print(f"Features saved to {output_dir}")


class ImmuVisFeatureExtractor:
    """
    Feature extractor for ImmuVis multi-panel .npy data.

    Handles the key differences from the HDF5-based FeatureExtractor:
        - Uses DatasetFromTIFF + PanelBatchSampler for panel-aware batching
        - Produces fixed-size marker features by padding to full tokenizer size
        - Each marker's feature is placed at its tokenizer index position

    CLS features:    (embed_dim,)                  — always fixed size
    Marker features: (num_total_markers * embed_dim,) — padded, fixed size

    Config keys:
        checkpoint_path: Path to trained model checkpoint.
        panel_config_path: Path to all_panels_config.yaml.
        tokenizer_config_path: Path to all_markers_tokenizer.yaml.
        output_dir: Directory to save extracted features.
        split: Data split to extract from ('train' or 'test').
        batch_size: Batch size for feature extraction (default: 16).
        device: 'cuda' or 'cpu'.
        model_config: Dict with embed_dim, depth, num_heads, patch_size, etc.
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get("device", "cuda"))

        # Load tokenizer
        from ruamel.yaml import YAML
        yaml = YAML(typ="safe")
        with open(config["tokenizer_config_path"], "r") as f:
            self.tokenizer = yaml.load(f)
        self.num_total_markers = len(self.tokenizer)

        # Load model
        model_cfg = config.get("model_config", {})
        model, self.embed_dim = load_immukronos_model(
            checkpoint_path=config["checkpoint_path"],
            num_markers=self.num_total_markers,
            embed_dim=model_cfg.get("embed_dim", 384),
            depth=model_cfg.get("depth", 12),
            num_heads=model_cfg.get("num_heads", 6),
            patch_size=model_cfg.get("patch_size", 8),
            out_dim=model_cfg.get("out_dim", 65536),
            num_register_tokens=model_cfg.get("num_register_tokens", 4),
            ibot_out_dim=model_cfg.get("ibot_out_dim", 8192),
            device=str(self.device),
        )
        self.wrapper = ImmuKronosWrapper(model, model_cfg.get("patch_size", 8))
        self.wrapper.eval()

    def extract_features(self, save_spatial=False):
        """
        Extract features from ImmuVis .npy data with panel-aware batching.

        Marker features are padded to (num_total_markers, embed_dim) so that
        features from different panels have identical dimensions. Each marker's
        feature is placed at its tokenizer index; unused slots remain zero.
        """
        # Import data pipeline from training codebase
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler
        from ruamel.yaml import YAML

        yaml = YAML(typ="safe")
        with open(self.config["panel_config_path"], "r") as f:
            panel_config = yaml.load(f)

        split = self.config.get("split", "test")
        dataset = DatasetFromTIFF(
            panels_config=panel_config,
            split=split,
            marker_tokenizer=self.tokenizer,
            transform=None,
            use_preprocessing=False,
            use_butterworth_filter=True,
            use_clip_normalization=True,
            file_extension="npy",
        )

        sampler = PanelBatchSampler(
            dataset, self.config.get("batch_size", 16), shuffle=False
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True,
        )

        output_dir = self.config["output_dir"]
        cls_dir = os.path.join(output_dir, "cls_features")
        marker_dir = os.path.join(output_dir, "marker_features")
        os.makedirs(cls_dir, exist_ok=True)
        os.makedirs(marker_dir, exist_ok=True)
        if save_spatial:
            spatial_dir = os.path.join(output_dir, "spatial_features")
            os.makedirs(spatial_dir, exist_ok=True)

        D = self.embed_dim
        N_total = self.num_total_markers

        with torch.no_grad():
            for img, channel_ids, dataset_name, img_path in tqdm(
                dataloader, desc=f"Extracting features ({split})"
            ):
                img = img.to(self.device, dtype=torch.float32)
                channel_ids = channel_ids.to(self.device)

                patch_features, marker_features, spatial_features = self.wrapper(
                    img, marker_ids=channel_ids
                )

                patch_features_np = patch_features.cpu().numpy()
                marker_features_np = marker_features.cpu().numpy()  # (B, M_panel, D)
                channel_ids_np = channel_ids.cpu().numpy()  # (B, M_panel)

                B = patch_features_np.shape[0]
                for j in range(B):
                    stem = os.path.splitext(os.path.basename(img_path[j]))[0]

                    # CLS features: (D,) — fixed size
                    np.save(os.path.join(cls_dir, f"{stem}.npy"), patch_features_np[j])

                    # Marker features: pad to (N_total, D) using tokenizer indices
                    padded = np.zeros((N_total, D), dtype=np.float32)
                    for k, tok_id in enumerate(channel_ids_np[j]):
                        padded[tok_id] = marker_features_np[j, k]
                    np.save(os.path.join(marker_dir, f"{stem}.npy"), padded.flatten())

                    if save_spatial:
                        spatial_np = spatial_features[j].cpu().numpy()
                        np.save(os.path.join(spatial_dir, f"{stem}.npy"), spatial_np)

        print(f"Features saved to {output_dir}")
        print(f"  CLS features: ({D},)")
        print(f"  Marker features: ({N_total * D},) — padded to full tokenizer size")


class H5ADBuilder:
    """
    Builds AnnData (h5ad) objects from extracted numpy features.

    Loads feature files, extracts metadata from filenames, optionally merges
    with sample-level metadata, and saves as h5ad for downstream analysis
    (e.g., clustering, UMAP, patient stratification).

    Config keys:
        feature_dir: Directory containing .npy feature files.
        output_path: Path to save the .h5ad file.
        metadata_csv_path: (optional) Path to sample-level metadata CSV.
        metadata_id_col: (optional) Column in metadata CSV to join on.
    """

    def __init__(self, config):
        self.config = config

    def build(self, feature_type="cls_features"):
        """
        Build h5ad from a specific feature type directory.

        Args:
            feature_type: Subdirectory name ('cls_features' or 'marker_features').

        Returns:
            adata: AnnData object.
        """
        feature_dir = os.path.join(self.config["feature_dir"], feature_type)
        npy_files = sorted(glob(os.path.join(feature_dir, "*.npy")))

        if len(npy_files) == 0:
            raise FileNotFoundError(f"No .npy files found in {feature_dir}")

        features = []
        obs_names = []
        for f in tqdm(npy_files, desc=f"Loading {feature_type}"):
            features.append(np.load(f))
            obs_names.append(os.path.basename(f).replace(".npy", ""))

        X = np.stack(features)
        adata = sc.AnnData(X=X)
        adata.obs_names = obs_names

        # Parse metadata from filenames (image_x_y pattern)
        image_ids = []
        for name in obs_names:
            parts = name.split("_")
            image_ids.append(parts[0] if parts else name)
        adata.obs["image_id"] = image_ids

        # Merge with external metadata if provided
        metadata_path = self.config.get("metadata_csv_path")
        if metadata_path and os.path.exists(metadata_path):
            meta_df = pd.read_csv(metadata_path)
            id_col = self.config.get("metadata_id_col", "image_id")
            if id_col in meta_df.columns:
                meta_df = meta_df.set_index(id_col)
                for col in meta_df.columns:
                    adata.obs[col] = adata.obs["image_id"].map(
                        meta_df[col].to_dict()
                    )

        output_path = self.config["output_path"]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        adata.write_h5ad(output_path)
        print(f"Saved h5ad to {output_path} ({adata.shape})")

        return adata


class PatchExtraction:
    """
    Extracts patches from whole multiplex images using a sliding window.

    Saves each patch as an HDF5 file with per-marker datasets.

    Config keys:
        image_dir: Directory containing multiplex images.
        output_dir: Directory to save extracted patches.
        marker_list: List of dicts with 'channel_id' and 'marker_name'.
        patch_size: Patch size in pixels (default: 256).
        stride: Stride for sliding window (default: 256).
        file_ext: Image file extension (default: '.ome.tiff').
    """

    def __init__(self, config):
        self.config = config
        self.image_dir = config["image_dir"]
        self.output_dir = config["output_dir"]
        self.marker_list = config["marker_list"]
        self.patch_size = config.get("patch_size", 256)
        self.stride = config.get("stride", 256)
        self.file_ext = config.get("file_ext", ".ome.tiff")

    def extract(self):
        """Extract patches from all images in image_dir."""
        os.makedirs(self.output_dir, exist_ok=True)
        image_files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.endswith(self.file_ext)
        ])

        for img_file in tqdm(image_files, desc="Extracting patches"):
            img_path = os.path.join(self.image_dir, img_file)
            image = skio.imread(img_path)  # (C, H, W) or (H, W, C)

            if image.ndim == 3 and image.shape[0] < image.shape[2]:
                pass  # already (C, H, W)
            elif image.ndim == 3:
                image = image.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)

            C, H, W = image.shape
            img_stem = os.path.splitext(img_file)[0].replace(".ome", "")

            for y in range(0, H - self.patch_size + 1, self.stride):
                for x in range(0, W - self.patch_size + 1, self.stride):
                    patch_name = f"{img_stem}_{y}_{x}.h5"
                    patch_path = os.path.join(self.output_dir, patch_name)

                    with h5py.File(patch_path, "w") as f:
                        for marker_info in self.marker_list:
                            ch_idx = marker_info["channel_id"]
                            m_name = marker_info["marker_name"]
                            patch = image[ch_idx, y:y + self.patch_size, x:x + self.patch_size]
                            f.create_dataset(m_name, data=patch.astype(np.float32))

        print(f"Patches saved to {self.output_dir}")
