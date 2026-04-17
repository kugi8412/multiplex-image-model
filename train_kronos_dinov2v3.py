#!/usr/bin/env python
# -*- coding: utf-8 -*-
# train_kronos_dinov2v3.py


"""
Train KRONOS ViT backbone with DINOv2 or DINOv3 self-distillation objectives.

Uses KRONOS native per-channel patch embedding + sinusoidal marker embeddings.
No Hyperkernel — pure KRONOS ViT as in the original KRONOS paper,
similar to how VIRTUES trains a foundation model on multiplex imaging data.

DINOv2 mode (training_mode: "dinov2"):
  - CLS-token self-distillation only (classic DINOv2 objective)

DINOv3 mode (training_mode: "dinov3"):
  - CLS-token self-distillation (DINO loss)
  - Patch-level self-distillation with spatial masking (iBOT loss)
  - KoLeo diversity regularizer on CLS features

Usage::
    python train_kronos_dinov2v3.py configs/train_kronos_dinov2_config.yaml
    python train_kronos_dinov2v3.py configs/train_kronos_dinov3_config.yaml
    python train_kronos_dinov2v3.py configs/train_kronos_dinov3_config.yaml --from-checkpoint checkpoints/model.pth
"""

import argparse
import csv
import os

import comet_ml  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomVerticalFlip,
)
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler
from multiplex_model.utils import init_experiment, finish_experiment, get_run_name
from multiplex_model.kronos.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from multiplex_model.kronos.dino_head import DINOHead


# -------------------------------------------
# 0. MARKER EMBEDDING ID RESOLUTION
# -------------------------------------------

def load_marker_metadata(metadata_csv_path):
    """Load marker_metadata.csv and return {marker_name: marker_id} mapping.

    The CSV is expected to have columns: marker_name, marker_id, marker_mean, marker_std.
    """
    marker_id_map = {}
    with open(metadata_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["marker_name"].strip().strip('"')
            mid = int(row["marker_id"])
            marker_id_map[name] = mid
    return marker_id_map


def build_custom_marker_ids_from_tokenizer(tokenizer):
    """Create sequential integer marker IDs from the tokenizer in sorted order.

    Returns {marker_name: id} mapping with IDs assigned in alphabetical order
    of marker names starting from 0.
    """
    sorted_names = sorted(tokenizer.keys(), key=str.lower)
    return {name: idx for idx, name in enumerate(sorted_names)}


def resolve_marker_embeddings(config, tokenizer):
    """Resolve marker embedding IDs for the KRONOS backbone.

    Strategy:
      1. If marker_metadata.csv (path from config or default) exists on disk,
         load it and use its ``marker_id`` values.  The backbone's
         ``num_markers`` will be ``max(ids) + 1`` so that the sinusoidal
         look-up table covers all IDs.
      2. Otherwise, build fresh sequential IDs from the tokenizer file,
         ordered alphabetically, and persist them to a CSV so subsequent
         runs are deterministic.

    Returns:
        marker_id_map (dict[str, int]): marker_name -> integer ID.
        num_markers (int): total number of marker slots for the sinusoidal
            embedding table (>= max(id)+1).
    """
    metadata_path = config.get("marker_metadata_csv", "configs/marker_metadata.csv")

    if os.path.isfile(metadata_path):
        print(f"[Marker IDs] Loading from existing metadata: {metadata_path}")
        marker_id_map = load_marker_metadata(metadata_path)
    else:
        print(f"[Marker IDs] {metadata_path} not found — creating custom IDs "
              f"from tokenizer ({len(tokenizer)} markers, alphabetical order).")
        marker_id_map = build_custom_marker_ids_from_tokenizer(tokenizer)

        # Persist so future runs are reproducible
        out_path = metadata_path
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["marker_name", "marker_id", "marker_mean", "marker_std"])
            for name in sorted(marker_id_map.keys(), key=str.lower):
                writer.writerow([name, marker_id_map[name], 0.0, 1.0])
        print(f"[Marker IDs] Saved new metadata to {out_path}")

    num_markers = max(marker_id_map.values()) + 1
    print(f"[Marker IDs] {len(marker_id_map)} markers, "
          f"sinusoidal table size = {num_markers}")
    return marker_id_map, num_markers


def build_tokenizer_from_marker_ids(tokenizer, marker_id_map):
    """Re-map the tokenizer values to the resolved marker IDs.

    Returns a new dict {marker_name: resolved_id} for every marker present
    in both the tokenizer and the marker_id_map.  Markers in the tokenizer
    that are NOT in the map are assigned new sequential IDs starting from
    ``max(marker_id_map.values()) + 1`` so that indices stay within the
    sinusoidal embedding table.

    Returns:
        new_tokenizer (dict): {marker_name: resolved_id}
        num_markers (int): updated total marker slots (accounts for any
            newly assigned IDs).
    """
    new_tokenizer = {}
    missing = []
    next_free_id = max(marker_id_map.values()) + 1 if marker_id_map else 0
    for name, old_id in tokenizer.items():
        if name in marker_id_map:
            new_tokenizer[name] = marker_id_map[name]
        else:
            # Try case-insensitive lookup
            found = False
            for map_name, map_id in marker_id_map.items():
                if map_name.lower() == name.lower():
                    new_tokenizer[name] = map_id
                    found = True
                    break
            if not found:
                new_tokenizer[name] = next_free_id
                next_free_id += 1
                missing.append(name)
    if missing:
        print(f"[Marker IDs] WARNING: {len(missing)} tokenizer markers not found "
              f"in metadata, assigned new IDs: {missing[:10]}{'...' if len(missing)>10 else ''}")
    updated_num_markers = next_free_id
    return new_tokenizer, updated_num_markers


# -------------------------------------------
# 1. MODEL WRAPPER
# -------------------------------------------

class KronosDINOv2v3(nn.Module):
    """KRONOS ViT backbone with DINO + optional iBOT projection heads.

    The KRONOS backbone (per-channel patch embedding, sinusoidal marker embeddings)
    is wrapped with DINOHead(s) for self-distillation training.

    In DINOv2 mode, only the DINO head (CLS token) is used.
    In DINOv3 mode, an additional iBOT head (patch tokens) is used.
    """

    def __init__(self, backbone, embed_dim, dino_out_dim, ibot_out_dim=None):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.dino_head = DINOHead(
            in_dim=embed_dim, out_dim=dino_out_dim,
            hidden_dim=2048, bottleneck_dim=256, nlayers=3,
        )
        self.ibot_head = (
            DINOHead(
                in_dim=embed_dim, out_dim=ibot_out_dim,
                hidden_dim=2048, bottleneck_dim=256, nlayers=3,
            ) if ibot_out_dim is not None else None
        )

    def forward_features_single(self, x, marker_ids, masks=None):
        """Run backbone on a single crop. Returns CLS and patch token features.

        Args:
            x: (B, C_markers, H, W) multiplex image crop.
            marker_ids: (B, C_markers) marker token IDs.
            masks: (B, C_markers * N_patches) boolean iBOT mask or None.
        Returns:
            cls_token: (B, embed_dim)
            patch_tokens: (B, C_markers * N_patches, embed_dim)
        """
        ids_list = [marker_ids]
        feat = self.backbone.forward_features(x, masks=masks, marker_ids=ids_list)
        return feat["x_norm_clstoken"], feat["x_norm_patchtokens"]

    def forward_dino_multicrop(self, crops, marker_ids):
        """DINOv2 mode: CLS token projection for multiple crops (no masking).

        Processes each crop independently to avoid xFormers dependency.
        """
        all_cls = []
        for crop in crops:
            cls, _ = self.forward_features_single(crop, marker_ids)
            all_cls.append(cls)
        cls_cat = torch.cat(all_cls, dim=0)
        return self.dino_head(cls_cat)


# -------------------------------------------
# 2. LOSSES
# -------------------------------------------

class DINOLoss(nn.Module):
    """Cross-entropy self-distillation loss between student and teacher CLS distributions."""

    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, nglobal_crops=2):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.nglobal_crops = nglobal_crops
        self.register_buffer("center", torch.zeros(1, out_dim))

        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(max(nepochs - warmup_teacher_temp_epochs, 0)) * teacher_temp,
        ))

    def forward(self, student_output, teacher_output, epoch, update_center=True):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        temp = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.nglobal_crops)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms

        if update_center:
            self.update_center(teacher_output)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class iBOTLoss(nn.Module):
    """Patch-level self-distillation loss (iBOT) for DINOv3 training."""

    def __init__(self, out_dim, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs,
                 student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(max(nepochs - warmup_teacher_temp_epochs, 0)) * teacher_temp,
        ))

    def forward(self, student_patch_logits, teacher_patch_logits, epoch, update_center=True):
        if student_patch_logits.shape[0] == 0:
            return student_patch_logits.sum() * 0

        temp = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)]
        student_out = student_patch_logits / self.student_temp
        teacher_out = F.softmax((teacher_patch_logits - self.center) / temp, dim=-1).detach()
        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)

        if update_center:
            self._update_center(teacher_patch_logits)

        return loss.mean()

    @torch.no_grad()
    def _update_center(self, teacher_output):
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class KoLeoLoss(nn.Module):
    """KoLeo diversity regularizer — encourages uniform CLS feature spacing."""

    def forward(self, features):
        features = F.normalize(features, dim=-1, p=2)
        dists = torch.cdist(features, features)
        diag_mask = torch.eye(dists.shape[0], dtype=torch.bool, device=dists.device)
        dists = dists.masked_fill(diag_mask, float("inf"))
        min_dists = dists.min(dim=-1).values
        return -torch.log(min_dists + 1e-8).mean()


# -------------------------------------------
# 3. iBOT SPATIAL MASKING FOR KRONOS
# -------------------------------------------

def generate_spatial_ibot_mask(batch_size, num_markers, num_spatial_patches,
                                mask_ratio, device):
    """Generate iBOT mask that masks spatial positions uniformly across all markers.

    In the KRONOS ViT, tokens are ordered as:
      [marker0_patch0, ..., marker0_patchN, marker1_patch0, ..., markerC_patchN]

    Masking spatial position p means masking tokens at
    positions p, p+N, p+2N, ..., p+(C-1)*N across all markers.

    Args:
        batch_size: Batch size B.
        num_markers: Number of markers C.
        num_spatial_patches: Spatial patches N per marker.
        mask_ratio: Fraction of spatial positions to mask.
        device: Torch device.
    Returns:
        Boolean tensor (B, C*N) with True at masked positions.
    """
    num_masked = max(1, int(num_spatial_patches * mask_ratio))
    total_tokens = num_markers * num_spatial_patches
    mask = torch.zeros(batch_size, total_tokens, dtype=torch.bool, device=device)

    for b in range(batch_size):
        spatial_indices = torch.randperm(num_spatial_patches, device=device)[:num_masked]
        for m in range(num_markers):
            mask[b, m * num_spatial_patches + spatial_indices] = True

    return mask


# -------------------------------------------
# 4. MULTI-CROP AUGMENTATION & DATASET
# -------------------------------------------

class SpatialProteomicsMultiCrop:
    """Multi-crop augmentation for spatial proteomics data.
    No pixel-level color jitter — only spatial augmentations suitable for
    multiplexed fluorescence/mass-spec images."""

    def __init__(self, global_scale, local_scale, global_size, local_size,
                 local_crops_num, global_crops_num=2):
        self.global_crops_num = global_crops_num
        self.local_crops_num = local_crops_num
        self.global_transform = Compose([
            RandomResizedCrop(global_size, scale=tuple(global_scale),
                              interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
        ])
        self.local_transform = Compose([
            RandomResizedCrop(local_size, scale=tuple(local_scale),
                              interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
        ])

    def __call__(self, img):
        crops = []
        for _ in range(self.global_crops_num):
            crops.append(self.global_transform(img))
        for _ in range(self.local_crops_num):
            crops.append(self.local_transform(img))
        return crops


class DINODatasetWrapper(torch.utils.data.Dataset):
    """Wraps DatasetFromTIFF to apply Multi-Crop augmentation."""

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
    """Groups crop lists into a proper batch format."""
    num_crops = len(batch[0][0])
    collated_crops = [torch.stack([item[0][i] for item in batch]) for i in range(num_crops)]
    channel_ids = torch.stack([item[1] for item in batch])
    dataset_names = [item[2] for item in batch]
    img_paths = [item[3] for item in batch]
    return collated_crops, channel_ids, dataset_names, img_paths


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0):
    warmup_iters = warmup_epochs * niter_per_ep
    warmup_schedule = np.array([])
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(0, base_value, warmup_iters)
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    return schedule


# -------------------------------------------
# 5. BACKBONE FACTORY
# -------------------------------------------

MODEL_CONFIGS = {
    "vits16":    {"factory": vit_small,  "embed_dim": 384},
    "vit_small": {"factory": vit_small,  "embed_dim": 384},
    "vitb16":    {"factory": vit_base,   "embed_dim": 768},
    "vit_base":  {"factory": vit_base,   "embed_dim": 768},
    "vitl16":    {"factory": vit_large,  "embed_dim": 1024},
    "vit_large": {"factory": vit_large,  "embed_dim": 1024},
    "vitg14":    {"factory": vit_giant2, "embed_dim": 1536},
    "vit_giant2":{"factory": vit_giant2, "embed_dim": 1536},
}


def build_backbone(model_name, num_markers, img_size, patch_size, drop_path_rate,
                   num_register_tokens=0, ffn_layer="mlp", init_values=None):
    """Build KRONOS ViT backbone with given configuration."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_CONFIGS.keys())}")

    cfg = MODEL_CONFIGS[model_name]
    backbone = cfg["factory"](
        patch_size=patch_size,
        stride_size=patch_size,  # non-overlapping patches (standard ViT)
        num_markers=num_markers,
        img_size=img_size,
        drop_path_rate=drop_path_rate,
        num_register_tokens=num_register_tokens,
        ffn_layer=ffn_layer,
        init_values=init_values,
    )
    return backbone, cfg["embed_dim"]


# -------------------------------------------
# 6. MAIN TRAINING LOOP
# -------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="KRONOS DINOv2/v3 training on ImmuVis")
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
                f"but you ran train_kronos_dinov2v3.py which needs DINO multi-crop config.\n"
                f"Use 'python train_masked_model.py {args.config}' instead.\n"
                f"Missing keys: {missing}"
            )
        raise SystemExit(
            f"ERROR: Config '{args.config}' is missing required keys: {missing}\n"
            f"See configs/train_kronos_dinov2_config.yaml for an example."
        )

    device = torch.device(args.device or config.get("device", "cuda"))
    training_mode = config.get("training_mode", "dinov2")
    assert training_mode in ("dinov2", "dinov3"), (
        f"training_mode must be 'dinov2' or 'dinov3', got '{training_mode}'"
    )
    print(f"Training mode: {training_mode.upper()} | Device: {device}")

    # ---- Marker embedding resolution ----
    PANEL_CONFIG = YAML().load(open(config["panel_config"]))
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

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=config.get("num_workers", 4),
        collate_fn=dino_collate_fn,
        pin_memory=True,
        persistent_workers=config.get("num_workers", 4) > 0,
        prefetch_factor=4 if config.get("num_workers", 4) > 0 else None,
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
        drop_path_rate=0.0,  # teacher: no stochastic depth
        num_register_tokens=num_register_tokens,
        ffn_layer=ffn_layer,
        init_values=init_values,
    )

    student = KronosDINOv2v3(backbone_student, embed_dim, config["out_dim"], ibot_out_dim).to(device)
    teacher = KronosDINOv2v3(backbone_teacher, embed_dim, config["out_dim"], ibot_out_dim).to(device)

    # Initialize teacher from student weights
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
            warmup_teacher_temp=config.get("ibot_warmup_teacher_temp", config["warmup_teacher_temp"]),
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

    print(f"KRONOS {training_mode.upper()} | {model_name} | embed={embed_dim} | "
          f"patch={patch_size} | markers={num_markers} | "
          f"registers={num_register_tokens} | ffn={ffn_layer}")
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

        for batch_idx, (crops, channel_ids, dataset_names, img_paths) in enumerate(
            tqdm(train_dataloader, desc=f"Epoch {epoch}")
        ):
            global_step = niter_per_ep * epoch + batch_idx
            sched_idx = min(global_step, len(lr_schedule) - 1)

            # Update LR and WD
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule[sched_idx]
                param_group["weight_decay"] = wd_schedule[sched_idx]

            # Move to device
            crops = [c.to(device, dtype=torch.float32, non_blocking=True) for c in crops]
            channel_ids = channel_ids.to(device, dtype=torch.long, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=autocast_dtype):

                if training_mode == "dinov2":
                    # === DINOv2: CLS-only distillation ===
                    student_output = student.forward_dino_multicrop(crops, channel_ids)

                    with torch.no_grad():
                        teacher_output = teacher.forward_dino_multicrop(crops[:n_global], channel_ids)

                    loss_dino = dino_loss_fn(student_output, teacher_output, epoch)
                    total_loss = loss_dino

                else:
                    # === DINOv3: CLS + iBOT + KoLeo ===
                    B = crops[0].shape[0]
                    C_markers = channel_ids.shape[1]
                    # Use backbone's actual patch count per marker (accounts for stride)
                    n_spatial = student.backbone.patch_embed.num_patches

                    # Generate spatial iBOT masks for each global crop independently
                    ibot_masks_per_crop = [
                        generate_spatial_ibot_mask(B, C_markers, n_spatial, ibot_mask_ratio, device)
                        for _ in range(n_global)
                    ]

                    # TEACHER: global crops, no masking
                    with torch.no_grad():
                        teacher_cls_all = []
                        teacher_patch_masked_all = []
                        for gi in range(n_global):
                            t_cls, t_patches = teacher.forward_features_single(
                                crops[gi], channel_ids, masks=None
                            )
                            teacher_cls_all.append(t_cls)
                            # iBOT targets: teacher patch tokens at masked positions
                            teacher_patch_masked_all.append(
                                teacher.ibot_head(t_patches[ibot_masks_per_crop[gi]])
                            )
                        teacher_dino_out = teacher.dino_head(torch.cat(teacher_cls_all, dim=0))
                        teacher_ibot_out = torch.cat(teacher_patch_masked_all, dim=0)

                    # STUDENT: global crops WITH iBOT masking
                    student_cls_global = []
                    student_ibot_all = []
                    student_cls_features_for_koleo = []

                    for gi in range(n_global):
                        s_cls, s_patches = student.forward_features_single(
                            crops[gi], channel_ids, masks=ibot_masks_per_crop[gi]
                        )
                        student_cls_global.append(s_cls)
                        student_cls_features_for_koleo.append(s_cls)
                        # iBOT predictions: student patch tokens at masked positions
                        student_ibot_all.append(
                            student.ibot_head(s_patches[ibot_masks_per_crop[gi]])
                        )

                    # STUDENT: local crops (no masking, DINO only)
                    student_cls_local = []
                    for li in range(n_global, ncrops):
                        s_cls, _ = student.forward_features_single(
                            crops[li], channel_ids, masks=None
                        )
                        student_cls_local.append(s_cls)

                    # DINO head on all student CLS tokens
                    student_dino_out = student.dino_head(
                        torch.cat(student_cls_global + student_cls_local, dim=0)
                    )
                    student_ibot_out = torch.cat(student_ibot_all, dim=0)

                    # Losses
                    loss_dino = dino_loss_fn(student_dino_out, teacher_dino_out, epoch)
                    loss_ibot = ibot_loss_fn(student_ibot_out, teacher_ibot_out, epoch)
                    loss_koleo = koleo_loss_fn(torch.cat(student_cls_features_for_koleo, dim=0))

                    total_loss = loss_dino + ibot_weight * loss_ibot + koleo_weight * loss_koleo

                    running_dino += loss_dino.item()
                    running_ibot += loss_ibot.item()
                    running_koleo += loss_koleo.item()

            # Backward with gradient accumulation
            accumulated_loss = total_loss / grad_accum_steps
            scaler.scale(accumulated_loss).backward()

            if ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(train_dataloader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # EMA teacher update — every iteration
            m = momentum_schedule[sched_idx]
            with torch.no_grad():
                for param_s, param_t in zip(student.parameters(), teacher.parameters()):
                    param_t.data.mul_(m).add_((1 - m) * param_s.detach().data)

            running_loss += total_loss.item()

            # Comet logging
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
                f"{checkpoints_path}/kronos_{training_mode}-{run_name}-epoch_{epoch}.pth",
            )

    # Final save
    final_dict = {
        "student_state_dict": student.state_dict(),
        "teacher_state_dict": teacher.state_dict(),
        "epoch": config["epochs"] - 1,
        "config": config,
    }
    torch.save(final_dict, f"{checkpoints_path}/kronos_{training_mode}-{run_name}-final.pth")

    print("Training finished!")
    finish_experiment()


if __name__ == "__main__":
    main()
