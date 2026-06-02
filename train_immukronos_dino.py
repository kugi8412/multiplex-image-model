#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified ImmuKRONOS DINO training with configurable backbone.

Supports:
  - DINOv2 mode (CLS self-distillation)
  - DINOv3 mode (CLS + iBOT + KoLeo + optional Gram)
  - Backbone types: "vit" (default), "convnext", "swin"
  - Configurable hyperkernel, marker-agnostic encoder, pan-marker encoder

Architecture:
  Input (B, C, H, W) → [Optional MA Encoder] → Hyperkernel → PM Encoder → CLS/DINOHead

Usage:
  python train_immukronos_dino.py configs/exp_immukronos_vit_v2.yaml
  python train_immukronos_dino.py configs/exp_immukronos_convnext_v3.yaml
"""

import argparse
import csv
import functools
import math
import os
import random

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
    RandomRotation,
    RandomVerticalFlip,
)
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler
from multiplex_model.modules.immuvis import Hyperkernel, MultiplexImageEncoder
from multiplex_model.modules.registry import resolve_encoder_class
from multiplex_model.kronos.dino_head import DINOHead
from multiplex_model.utils import init_experiment, finish_experiment, get_run_name

# ── Re-use KRONOS-style marker ID resolution ──────────────────────────
from train_kronos_dinov2v3 import (
    resolve_marker_embeddings,
    build_tokenizer_from_marker_ids,
)

# =====================================================================
# 1. BACKBONE ABSTRACTION — wraps different encoder types for DINO
# =====================================================================

class ImmuKRONOSBackbone(nn.Module):
    """Configurable backbone: Hyperkernel + encoder → CLS vector + patch tokens.

    For isotropic encoders (ViT):
      - Prepend learnable CLS token, add pos_embed, run through blocks.
      - CLS = first token, patches = remaining tokens.

    For hierarchical encoders (ConvNeXt, Swin):
      - Run through encoder stages producing spatial feature maps.
      - CLS = global average pooled feature.
      - Patches = flatten(spatial features) for iBOT.
    """

    def __init__(
        self,
        num_channels: int,
        encoder_config: dict,
        dino_out_dim: int = 65536,
        ibot_out_dim: int | None = None,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        # ── Parse encoder config ──
        # Accepts the same format as MultiplexImageEncoder from train_masked_model.py
        ma_layers_blocks = encoder_config.get("ma_layers_blocks", [])
        ma_embedding_dims = encoder_config.get("ma_embedding_dims", [])
        pm_layers_blocks = encoder_config.get("pm_layers_blocks", [16])
        pm_embedding_dims = encoder_config.get("pm_embedding_dims", [768])
        hyperkernel_cfg = encoder_config.get("hyperkernel", {
            "kernel_size": 8, "stride": 8, "padding": 0, "use_bias": True,
        })
        encoder_type = encoder_config.get("encoder_type", "vit")

        # Inject drop_path_rate into encoder parameters
        if isinstance(encoder_type, dict):
            encoder_type = dict(encoder_type)  # copy
            mp = encoder_type.setdefault("module_parameters", {})
            bp = mp.setdefault("block_parameters", {})
            bp["drop_path_rate"] = drop_path_rate
        else:
            encoder_type = {
                "type": encoder_type,
                "module_parameters": {
                    "block_parameters": {"drop_path_rate": drop_path_rate},
                },
            }

        # Determine backbone style
        etype = encoder_type["type"] if isinstance(encoder_type, dict) else encoder_type
        self.is_isotropic = etype in ("vit",)  # ViT is isotropic (flat token sequence)
        self.backbone_type = etype

        # Build the modular encoder (same as MultiplexImageEncoder)
        self.encoder = MultiplexImageEncoder(
            num_channels=num_channels,
            ma_layers_blocks=ma_layers_blocks,
            ma_embedding_dims=ma_embedding_dims,
            hyperkernel_config=hyperkernel_cfg,
            pm_layers_blocks=pm_layers_blocks,
            pm_embedding_dims=pm_embedding_dims,
            use_latent_norm=False,
            encoder_type=encoder_type,
        )

        self.latent_dim = pm_embedding_dims[-1] if pm_embedding_dims else 768
        self.norm = nn.LayerNorm(self.latent_dim)

        # DINO and iBOT heads
        self.dino_head = DINOHead(in_dim=self.latent_dim, out_dim=dino_out_dim)
        self.ibot_head = (
            DINOHead(in_dim=self.latent_dim, out_dim=ibot_out_dim)
            if ibot_out_dim is not None
            else None
        )

    def forward_features(self, x, channel_ids, spatial_mask=None):
        """Extract CLS and patch features.

        Args:
            x: (B, C, H, W) multiplex image.
            channel_ids: (B, C) marker token IDs.
            spatial_mask: (B, C, H, W) boolean mask for iBOT (True=masked), or None.

        Returns:
            dict with "cls_token" (B, D) and "patch_tokens" (B, N, D).
        """
        enc_out = self.encoder(
            x, channel_ids,
            return_features=False,
            spatial_mask=spatial_mask,
        )
        feat_map = enc_out["output"]  # (B, D, H', W')

        B, D, H, W = feat_map.shape

        if self.is_isotropic:
            # ViT already produces a flat token sequence internally.
            # We treat the spatial map as tokens and add a CLS via pooling.
            tokens = feat_map.flatten(2).transpose(1, 2)  # (B, N, D)
            tokens = self.norm(tokens)
            cls_token = tokens.mean(dim=1)  # global average pool as CLS
            return {"cls_token": cls_token, "patch_tokens": tokens}
        else:
            # Hierarchical (ConvNeXt, Swin): spatial feature map
            tokens = feat_map.flatten(2).transpose(1, 2)  # (B, N, D)
            tokens = self.norm(tokens)
            cls_token = tokens.mean(dim=1)  # GAP
            return {"cls_token": cls_token, "patch_tokens": tokens}

    def forward(self, x, channel_ids, spatial_mask=None, return_ibot=False):
        feats = self.forward_features(x, channel_ids, spatial_mask=spatial_mask)
        dino_out = self.dino_head(feats["cls_token"])

        if return_ibot and self.ibot_head is not None and spatial_mask is not None:
            # Get patch-level iBOT mask — downsample spatial_mask to patch level
            B, C, H, W = spatial_mask.shape if spatial_mask is not None else (0, 0, 0, 0)
            # For iBOT we need a flat mask over patch tokens
            # spatial_mask is per-channel, we reduce to per-spatial-position
            patch_mask = spatial_mask.any(dim=1)  # (B, H, W)
            # Downsample to match patch resolution
            n_patches = feats["patch_tokens"].shape[1]
            h_p = int(math.sqrt(n_patches))
            if h_p * h_p != n_patches:
                h_p = n_patches  # fallback for non-square
                patch_mask_ds = F.adaptive_max_pool1d(
                    patch_mask.reshape(B, -1).float().unsqueeze(1), n_patches
                ).squeeze(1) > 0.5
            else:
                patch_mask_ds = F.adaptive_max_pool2d(
                    patch_mask.float().unsqueeze(1), (h_p, h_p)
                ).squeeze(1).reshape(B, -1) > 0.5

            masked_patches = feats["patch_tokens"][patch_mask_ds]
            ibot_out = self.ibot_head(masked_patches)
            return dino_out, ibot_out, patch_mask_ds

        return dino_out


# =====================================================================
# 2. LOSSES (reused from train_kronos_dinov2v3 / train_kronos_immuvis_v3)
# =====================================================================

class DINOLoss(nn.Module):
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

    def forward(self, student_logits, teacher_logits, epoch, update_center=True):
        if student_logits.shape[0] == 0:
            return student_logits.sum() * 0
        temp = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)]
        student_out = student_logits / self.student_temp
        teacher_out = F.softmax((teacher_logits - self.center) / temp, dim=-1).detach()
        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)
        if update_center:
            self._update_center(teacher_logits)
        return loss.mean()

    @torch.no_grad()
    def _update_center(self, teacher_output):
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class KoLeoLoss(nn.Module):
    def forward(self, features):
        features = F.normalize(features, dim=-1, p=2)
        dists = torch.cdist(features, features)
        diag_mask = torch.eye(dists.shape[0], dtype=torch.bool, device=dists.device)
        dists = dists.masked_fill(diag_mask, float("inf"))
        min_dists = dists.min(dim=-1).values
        return -torch.log(min_dists + 1e-8).mean()


# =====================================================================
# 3. MULTI-CROP AUGMENTATION & DATA
# =====================================================================

class MultiCropTransform:
    def __init__(self, global_size, local_size, global_scale, local_scale,
                 local_crops_number, global_crops_number=2):
        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number
        self.global_transform = Compose([
            RandomResizedCrop(global_size, scale=tuple(global_scale),
                              interpolation=InterpolationMode.BILINEAR),
            RandomRotation(180, interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
        ])
        self.local_transform = Compose([
            RandomResizedCrop(local_size, scale=tuple(local_scale),
                              interpolation=InterpolationMode.BILINEAR),
            RandomRotation(180, interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
        ])

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
        crops = []
        for _ in range(self.global_crops_number):
            crops.append(self.global_transform(img))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(img))
        return crops


class DINODatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, channel_ids, panel_idx, img_path = self.base_dataset[idx]
        crops = self.transform(img)
        return crops, channel_ids, panel_idx, img_path


def dino_collate_fn(batch, channel_fraction=None):
    num_crops = len(batch[0][0])
    crops = [torch.stack([item[0][i] for item in batch]) for i in range(num_crops)]
    channel_ids = torch.stack([item[1] for item in batch])
    if channel_fraction is not None:
        C = crops[0].shape[1]
        frac = random.uniform(*channel_fraction)
        n_keep = max(1, int(C * frac))
        if n_keep < C:
            perm = torch.randperm(C)[:n_keep].sort().values
            crops = [crop[:, perm] for crop in crops]
            channel_ids = channel_ids[:, perm]
    return crops, channel_ids


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0):
    warmup_iters = warmup_epochs * niter_per_ep
    warmup_schedule = np.array([]) if warmup_epochs == 0 else np.linspace(0, base_value, warmup_iters)
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    return np.concatenate((warmup_schedule, schedule))


# =====================================================================
# 4. iBOT SPATIAL MASKING
# =====================================================================

def generate_spatial_mask(batch_size, num_channels, img_h, img_w, mask_patch_size,
                          mask_ratio, device):
    """Generate spatial mask at pixel level for iBOT (shared across markers).

    Returns:
        Boolean tensor (B, C, H, W) with True at masked positions.
    """
    h_patches = img_h // mask_patch_size
    w_patches = img_w // mask_patch_size
    n_patches = h_patches * w_patches
    n_masked = max(1, int(n_patches * mask_ratio))

    mask = torch.zeros(batch_size, 1, img_h, img_w, dtype=torch.bool, device=device)
    for b in range(batch_size):
        indices = torch.randperm(n_patches, device=device)[:n_masked]
        patch_y = (indices // w_patches) * mask_patch_size
        patch_x = (indices % w_patches) * mask_patch_size
        for py, px in zip(patch_y, patch_x):
            mask[b, 0, py:py + mask_patch_size, px:px + mask_patch_size] = True

    return mask.expand(-1, num_channels, -1, -1)


# =====================================================================
# 5. MAIN TRAINING LOOP
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="ImmuKRONOS DINO training (multi-backbone)")
    parser.add_argument("config", help="Path to config YAML")
    parser.add_argument("--from-checkpoint", default=None, help="Resume from checkpoint")
    parser.add_argument("--device", default=None)
    parser.add_argument("--marker-metadata-csv", default=None,
                        help="Override marker_metadata_csv from config")
    args = parser.parse_args()

    yaml = YAML(typ="safe")
    with open(args.config, "r") as f:
        config = yaml.load(f)

    device = torch.device(args.device or config.get("device", "cuda"))

    # ── Training mode ──
    training_mode = config.get("training_mode", "dinov2")
    assert training_mode in ("dinov2", "dinov3"), f"training_mode must be dinov2/dinov3, got {training_mode}"

    # ── Marker ID resolution ──
    if args.marker_metadata_csv:
        config["marker_metadata_csv"] = args.marker_metadata_csv

    # Support both old (panel_config_path) and new (panel_config) key names
    panel_path = config.get("panel_config", config.get("panel_config_path"))
    tokenizer_path = config.get("tokenizer_config", config.get("tokenizer_config_path"))
    PANEL_CONFIG = YAML().load(open(panel_path))
    TOKENIZER_RAW = YAML().load(open(tokenizer_path))

    marker_id_map, num_markers = resolve_marker_embeddings(config, TOKENIZER_RAW)
    TOKENIZER, num_markers = build_tokenizer_from_marker_ids(TOKENIZER_RAW, marker_id_map)

    # ── Multi-crop augmentation ──
    n_global = config.get("global_crops_number", 2)
    n_local = config.get("local_crops_number", 6)
    ncrops = n_global + n_local

    transform = MultiCropTransform(
        global_size=config["global_crops_size"],
        local_size=config["local_crops_size"],
        global_scale=config["global_crops_scale"],
        local_scale=config["local_crops_scale"],
        local_crops_number=n_local,
        global_crops_number=n_global,
    )

    # ── Datasets ──
    train_base = DatasetFromTIFF(
        panels_config=PANEL_CONFIG, split="train", marker_tokenizer=TOKENIZER,
        transform=None, use_preprocessing=False, use_butterworth_filter=True,
        use_clip_normalization=True, file_extension=config.get("file_extension", "npy"),
    )
    train_dataset = DINODatasetWrapper(train_base, transform)
    train_sampler = PanelBatchSampler(train_base, config["batch_size"])

    channel_fraction = config.get("channel_fraction", None)
    if channel_fraction is not None:
        channel_fraction = tuple(channel_fraction)
        collate_fn = functools.partial(dino_collate_fn, channel_fraction=channel_fraction)
    else:
        collate_fn = dino_collate_fn

    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler,
        num_workers=config.get("num_workers", 4), collate_fn=collate_fn,
        pin_memory=True, persistent_workers=config.get("num_workers", 4) > 0,
        prefetch_factor=4 if config.get("num_workers", 4) > 0 else None,
    )

    # ── Build student & teacher ──
    encoder_config = config["encoder"]
    ibot_out_dim = config.get("ibot_out_dim", None) if training_mode == "dinov3" else None

    student = ImmuKRONOSBackbone(
        num_channels=num_markers,
        encoder_config=encoder_config,
        dino_out_dim=config["out_dim"],
        ibot_out_dim=ibot_out_dim,
        drop_path_rate=config.get("drop_path_rate", 0.1),
    ).to(device)

    teacher = ImmuKRONOSBackbone(
        num_channels=num_markers,
        encoder_config=encoder_config,
        dino_out_dim=config["out_dim"],
        ibot_out_dim=ibot_out_dim,
        drop_path_rate=0.0,  # teacher has no stochastic depth
    ).to(device)

    teacher.load_state_dict(student.state_dict(), strict=False)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    # ── Losses ──
    dino_loss_fn = DINOLoss(
        out_dim=config["out_dim"], ncrops=ncrops,
        warmup_teacher_temp=config["warmup_teacher_temp"],
        teacher_temp=config["teacher_temp"],
        warmup_teacher_temp_epochs=config["warmup_teacher_temp_epochs"],
        nepochs=config["epochs"], nglobal_crops=n_global,
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

    # ── Optimizer ──
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=config["lr"], weight_decay=config["weight_decay"],
    )

    niter_per_ep = len(train_loader)
    lr_schedule = cosine_scheduler(config["lr"], config["final_lr"], config["epochs"], niter_per_ep, config["warmup_epochs"])
    wd_schedule = cosine_scheduler(config["weight_decay"], config["weight_decay_final"], config["epochs"], niter_per_ep)
    momentum_schedule = cosine_scheduler(config["teacher_momentum"], 1.0, config["epochs"], niter_per_ep)

    # ── Checkpoint resume ──
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

    init_experiment(config)
    run_name = get_run_name()
    ckpt_dir = config.get("checkpoints_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    grad_accum_steps = config.get("gradient_accumulation_steps", 1)
    clip_grad = config.get("clip_grad", 3.0)
    ibot_weight = config.get("ibot_loss_weight", 1.0)
    koleo_weight = config.get("koleo_loss_weight", 0.1)
    mask_ratio = config.get("ibot_mask_ratio", 0.3)
    mask_patch_size = config.get("mask_patch_size", 8)

    backbone_type = encoder_config.get("encoder_type", "vit")
    if isinstance(backbone_type, dict):
        backbone_type = backbone_type.get("type", "vit")

    print(f"ImmuKRONOS {training_mode.upper()} | backbone={backbone_type} | "
          f"markers={num_markers} | epochs={config['epochs']}")
    print(f"Crops: {n_global} global + {n_local} local | "
          f"Batch: {config['batch_size']} x {grad_accum_steps} accum")

    # ── Training loop ──
    for epoch in range(start_epoch, config["epochs"]):
        student.train()
        teacher.eval()
        running = {"total": 0.0, "dino": 0.0, "ibot": 0.0, "koleo": 0.0}

        for batch_idx, (crops, channel_ids) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}")
        ):
            global_step = niter_per_ep * epoch + batch_idx
            sched_idx = min(global_step, len(lr_schedule) - 1)

            for pg in optimizer.param_groups:
                pg["lr"] = lr_schedule[sched_idx]
                pg["weight_decay"] = wd_schedule[sched_idx]

            crops = [c.to(device, dtype=torch.float32, non_blocking=True) for c in crops]
            channel_ids = channel_ids.to(device, dtype=torch.long, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                if training_mode == "dinov2":
                    # ── DINOv2: CLS-only ──
                    with torch.no_grad():
                        teacher_global = torch.cat(crops[:n_global])
                        teacher_cids = channel_ids.repeat(n_global, 1)
                        teacher_out = teacher(teacher_global, teacher_cids)

                    student_all_crops = []
                    for crop in crops:
                        out = student(crop, channel_ids)
                        student_all_crops.append(out)
                    student_out = torch.cat(student_all_crops)

                    loss_dino = dino_loss_fn(student_out, teacher_out, epoch)
                    total_loss = loss_dino

                else:
                    # ── DINOv3: CLS + iBOT + KoLeo ──
                    B = crops[0].shape[0]
                    C_markers = channel_ids.shape[1]
                    img_h, img_w = crops[0].shape[2], crops[0].shape[3]

                    # Generate spatial mask for global crops
                    spatial_masks = [
                        generate_spatial_mask(B, C_markers, img_h, img_w,
                                              mask_patch_size, mask_ratio, device)
                        for _ in range(n_global)
                    ]

                    # TEACHER: global crops, no mask
                    with torch.no_grad():
                        teacher_cls_all = []
                        teacher_patches_masked_all = []
                        for gi in range(n_global):
                            t_feats = teacher.forward_features(crops[gi], channel_ids)
                            teacher_cls_all.append(t_feats["cls_token"])
                            # Get teacher patch tokens at masked positions
                            # Downsample mask to patch resolution
                            n_p = t_feats["patch_tokens"].shape[1]
                            h_p = int(math.sqrt(n_p))
                            mask_ds = F.adaptive_max_pool2d(
                                spatial_masks[gi][:, 0:1].float(), (h_p, h_p)
                            ).squeeze(1).reshape(B, -1) > 0.5
                            t_masked = t_feats["patch_tokens"][mask_ds]
                            teacher_patches_masked_all.append(
                                teacher.ibot_head(t_masked)
                            )
                        teacher_dino_out = teacher.dino_head(torch.cat(teacher_cls_all))
                        teacher_ibot_out = torch.cat(teacher_patches_masked_all)

                    # STUDENT: global crops WITH mask
                    student_cls_global = []
                    student_ibot_all = []
                    student_cls_for_koleo = []
                    for gi in range(n_global):
                        s_feats = student.forward_features(
                            crops[gi], channel_ids, spatial_mask=spatial_masks[gi]
                        )
                        student_cls_global.append(s_feats["cls_token"])
                        student_cls_for_koleo.append(s_feats["cls_token"])
                        # Same mask downsampling
                        n_p = s_feats["patch_tokens"].shape[1]
                        h_p = int(math.sqrt(n_p))
                        mask_ds = F.adaptive_max_pool2d(
                            spatial_masks[gi][:, 0:1].float(), (h_p, h_p)
                        ).squeeze(1).reshape(B, -1) > 0.5
                        s_masked = s_feats["patch_tokens"][mask_ds]
                        student_ibot_all.append(student.ibot_head(s_masked))

                    # STUDENT: local crops, no mask
                    student_cls_local = []
                    for li in range(n_global, ncrops):
                        s_out = student(crops[li], channel_ids)
                        student_cls_local.append(s_out)

                    student_dino_out = student.dino_head(
                        torch.cat(student_cls_global + student_cls_local)
                    )
                    student_ibot_out = torch.cat(student_ibot_all)

                    loss_dino = dino_loss_fn(student_dino_out, teacher_dino_out, epoch)
                    loss_ibot = ibot_loss_fn(student_ibot_out, teacher_ibot_out, epoch)
                    loss_koleo = koleo_loss_fn(torch.cat(student_cls_for_koleo))
                    total_loss = loss_dino + ibot_weight * loss_ibot + koleo_weight * loss_koleo

                    running["ibot"] += loss_ibot.item()
                    running["koleo"] += loss_koleo.item()

            running["dino"] += loss_dino.item()
            running["total"] += total_loss.item()

            accumulated_loss = total_loss / grad_accum_steps
            scaler.scale(accumulated_loss).backward()

            if ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(train_loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # EMA teacher update
            m = momentum_schedule[sched_idx]
            with torch.no_grad():
                for ps, pt in zip(student.parameters(), teacher.parameters()):
                    pt.data.mul_(m).add_((1 - m) * ps.detach().data)

            # Logging
            if (batch_idx + 1) % 10 == 0:
                exp = comet_ml.get_global_experiment()
                if exp is not None:
                    log_dict = {
                        "train/total_loss": total_loss.item(),
                        "train/dino_loss": loss_dino.item(),
                        "train/lr": lr_schedule[sched_idx],
                    }
                    if training_mode == "dinov3":
                        log_dict["train/ibot_loss"] = loss_ibot.item()
                        log_dict["train/koleo_loss"] = loss_koleo.item()
                    exp.log_metrics(log_dict, step=global_step)

        # Epoch summary
        n_batches = max(len(train_loader), 1)
        epoch_msg = f"Epoch {epoch} | Total: {running['total']/n_batches:.4f}"
        if training_mode == "dinov3":
            epoch_msg += (f" (DINO: {running['dino']/n_batches:.4f}, "
                          f"iBOT: {running['ibot']/n_batches:.4f}, "
                          f"KoLeo: {running['koleo']/n_batches:.4f})")
        print(epoch_msg)

        exp = comet_ml.get_global_experiment()
        if exp is not None:
            exp.log_metrics({"epoch/total_loss": running["total"] / n_batches}, epoch=epoch)

        # Save checkpoint
        if (epoch + 1) % config.get("save_checkpoint_freq", 10) == 0:
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
            torch.save(save_dict, f"{ckpt_dir}/immukronos_{training_mode}-{run_name}-epoch_{epoch}.pth")

    # Final save
    torch.save({
        "student_state_dict": student.state_dict(),
        "teacher_state_dict": teacher.state_dict(),
        "epoch": config["epochs"] - 1,
        "config": config,
    }, f"{ckpt_dir}/immukronos_{training_mode}-{run_name}-final.pth")

    print("Training finished!")
    finish_experiment()


if __name__ == "__main__":
    main()
