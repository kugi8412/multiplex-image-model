#!/usr/bin/env python
# -*- coding: utf-8 -*-
# train_ijepa.py
#
# I-JEPA (Image-based Joint-Embedding Predictive Architecture) training
# for multiplex imaging with ImmuKRONOS encoder.
#
# I-JEPA learns representations by predicting target patch representations
# from context patches in representation space — no data augmentation needed,
# no negative pairs, no softmax. Uses a lightweight predictor transformer
# between context encoder output and target encoder output.
#
# Two modes:
#   - Immu-JEPA (use_hyperkernel: true)  — marker-aware hyperkernel stem
#   - I-JEPA    (use_hyperkernel: false) — shared conv patch embedding (marker-agnostic)
#
# Supports all backbones: kronos_vit, convnext, swin, vim
# Supports all preprocessings: immuvis, virtues
#
# Usage:
#   python train_ijepa.py configs/ijepa_vit.yaml
#   python train_ijepa.py configs/ijepa_convnext.yaml
#   python train_ijepa.py configs/ijepa_vim.yaml
#
# After I-JEPA pretraining, finetune with DINO:
#   python train_immukronos_unified.py configs/immukronos_vit_v2.yaml \
#       --from-checkpoint checkpoints/ijepa_vit/ijepa_kronos_vit-...-encoder.pth

import argparse
import functools
import os
import random
import math

import comet_ml  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
)
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler
from multiplex_model.utils import init_experiment, finish_experiment, get_run_name
from multiplex_model.kronos.vision_transformer import Block as KronosBlock, MemEffAttention

# Reuse ImmuKRONOS model and utilities from unified training script
from train_immukronos_unified import ImmuKRONOS, cosine_scheduler


# ============================================================
# I-JEPA PREDICTOR
# ============================================================

class IJEPAPredictor(nn.Module):
    """Lightweight transformer predictor for I-JEPA.

    Predicts target patch representations from context encoder output.
    Narrower than the main encoder to prevent representation collapse.

    Architecture:
        Linear(embed_dim → predictor_dim) → Transformer blocks → Linear(predictor_dim → embed_dim)

    The predictor receives ONLY the context tokens (sparse) plus their
    position indices. It reconstructs a full-length sequence by placing
    context features at their original positions and learnable mask tokens
    at target positions. After self-attention, only target-position outputs
    are extracted for the loss.
    """

    def __init__(self, embed_dim, predictor_dim=384, depth=6, num_heads=6,
                 max_patches=256):
        super().__init__()
        self.max_patches = max_patches
        self.input_proj = nn.Linear(embed_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, predictor_dim))
        self.blocks = nn.ModuleList([
            KronosBlock(
                dim=predictor_dim, num_heads=num_heads, mlp_ratio=4.0,
                qkv_bias=True, attn_class=MemEffAttention,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, context_tokens, context_indices, n_ctx, target_mask):
        """Predict target representations from sparse context encoder output.

        Args:
            context_tokens: (B, N_ctx, D) encoder output for context patches only.
            context_indices: (B, N_ctx) original position indices of context tokens.
            n_ctx: (B,) actual number of context tokens per sample (rest is padding).
            target_mask: (B, N) bool — True at positions to predict.

        Returns:
            predictions: (total_targets, D) predicted representations at
                target positions across the batch.
        """
        B = context_tokens.shape[0]
        N = target_mask.shape[1]

        # Project context tokens to predictor dimension
        ctx_proj = self.input_proj(context_tokens)  # (B, N_ctx, predictor_dim)
        D_pred = ctx_proj.shape[-1]

        # Build full-length sequence: mask_token at all positions initially
        x = self.mask_token.expand(B, N, -1).clone()  # (B, N, predictor_dim)

        # Scatter context features into their original positions
        for i in range(B):
            nc = n_ctx[i].item()
            idx = context_indices[i, :nc]  # positions of context tokens
            x[i, idx] = ctx_proj[i, :nc]

        # Add positional embeddings
        x = x + self.pos_embed[:, :N, :]

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.output_proj(x)

        # Extract predictions at target positions only
        return x[target_mask]


# ============================================================
# I-JEPA MULTI-BLOCK MASKING
# ============================================================

def generate_ijepa_masks(batch_size, h_p, w_p, num_targets=4,
                          target_scale=(0.15, 0.2),
                          target_aspect_ratio=(0.75, 1.5),
                          device=None):
    """Generate I-JEPA multi-block target masks.

    Samples multiple rectangular target blocks per sample. Target positions
    are masked from the context encoder input and supervised by the predictor.

    Args:
        batch_size: number of samples
        h_p, w_p: patch grid dimensions (height, width in patches)
        num_targets: number of target blocks per sample
        target_scale: (min, max) fraction of total patches per block
        target_aspect_ratio: (min, max) aspect ratio of each block
        device: torch device

    Returns:
        target_mask: (B, N) bool — True at positions to predict (target blocks)
    """
    N = h_p * w_p
    target_mask = torch.zeros(batch_size, N, dtype=torch.bool, device=device)

    for b in range(batch_size):
        for _ in range(num_targets):
            scale = random.uniform(*target_scale)
            n_patches = max(1, int(N * scale))
            ar = random.uniform(*target_aspect_ratio)
            h_block = max(1, int(math.sqrt(n_patches * ar)))
            w_block = max(1, int(n_patches / h_block))
            h_block = min(h_block, h_p)
            w_block = min(w_block, w_p)

            top = random.randint(0, max(0, h_p - h_block))
            left = random.randint(0, max(0, w_p - w_block))

            for row in range(top, min(top + h_block, h_p)):
                start_idx = row * w_p + left
                end_idx = row * w_p + min(left + w_block, w_p)
                target_mask[b, start_idx:end_idx] = True

    return target_mask


def downsample_mask(mask, h_in, w_in, h_out, w_out):
    """Downsample boolean mask from input to output resolution.

    Uses max pooling: an output position is masked if ANY corresponding
    input position is masked. This ensures target blocks are not lost
    when hierarchical backbones reduce spatial dimensions.
    """
    if h_in == h_out and w_in == w_out:
        return mask
    B = mask.shape[0]
    mask_2d = mask.float().reshape(B, 1, h_in, w_in)
    pool_h = h_in // h_out
    pool_w = w_in // w_out
    mask_out = F.max_pool2d(mask_2d, kernel_size=(pool_h, pool_w),
                             stride=(pool_h, pool_w))
    return mask_out.reshape(B, -1).bool()


# ============================================================
# I-JEPA DATA
# ============================================================

class IJEPADatasetWrapper(torch.utils.data.Dataset):
    """Wraps base dataset with a single-crop transform for I-JEPA.

    Unlike DINO multi-crop, I-JEPA uses a single random crop per sample.
    Position information comes from masking, not augmentation diversity.
    """

    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, channel_ids, panel_idx, img_path = self.base_dataset[idx]
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
        return self.transform(img), channel_ids, panel_idx, img_path


def ijepa_collate_fn(batch, channel_fraction=None):
    """Collate for I-JEPA: single image per sample, optional channel dropout."""
    imgs = torch.stack([item[0] for item in batch])
    channel_ids = torch.stack([item[1] for item in batch])
    if channel_fraction is not None:
        C = imgs.shape[1]
        frac = random.uniform(*channel_fraction)
        n_keep = max(1, int(C * frac))
        if n_keep < C:
            perm = torch.randperm(C)[:n_keep].sort().values
            imgs = imgs[:, perm]
            channel_ids = channel_ids[:, perm]
    return imgs, channel_ids


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="I-JEPA training with ImmuKRONOS encoder")
    parser.add_argument("config", help="Path to config YAML")
    parser.add_argument("--from-checkpoint", default=None, help="Resume from checkpoint")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    yaml = YAML(typ="safe")
    with open(args.config, "r") as f:
        config = yaml.load(f)

    device = torch.device(args.device or config.get("device", "cuda"))
    backbone = config.get("backbone", "kronos_vit")
    dino_version = config.get("dino_version", "v2")  # determines ViT block type only
    patch_size = config.get("patch_size", 8)

    print(f"=== I-JEPA | backbone={backbone} | blocks={dino_version} | device={device} ===")
    use_hk = config.get("use_hyperkernel", True)
    stem_name = "Hyperkernel (Immu-JEPA)" if use_hk else "SharedPatchEmbed (I-JEPA)"
    print(f"Stem: {stem_name}")

    # ---- Preprocessing mode ----
    preprocessing = config.get("preprocessing", "immuvis")
    assert preprocessing in ("immuvis", "virtues"), \
        f"preprocessing must be 'immuvis' or 'virtues', got '{preprocessing}'"
    use_butterworth = preprocessing == "immuvis"
    use_clip_norm = preprocessing == "immuvis"
    print(f"Preprocessing: {preprocessing} (butterworth={use_butterworth}, clip_norm={use_clip_norm})")

    # ---- Data ----
    PANEL_CONFIG = YAML().load(open(config["panel_config_path"]))
    TOKENIZER = YAML().load(open(config["tokenizer_config_path"]))
    num_markers = len(TOKENIZER)

    channel_fraction = config.get("channel_fraction", None)
    if channel_fraction is not None:
        channel_fraction = tuple(channel_fraction)
        collate_fn = functools.partial(ijepa_collate_fn, channel_fraction=channel_fraction)
    else:
        collate_fn = ijepa_collate_fn

    # I-JEPA uses a single crop (no multi-crop augmentation)
    crop_size = config.get("crop_size", config.get("global_crops_size", 128))
    crop_scale = config.get("crop_scale", config.get("global_crops_scale", [0.48, 1.0]))
    if isinstance(crop_size, list):
        crop_size = tuple(crop_size)

    ijepa_transform = Compose([
        RandomResizedCrop(crop_size, scale=tuple(crop_scale),
                          interpolation=InterpolationMode.BILINEAR),
        RandomRotation(180, interpolation=InterpolationMode.BILINEAR),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
    ])

    train_base = DatasetFromTIFF(
        panels_config=PANEL_CONFIG, split="train", marker_tokenizer=TOKENIZER,
        transform=None, use_preprocessing=False, use_butterworth_filter=use_butterworth,
        use_clip_normalization=use_clip_norm, file_extension=config.get("file_extension", "npy"),
    )
    train_dataset = IJEPADatasetWrapper(train_base, ijepa_transform)
    train_sampler = PanelBatchSampler(train_base, config["batch_size"])
    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler,
        num_workers=config.get("num_workers", 4), collate_fn=collate_fn,
        pin_memory=True, persistent_workers=config.get("num_workers", 4) > 0,
        prefetch_factor=4 if config.get("num_workers", 4) > 0 else None,
    )

    val_base = DatasetFromTIFF(
        panels_config=PANEL_CONFIG, split="test", marker_tokenizer=TOKENIZER,
        transform=None, use_preprocessing=False, use_butterworth_filter=use_butterworth,
        use_clip_normalization=use_clip_norm, file_extension=config.get("file_extension", "npy"),
    )
    val_dataset = IJEPADatasetWrapper(val_base, ijepa_transform)
    val_sampler = PanelBatchSampler(val_base, config["batch_size"], shuffle=False)
    val_loader = DataLoader(
        val_dataset, batch_sampler=val_sampler,
        num_workers=config.get("num_workers", 4), collate_fn=ijepa_collate_fn,
        pin_memory=True,
    )

    # ---- Model ----
    embed_dim = config.get("embed_dim", 768)
    depth = config.get("depth", 12)
    num_heads = config.get("num_heads", 12)
    backbone_config = config.get("backbone_config", None)

    model_kwargs = dict(
        num_markers=num_markers,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        patch_size=patch_size,
        out_dim=config.get("out_dim", 65536),
        drop_path_rate=config.get("drop_path_rate", 0.1),
        num_register_tokens=config.get("num_register_tokens", 0),
        ibot_out_dim=None,  # No iBOT for I-JEPA
        init_values=config.get("init_values", None),
        mask_strategy=config.get("mask_strategy", "zero"),
        backbone=backbone,
        dino_version=dino_version,
        backbone_config=backbone_config,
        use_hyperkernel=config.get("use_hyperkernel", True),
    )

    # Context encoder (student) and target encoder (teacher — EMA)
    student = ImmuKRONOS(**model_kwargs).to(device)
    teacher_kwargs = {**model_kwargs, "drop_path_rate": 0.0}
    teacher = ImmuKRONOS(**teacher_kwargs).to(device)

    teacher.load_state_dict(student.state_dict(), strict=False)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    # ---- I-JEPA Predictor ----
    predictor_dim = config.get("predictor_dim", 384)
    predictor_depth = config.get("predictor_depth", 6)
    predictor_num_heads = config.get("predictor_num_heads", 6)

    # Compute output spatial dimensions (hierarchical backbones downsample)
    if backbone in ("convnext", "swin"):
        n_stages = len((backbone_config or {}).get("layers_blocks", [depth]))
        downsample_factor = 2 ** (n_stages - 1)
    else:
        downsample_factor = 1

    crop_h = crop_size if isinstance(crop_size, int) else crop_size[0]
    h_p_max = crop_h // patch_size
    n_out_patches = (h_p_max // downsample_factor) ** 2

    predictor = IJEPAPredictor(
        embed_dim=embed_dim,
        predictor_dim=predictor_dim,
        depth=predictor_depth,
        num_heads=predictor_num_heads,
        max_patches=max(n_out_patches, 256),
    ).to(device)

    student_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    predictor_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    print(f"Context encoder: {student_params:,} ({student_params / 1e6:.1f}M)")
    print(f"Predictor: {predictor_params:,} ({predictor_params / 1e6:.1f}M)")

    # ---- Optimizer (encoder + predictor jointly) ----
    optimizer = torch.optim.AdamW(
        list(filter(lambda p: p.requires_grad, student.parameters()))
        + list(predictor.parameters()),
        lr=config["lr"], weight_decay=config["weight_decay"],
    )

    niter_per_ep = len(train_loader)
    lr_schedule = cosine_scheduler(
        config["lr"], config["final_lr"],
        config["epochs"], niter_per_ep, config["warmup_epochs"])
    wd_schedule = cosine_scheduler(
        config["weight_decay"], config["weight_decay_final"],
        config["epochs"], niter_per_ep)
    momentum_schedule = cosine_scheduler(
        config["teacher_momentum"], 1.0,
        config["epochs"], niter_per_ep)

    grad_accum_steps = config.get("gradient_accumulation_steps", 1)
    clip_grad = config.get("clip_grad", 1.0)

    # I-JEPA masking parameters
    num_target_blocks = config.get("num_target_blocks", 4)
    target_scale = tuple(config.get("target_scale", [0.15, 0.2]))
    target_aspect_ratio = tuple(config.get("target_aspect_ratio", [0.75, 1.5]))

    # ---- Resume ----
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
        if "predictor_state_dict" in ckpt:
            predictor.load_state_dict(ckpt["predictor_state_dict"])
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except (ValueError, KeyError):
                print("  Warning: could not load optimizer state (param mismatch), starting fresh")
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"  Resumed at epoch {start_epoch}")

    init_experiment(config)
    experiment = comet_ml.get_global_experiment()
    run_name = get_run_name()

    ckpt_dir = config.get("checkpoints_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"Backbone: {backbone} | blocks: {dino_version} | embed={embed_dim} depth={depth} "
          f"heads={num_heads} patch={patch_size}")
    print(f"Predictor: dim={predictor_dim} depth={predictor_depth} heads={predictor_num_heads}")
    print(f"Masking: {num_target_blocks} target blocks, scale={target_scale}, "
          f"aspect_ratio={target_aspect_ratio}")
    print(f"Batch: {config['batch_size']} × {grad_accum_steps} accum | "
          f"downsample={downsample_factor}×")
    print(f"Training: epochs {start_epoch}..{config['epochs'] - 1} | {niter_per_ep} iters/ep")

    # ============================================================
    # TRAINING LOOP
    # ============================================================

    for epoch in range(start_epoch, config["epochs"]):
        student.train()
        predictor.train()
        teacher.eval()
        r_loss = 0.0

        for batch_idx, (imgs, channel_ids) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            gs = niter_per_ep * epoch + batch_idx
            si = min(gs, len(lr_schedule) - 1)

            for pg in optimizer.param_groups:
                pg["lr"] = lr_schedule[si]
                pg["weight_decay"] = wd_schedule[si]

            imgs = imgs.to(device, dtype=torch.float32, non_blocking=True)
            channel_ids = channel_ids.to(device, dtype=torch.long, non_blocking=True)

            B = imgs.shape[0]
            h_p = imgs.shape[2] // patch_size
            w_p = imgs.shape[3] // patch_size

            # Generate I-JEPA multi-block masks at input resolution
            target_mask_input = generate_ijepa_masks(
                B, h_p, w_p,
                num_targets=num_target_blocks,
                target_scale=target_scale,
                target_aspect_ratio=target_aspect_ratio,
                device=device,
            )

            # Downsample mask for hierarchical backbones
            h_out = h_p // downsample_factor
            w_out = w_p // downsample_factor
            target_mask_output = downsample_mask(
                target_mask_input, h_p, w_p, h_out, w_out
            )

            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                # Target encoder: full image, no masking → ground truth representations
                with torch.no_grad():
                    t_feats = teacher.forward_features(imgs, channel_ids, mask=None)
                    target_reps = t_feats["patch_tokens"][target_mask_output]  # (num_targets, D)

                # Context encoder: only process context (unmasked) patches
                context_mask = ~target_mask_input  # True = keep
                s_feats = student.forward_features_context(imgs, channel_ids, context_mask=context_mask)

                # Predictor: reconstruct full grid → predict target representations
                predictions = predictor(
                    s_feats["context_tokens"],
                    s_feats["context_indices"],
                    s_feats["n_ctx"],
                    target_mask_output,
                )  # (num_targets, D)

                # Loss: smooth L1 in representation space
                loss = F.smooth_l1_loss(predictions, target_reps)

            acc_loss = loss / grad_accum_steps
            scaler.scale(acc_loss).backward()

            if ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(train_loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(student.parameters()) + list(predictor.parameters()), clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # EMA teacher update
                with torch.no_grad():
                    m = momentum_schedule[si]
                    for ps, pt in zip(student.parameters(), teacher.parameters()):
                        pt.data.mul_(m).add_((1 - m) * ps.detach().data)

            r_loss += loss.item()

            if (batch_idx + 1) % 10 == 0 and experiment is not None:
                experiment.log_metrics({
                    "train/ijepa_loss": loss.item(),
                    "train/lr": lr_schedule[si],
                    "train/momentum": momentum_schedule[si],
                }, step=gs)

        # ---- Validation ----
        student.eval()
        predictor.eval()
        v_loss = 0.0
        with torch.no_grad():
            for imgs, channel_ids in tqdm(val_loader, desc=f"Val {epoch}"):
                imgs = imgs.to(device, dtype=torch.float32, non_blocking=True)
                channel_ids = channel_ids.to(device, dtype=torch.long, non_blocking=True)

                B = imgs.shape[0]
                h_p = imgs.shape[2] // patch_size
                w_p = imgs.shape[3] // patch_size
                h_out = h_p // downsample_factor
                w_out = w_p // downsample_factor

                target_mask_input = generate_ijepa_masks(
                    B, h_p, w_p,
                    num_targets=num_target_blocks,
                    target_scale=target_scale,
                    target_aspect_ratio=target_aspect_ratio,
                    device=device,
                )
                target_mask_output = downsample_mask(
                    target_mask_input, h_p, w_p, h_out, w_out
                )

                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    t_feats = teacher.forward_features(imgs, channel_ids, mask=None)
                    target_reps = t_feats["patch_tokens"][target_mask_output]

                    context_mask = ~target_mask_input
                    s_feats = student.forward_features_context(imgs, channel_ids, context_mask=context_mask)
                    predictions = predictor(
                        s_feats["context_tokens"],
                        s_feats["context_indices"],
                        s_feats["n_ctx"],
                        target_mask_output,
                    )
                    v_loss += F.smooth_l1_loss(predictions, target_reps).item()

        n_train = max(len(train_loader), 1)
        n_val = max(len(val_loader), 1)
        print(f"Epoch {epoch} | Train: {r_loss / n_train:.4f} | Val: {v_loss / n_val:.4f}")

        if experiment is not None:
            experiment.log_metrics({
                "epoch/train_loss": r_loss / n_train,
                "epoch/val_loss": v_loss / n_val,
            }, epoch=epoch)

        if (epoch + 1) % config.get("save_checkpoint_freq", 10) == 0:
            torch.save({
                "student_state_dict": student.state_dict(),
                "teacher_state_dict": teacher.state_dict(),
                "predictor_state_dict": predictor.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
                "config": config,
            }, f"{ckpt_dir}/ijepa_{backbone}-{run_name}-epoch_{epoch}.pth")

    # Final saves
    # Full checkpoint (for resuming I-JEPA training)
    torch.save({
        "student_state_dict": student.state_dict(),
        "teacher_state_dict": teacher.state_dict(),
        "predictor_state_dict": predictor.state_dict(),
        "epoch": config["epochs"] - 1,
        "config": config,
    }, f"{ckpt_dir}/ijepa_{backbone}-{run_name}-final.pth")

    # Encoder-only checkpoint (for DINO finetuning / downstream use)
    # Does NOT contain predictor or optimizer — safe to load in the DINO script
    torch.save({
        "student_state_dict": student.state_dict(),
        "teacher_state_dict": teacher.state_dict(),
        "epoch": config["epochs"] - 1,
        "config": config,
    }, f"{ckpt_dir}/ijepa_{backbone}-{run_name}-encoder.pth")

    print("I-JEPA training finished!")
    finish_experiment()


if __name__ == "__main__":
    main()
