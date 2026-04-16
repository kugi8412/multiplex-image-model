#!/usr/bin/env python
# -*- coding: utf-8 -*-
# train_kronos_dino.py


"""
Finetune KRONOS DINO on the ImmuVis multiplex image dataset.

Performs DINOv2 self-distillation using the KRONOS ViT backbone with
marker-aware patch embedding and sinusoidal marker embeddings.

Usage::
    python train_kronos_dino.py configs/train_kronos_config.yaml
    python train_kronos_dino.py configs/train_kronos_config.yaml --from-checkpoint checkpoints/kronos_dino-epoch_50.pth
"""

import argparse
import os

import comet_ml  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler
from multiplex_model.utils import init_experiment, finish_experiment, get_run_name

from multiplex_model.kronos.vision_transformer import vit_small, vit_base, vit_large
from multiplex_model.kronos.dino_head import DINOHead
from train_kronos_dinov2v3 import resolve_marker_embeddings, build_tokenizer_from_marker_ids


# -------------------------------------------
# 1. KRONOS DINO ARCHITECTURE
# -------------------------------------------

class KronosDINO(nn.Module):
    """Wrapper combining KRONOS ViT backbone with DINOHead projection head."""

    def __init__(self, backbone, embed_dim, out_dim):
        super().__init__()
        self.backbone = backbone
        self.head = DINOHead(
            in_dim=embed_dim,
            out_dim=out_dim,
            hidden_dim=2048,
            bottleneck_dim=256,
            nlayers=3,
        )

    def forward(self, x_list, marker_ids):
        """Student/Teacher forward pass for multiple crops.

        Processes each crop independently through forward_features to
        avoid the xFormers dependency of forward_features_list.

        Args:
            x_list: List of crop tensors, each (B, C_markers, H, W).
            marker_ids: (B, C_markers) token IDs — wrapped as [marker_ids]
                for apply_masks which expects a list of (B, N) tensors.
        """
        cls_tokens = []
        ids_list = [marker_ids]

        for crop in x_list:
            feat = self.backbone.forward_features(crop, masks=None, marker_ids=ids_list)
            cls_tokens.append(feat["x_norm_clstoken"])

        cls_tokens = torch.cat(cls_tokens, dim=0)
        return self.head(cls_tokens)


class DINOLoss(nn.Module):
    """Cross-entropy loss between student and teacher softmax distributions."""

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

    def forward(self, student_output, teacher_output, epoch):
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
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


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
# 2. MULTI-CROP AUGMENTATION & DATASET WRAPPER
# -------------------------------------------

class SpatialProteomicsMultiCrop:
    """Crops image into N global and M local crops. No pixel alterations."""

    def __init__(self, global_scale, local_scale, global_size, local_size,
                 local_crops_num, global_crops_num=2):
        self.global_transform = Compose([
            RandomResizedCrop(global_size, scale=tuple(global_scale),
                              interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
        ])
        self.global_crops_num = global_crops_num
        self.local_crops_num = local_crops_num
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


# -------------------------------------------
# 3. MAIN TRAINING LOOP
# -------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="KRONOS DINO training on ImmuVis")
    parser.add_argument("config", help="KRONOS DINO config YAML")
    parser.add_argument("--from-checkpoint", default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    yaml = YAML(typ="safe")
    with open(args.config, "r") as f:
        config = yaml.load(f)

    # Validate this is a KRONOS DINO config, not a masked-model config
    REQUIRED_KEYS = ["global_crops_scale", "global_crops_size", "local_crops_scale",
                     "local_crops_size", "global_crops_number", "local_crops_number",
                     "model_name", "out_dim", "teacher_temp", "teacher_momentum"]
    missing = [k for k in REQUIRED_KEYS if k not in config]
    if missing:
        # Check if it looks like a masked-model config (has encoder/decoder)
        if "encoder" in config or "decoder" in config:
            raise SystemExit(
                f"ERROR: '{args.config}' is a masked-model config (has encoder/decoder),\n"
                f"but you ran train_kronos_dino.py which needs DINO multi-crop config keys.\n"
                f"Use 'python train_masked_model.py {args.config}' instead.\n"
                f"Missing keys: {missing}"
            )
        raise SystemExit(
            f"ERROR: Config '{args.config}' is missing required keys: {missing}\n"
            f"See configs/train_kronos_config.yaml for an example."
        )

    device = torch.device(args.device or config.get("device", "cuda"))
    print(f"Using device: {device}")

    # ---- Marker embedding resolution ----
    PANEL_CONFIG = YAML().load(open(config["panel_config"]))
    TOKENIZER_RAW = YAML().load(open(config["tokenizer_config"]))

    marker_id_map, num_markers = resolve_marker_embeddings(config, TOKENIZER_RAW)
    TOKENIZER, num_markers = build_tokenizer_from_marker_ids(TOKENIZER_RAW, marker_id_map)

    train_transform = SpatialProteomicsMultiCrop(
        global_scale=config["global_crops_scale"],
        local_scale=config["local_crops_scale"],
        global_size=config["global_crops_size"],
        local_size=config["local_crops_size"],
        local_crops_num=config["local_crops_number"],
        global_crops_num=config["global_crops_number"],
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

    train_dataset_dino = DINODatasetWrapper(train_dataset_base, train_transform)
    train_batch_sampler = PanelBatchSampler(train_dataset_base, config["batch_size"])

    train_dataloader = DataLoader(
        train_dataset_dino,
        batch_sampler=train_batch_sampler,
        num_workers=config.get("num_workers", 4),
        collate_fn=dino_collate_fn,
        pin_memory=True,
        persistent_workers=config.get("num_workers", 4) > 0,
        prefetch_factor=4 if config.get("num_workers", 4) > 0 else None,
    )

    # Model for finetune
    model_name = config["model_name"]
    patch_size = config.get("patch_size", 16)
    img_size = config.get("global_crops_size", [128, 128])
    if isinstance(img_size, list):
        img_size = img_size[0]

    backbone_kwargs = dict(
        patch_size=patch_size,
        stride_size=patch_size,  # non-overlapping patches (standard ViT)
        num_markers=num_markers,
        img_size=img_size,
    )

    if model_name == "vits16" or model_name == "vit_small":
        backbone_student = vit_small(drop_path_rate=config["drop_path_rate"], **backbone_kwargs)
        backbone_teacher = vit_small(drop_path_rate=0.0, **backbone_kwargs)
        embed_dim = 384
    elif model_name == "vitb16" or model_name == "vit_base":
        backbone_student = vit_base(drop_path_rate=config["drop_path_rate"], **backbone_kwargs)
        backbone_teacher = vit_base(drop_path_rate=0.0, **backbone_kwargs)
        embed_dim = 768
    elif model_name == "vitl16" or model_name == "vit_large":
        backbone_student = vit_large(drop_path_rate=config["drop_path_rate"], **backbone_kwargs)
        backbone_teacher = vit_large(drop_path_rate=0.0, **backbone_kwargs)
        embed_dim = 1024
    else:
        raise ValueError(f"Unknown model_name: {model_name}. Use vits16, vitb16, or vitl16.")

    student = KronosDINO(backbone_student, embed_dim, config["out_dim"]).to(device)
    teacher = KronosDINO(backbone_teacher, embed_dim, config["out_dim"]).to(device)

    # Initialize teacher from student (before disabling gradients)
    teacher.load_state_dict(student.state_dict(), strict=False)
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()

    # Loss
    n_global = config["global_crops_number"]
    n_local = config["local_crops_number"]
    dino_loss = DINOLoss(
        out_dim=config["out_dim"],
        ncrops=n_global + n_local,
        warmup_teacher_temp=config["warmup_teacher_temp"],
        teacher_temp=config["teacher_temp"],
        warmup_teacher_temp_epochs=config["warmup_teacher_temp_epochs"],
        nepochs=config["epochs"],
        nglobal_crops=n_global,
    ).to(device)

    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    niter_per_ep = len(train_dataloader)
    total_iters = config["epochs"] * niter_per_ep
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

    # From checkpoint
    start_epoch = 0
    ckpt_path = args.from_checkpoint or config.get("from_checkpoint")

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        student.load_state_dict(ckpt["student_state_dict"])
        teacher.load_state_dict(ckpt["teacher_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "dino_loss_state_dict" in ckpt:
            dino_loss.load_state_dict(ckpt["dino_loss_state_dict"])
        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"  Resumed at epoch {start_epoch}")

    student = student.to(device)
    teacher = teacher.to(device)
    dino_loss = dino_loss.to(device)

    # Logging
    init_experiment(config)
    run_name = get_run_name()

    checkpoints_path = config.get("checkpoints_dir", "checkpoints")
    os.makedirs(checkpoints_path, exist_ok=True)

    grad_accum_steps = config.get("gradient_accumulation_steps", 1)
    clip_grad = config.get("clip_grad", 3.0)
    use_fp16 = config.get("use_fp16", False)
    autocast_dtype = torch.float16 if use_fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    print(f"KRONOS DINO | {model_name} | embed_dim={embed_dim} | "
          f"patches={patch_size}x{patch_size} | markers={num_markers}")
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

        for batch_idx, (crops, channel_ids, dataset_names, img_paths) in enumerate(
            tqdm(train_dataloader, desc=f"Epoch {epoch}")
        ):
            global_step = niter_per_ep * epoch + batch_idx
            # Clamp to schedule length to prevent IndexError
            sched_idx = min(global_step, len(lr_schedule) - 1)

            # Update LR and WD
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule[sched_idx]
                param_group["weight_decay"] = wd_schedule[sched_idx]

            m = momentum_schedule[sched_idx]

            # Move to device
            crops = [c.to(device, dtype=torch.float32, non_blocking=True) for c in crops]
            channel_ids = channel_ids.to(device, dtype=torch.long, non_blocking=True)

            # Forward
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                student_output = student(crops, channel_ids)

                with torch.no_grad():
                    teacher_output = teacher(crops[:n_global], channel_ids)

                loss = dino_loss(student_output, teacher_output, epoch)

            # Backward with gradient accumulation
            accumulated_loss = loss / grad_accum_steps
            scaler.scale(accumulated_loss).backward()

            if ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(train_dataloader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # EMA update teacher — every iteration
            with torch.no_grad():
                for param_s, param_t in zip(student.parameters(), teacher.parameters()):
                    param_t.data.mul_(m).add_((1 - m) * param_s.detach().data)

            running_loss += loss.item()

            # Comet logging
            if (batch_idx + 1) % 10 == 0:
                exp = comet_ml.get_global_experiment()
                if exp is not None:
                    exp.log_metrics({
                        "train/dino_loss": loss.item(),
                        "train/lr": lr_schedule[sched_idx],
                        "train/weight_decay": wd_schedule[sched_idx],
                        "train/teacher_momentum": m,
                    }, step=global_step)

        # End of epoch
        epoch_loss = running_loss / max(len(train_dataloader), 1)
        print(f"Epoch {epoch} | Loss: {epoch_loss:.4f} | "
              f"LR: {lr_schedule[min(niter_per_ep * (epoch + 1) - 1, len(lr_schedule) - 1)]:.2e}")

        if (epoch + 1) % config.get("save_checkpoint_freq", 5) == 0:
            torch.save({
                "student_state_dict": student.state_dict(),
                "teacher_state_dict": teacher.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dino_loss_state_dict": dino_loss.state_dict(),
                "epoch": epoch,
                "config": config,
            }, f"{checkpoints_path}/kronos_dino-{run_name}-epoch_{epoch}.pth")

    # Final save
    torch.save({
        "student_state_dict": student.state_dict(),
        "teacher_state_dict": teacher.state_dict(),
        "epoch": config["epochs"] - 1,
        "config": config,
    }, f"{checkpoints_path}/kronos_dino-{run_name}-final.pth")

    print("Training finished!")
    finish_experiment()


if __name__ == "__main__":
    main()
