#!/usr/bin/env python
# -*- coding: utf-8 -*-
# finetune_immukronos_decoder.py
#
# Finetune a decoder on top of a frozen ImmunoKronos (ImmuvisDINO) encoder,
# producing a MultiplexAutoencoder-compatible model that can be evaluated
# with run_validation_leave_one_out.py for fair comparison against
# ViT Baseline (Exp 7c) and Finetuned DINO (Exp 5a/5b).
#
# Architecture:
#   - Encoder: Frozen ImmuvisDINO backbone (Hyperkernel + ViT blocks)
#     wrapped as a MultiplexAutoencoder-compatible encoder.
#   - Decoder: Trainable ViT-based decoder (same as train_masked_model.py).
#
# Usage:
#   python finetune_immukronos_decoder.py \
#       --kronos-config configs/exp6a_immukronos_v2.yaml \
#       --kronos-checkpoint checkpoints/exp6a_immukronos_v2/kronos_dino-<RUN>-epoch_199.pth \
#       --decoder-config configs/finetune_immukronos_decoder.yaml \
#       --epochs 100 --lr 3e-4

import argparse
import os
import sys

import comet_ml  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ruamel.yaml import YAML
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, RandomRotation
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler, TestCrop
from multiplex_model.losses import (
    beta_nll_loss,
    evidential_loss,
    evidential_uncertainty,
    get_output_activation,
)
from multiplex_model.modules.immuvis import Hyperkernel, MultiplexImageDecoder
from multiplex_model.utils import (
    ClampWithGrad,
    apply_channel_masking,
    finish_experiment,
    get_run_name,
    get_scheduler_with_warmup,
    init_experiment,
    log_training_metrics,
    log_validation_images,
    log_validation_metrics,
    plot_reconstructs_with_masks,
)

import matplotlib.pyplot as plt

# Import ImmuvisDINO from the training script
from train_kronos_immuvis_v2 import ImmuvisDINO

# Also support v3
try:
    from train_kronos_immuvis_v3 import ImmuvisDINOv3
except ImportError:
    ImmuvisDINOv3 = None


class ImmukronosAutoencoder(nn.Module):
    """Wraps a frozen ImmunoKronos encoder + trainable decoder into a
    MultiplexAutoencoder-compatible interface for run_validation_leave_one_out.py.

    The encode() and decode() methods match MultiplexAutoencoder's signatures so
    the leave-one-out script can call model(masked_img, active_ch_ids, output_ch_ids).
    """

    def __init__(self, kronos_model, decoder, num_channels, patch_size, embed_dim,
                 version="v2", freeze_encoder=True):
        super().__init__()
        self.version = version
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_channels = num_channels

        # Encoder components from frozen KRONOS model
        self.hyperkernel = kronos_model.hyperkernel
        self.cls_token = kronos_model.cls_token
        self.blocks = kronos_model.blocks
        self.norm = kronos_model.norm

        if version == "v2":
            self.pos_embed = kronos_model.pos_embed
            self.rope = None
            self.register_tokens = None
        elif version == "v3":
            self.rope = kronos_model.rope if hasattr(kronos_model, "rope") else None
            self.register_tokens = kronos_model.register_tokens if hasattr(kronos_model, "register_tokens") else None
            self.pos_embed = None

        if freeze_encoder:
            for name, param in self.named_parameters():
                if name.startswith("decoder"):
                    continue
                param.requires_grad = False

        # Trainable decoder
        self.decoder = decoder

    def encode(self, x, encoded_indices, return_features=False, spatial_mask=None):
        """Extract spatial features from the KRONOS encoder."""
        B, C, H, W = x.shape
        h_p, w_p = H // self.patch_size, W // self.patch_size
        N = h_p * w_p

        x_enc = self.hyperkernel(x, encoded_indices)
        x_enc = x_enc.flatten(2).transpose(1, 2)

        # CLS token + positional embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_enc = torch.cat((cls_tokens, x_enc), dim=1)

        if self.version == "v2" and self.pos_embed is not None:
            x_enc = x_enc + self.pos_embed[:, :N + 1, :]

        # Transformer blocks
        for blk in self.blocks:
            x_enc = blk(x_enc)
        x_enc = self.norm(x_enc)

        # Extract patch tokens (drop CLS)
        patch_tokens = x_enc[:, 1:, :]  # (B, N, embed_dim)

        # Reshape to spatial feature map
        spatial_features = patch_tokens.transpose(1, 2).reshape(B, self.embed_dim, h_p, w_p)

        result = {"output": spatial_features}
        if return_features:
            result["features"] = [spatial_features]
        return result

    def decode(self, x, decoded_indices):
        """Decode spatial features to pixel-space reconstruction."""
        return self.decoder(x, decoded_indices)

    def forward(self, x, encoded_indices, decoded_indices, return_features=False,
                spatial_mask=None):
        """Full forward pass: encode → decode."""
        enc_out = self.encode(x, encoded_indices, return_features=return_features,
                              spatial_mask=spatial_mask)
        latent = enc_out["output"]
        recon = self.decode(latent, decoded_indices)
        result = {"output": recon}
        if return_features and "features" in enc_out:
            result["features"] = enc_out["features"]
        return result


def load_kronos_checkpoint(config, checkpoint_path, device, version="v2"):
    """Load a pretrained ImmunoKronos model from checkpoint."""
    yaml = YAML(typ="safe")
    with open(config.get("tokenizer_config_path", "configs/all_markers_tokenizer.yaml")) as f:
        tokenizer = yaml.load(f)
    num_markers = len(tokenizer)

    patch_size = config.get("patch_size", 8)
    out_dim = config.get("out_dim", 65536)
    embed_dim = config.get("embed_dim", 768)
    depth = config.get("depth", 12)
    num_heads = config.get("num_heads", 12)

    if version == "v2":
        model = ImmuvisDINO(
            num_markers=num_markers,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            patch_size=patch_size,
            out_dim=out_dim,
        )
    elif version == "v3":
        if ImmuvisDINOv3 is None:
            raise ImportError("ImmuvisDINOv3 not available. Check train_kronos_immuvis_v3.py.")
        model = ImmuvisDINOv3(
            num_markers=num_markers,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            patch_size=patch_size,
            out_dim=out_dim,
            num_register_tokens=config.get("num_register_tokens", 4),
            ibot_out_dim=config.get("ibot_out_dim", 8192),
            init_values=config.get("init_values", None),
            mask_strategy=config.get("mask_strategy", "learnable"),
        )
    else:
        raise ValueError(f"Unknown version: {version}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "student_state_dict" in ckpt:
        model.load_state_dict(ckpt["student_state_dict"], strict=False)
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    return model, num_markers, tokenizer


def build_decoder(decoder_config, num_channels, embed_dim):
    """Build a MultiplexImageDecoder matching train_masked_model.py conventions."""
    dec_cfg = dict(decoder_config)
    dec_cfg.setdefault("block_type", "convnext")
    dec_cfg.setdefault("num_outputs", 2)
    if "hyperkernel" in dec_cfg and "hyperkernel_config" not in dec_cfg:
        dec_cfg["hyperkernel_config"] = dec_cfg.pop("hyperkernel")

    # Używamy **dec_cfg aby "wypakować" słownik bezpośrednio do argumentów
    decoder = MultiplexImageDecoder(
        input_embedding_dim=embed_dim,
        num_channels=num_channels,
        **dec_cfg 
    )
    return decoder


def apply_channel_masking_simple(img, channel_ids, min_frac=0.75, max_fully_masked_frac=0.5):
    """Channel masking for training: encode subset, decode ALL.

    Returns:
        img: original image (B, C, H, W) — reconstruction target
        channel_ids: all channel IDs (B, C) — decode target IDs
        masked_img: image with some channels dropped (B, C_active, H, W)
        active_ch_ids: IDs of kept channels (B, C_active)
    """
    return apply_channel_masking(
        img, channel_ids,
        min_channels_frac=min_frac,
        fully_masked_channels_max_frac=max_fully_masked_frac,
        apply_channel_subset_sampling=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a decoder on frozen ImmunoKronos encoder."
    )
    parser.add_argument("--kronos-config", required=True, help="KRONOS training config YAML")
    parser.add_argument("--kronos-checkpoint", required=True, help="KRONOS checkpoint path")
    parser.add_argument("--decoder-config", default=None,
                        help="Optional decoder config YAML. If not given, uses defaults.")
    parser.add_argument("--version", default="v2", choices=["v2", "v3"],
                        help="ImmunoKronos version (v2 or v3)")
    parser.add_argument("--freeze-encoder", action="store_true", default=True,
                        help="Freeze encoder (default: True)")
    parser.add_argument("--no-freeze-encoder", dest="freeze_encoder", action="store_false",
                        help="Allow encoder finetuning")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--final-lr", type=float, default=8e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--min-channels-frac", type=float, default=0.75)
    parser.add_argument("--fully-masked-channels-max-frac", type=float, default=0.5)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--uncertainty-method", default="beta_nll",
                        choices=["beta_nll", "evidential"])
    parser.add_argument("--evidence-reg-coeff", type=float, default=0.01)
    parser.add_argument("--output-activation", default="sigmoid",
                        choices=["sigmoid", "hardsigmoid", "none"])
    parser.add_argument("--checkpoints-dir", default="checkpoints/immukronos_finetuned")
    parser.add_argument("--save-checkpoint-freq", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frac-warmup-steps", type=float, default=0.1)
    parser.add_argument("--tags", nargs="*", default=["immukronos", "finetuned", "decoder"])
    parser.add_argument("--resume", default=None,
                        help="Path to a finetuning checkpoint to resume from.")
    parser.add_argument("--comet-project", default="immu-vis")
    parser.add_argument("--comet-workspace", default="kugi8412")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load KRONOS config
    yaml = YAML(typ="safe")
    with open(args.kronos_config) as f:
        kronos_config = yaml.load(f)

    # Load KRONOS model
    print(f"Loading ImmunoKronos {args.version} from {args.kronos_checkpoint}")
    kronos_model, num_markers, tokenizer = load_kronos_checkpoint(
        kronos_config, args.kronos_checkpoint, device, version=args.version
    )
    inv_tokenizer = {v: k for k, v in tokenizer.items()}
    embed_dim = kronos_config.get("embed_dim", 768)
    patch_size = kronos_config.get("patch_size", 8)

    # Build decoder
    if args.decoder_config:
        with open(args.decoder_config) as f:
            decoder_config = yaml.load(f)
        dec_cfg = decoder_config.get("decoder", decoder_config)
    else:
        # Default decoder matching ViT baseline / finetune DINO
        scaling_factor = patch_size  # must upsample from patch_size to pixel resolution
        dec_cfg = {
            "decoded_embed_dim": embed_dim,
            "num_blocks": 4,
            "scaling_factor": scaling_factor,
            "num_outputs": 4 if args.uncertainty_method == "evidential" else 2,
            "block_type": {
                "type": "vit",
                "module_parameters": {"num_heads": 12},
            },
            "hyperkernel": {
                "kernel_size": 1,
                "padding": 0,
                "stride": 1,
                "use_bias": True,
            },
        }

    if args.uncertainty_method == "evidential":
        dec_cfg["num_outputs"] = 4
    else:
        dec_cfg.setdefault("num_outputs", 2)

    decoder = build_decoder(dec_cfg, num_channels=num_markers, embed_dim=embed_dim)

    # Build combined model
    model = ImmukronosAutoencoder(
        kronos_model=kronos_model,
        decoder=decoder,
        num_channels=num_markers,
        patch_size=patch_size,
        embed_dim=embed_dim,
        version=args.version,
        freeze_encoder=args.freeze_encoder,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")

    # Data
    panel_config_path = kronos_config.get("panel_config_path",
                                           kronos_config.get("panel_config",
                                                             "configs/all_panels_config.yaml"))
    tokenizer_config_path = kronos_config.get("tokenizer_config_path",
                                               kronos_config.get("tokenizer_config",
                                                                  "configs/all_markers_tokenizer.yaml"))
    with open(panel_config_path) as f:
        panel_config = yaml.load(f)
    with open(tokenizer_config_path) as f:
        tok = yaml.load(f)

    train_transform = Compose([
        RandomCrop(args.crop_size),
        RandomRotation(180, interpolation=InterpolationMode.BILINEAR),
        RandomHorizontalFlip(p=0.5),
    ])
    test_transform = TestCrop(args.crop_size)

    train_dataset = DatasetFromTIFF(
        panels_config=panel_config, split="train", marker_tokenizer=tok,
        transform=train_transform, use_preprocessing=False,
        use_butterworth_filter=True, use_clip_normalization=True,
        file_extension="npy",
    )
    val_dataset = DatasetFromTIFF(
        panels_config=panel_config, split="test", marker_tokenizer=tok,
        transform=test_transform, use_preprocessing=False,
        use_butterworth_filter=True, use_clip_normalization=True,
        file_extension="npy",
    )

    train_sampler = PanelBatchSampler(train_dataset, args.batch_size)
    val_sampler = PanelBatchSampler(val_dataset, args.batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                            num_workers=args.num_workers, pin_memory=True)

    # Optimizer (decoder only if frozen encoder)
    if args.freeze_encoder:
        opt_params = [p for p in model.decoder.parameters() if p.requires_grad]
    else:
        opt_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    num_warmup_steps = int(args.frac_warmup_steps * total_steps)
    num_annealing_steps = total_steps - num_warmup_steps

    scheduler = get_scheduler_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_annealing_steps=num_annealing_steps,
        final_lr=args.final_lr,
    )

    # Logging
    import os

    if args.decoder_config:
        with open(args.decoder_config) as f:
            full_config = yaml.load(f)
            comet_settings = full_config.get("comet", {})
            
            if "api_key" in comet_settings:
                os.environ["COMET_API_KEY"] = comet_settings["api_key"]
                
            args.comet_project = comet_settings.get("project_name", args.comet_project)
            args.comet_workspace = comet_settings.get("workspace", args.comet_workspace)

    config_for_logging = {
        "kronos_config": args.kronos_config,
        "kronos_checkpoint": args.kronos_checkpoint,
        "version": args.version,
        "freeze_encoder": args.freeze_encoder,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "uncertainty_method": args.uncertainty_method,
        "tags": args.tags,
        "comet_project": args.comet_project,
        "comet_workspace": args.comet_workspace,
    }
    init_experiment(config_for_logging)
    experiment = comet_ml.get_global_experiment()
    run_name = get_run_name()

    os.makedirs(args.checkpoints_dir, exist_ok=True)

    # Build marker name mapping for image logging
    marker_names_map = {v: k for k, v in tok.items()}
    num_val_plots = min(4, len(val_dataset))  # number of images to log per epoch

    # ---- Training loop ----
    scaler = GradScaler()
    best_val_loss = float("inf")
    step = 0
    start_epoch = 0

    # Resume from previous finetuning checkpoint
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if "epoch" in resume_ckpt:
            start_epoch = resume_ckpt["epoch"] + 1
            # Advance scheduler to the correct step
            steps_to_skip = start_epoch * len(train_loader)
            for _ in range(steps_to_skip):
                scheduler.step()
            step = steps_to_skip
        if "val_loss" in resume_ckpt:
            best_val_loss = resume_ckpt["val_loss"]
        print(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if args.freeze_encoder:
            model.hyperkernel.eval()
            model.blocks.eval()
            model.norm.eval()

        train_loss_accum = 0.0
        num_batches = 0

        for batch_idx, (img, channel_ids, ds_name, img_path) in enumerate(
            tqdm(train_loader, desc=f"Train Epoch {epoch}")
        ):
            img = img.to(device, dtype=torch.float32)
            channel_ids = channel_ids.to(device, dtype=torch.long)

            # Channel masking: encode subset, decode ALL channels
            img, channel_ids, masked_img, active_ch_ids = apply_channel_masking_simple(
                img, channel_ids,
                min_frac=args.min_channels_frac,
                max_fully_masked_frac=args.fully_masked_channels_max_frac,
            )

            with autocast("cuda", dtype=torch.bfloat16):
                output = model(masked_img, active_ch_ids, channel_ids)["output"]

                if args.uncertainty_method == "evidential":
                    gamma = torch.sigmoid(output[:, :, :, :, 0])
                    nu = F.softplus(output[:, :, :, :, 1]) + 1e-6
                    alpha = F.softplus(output[:, :, :, :, 2]) + 1.0
                    beta_ev = F.softplus(output[:, :, :, :, 3]) + 1e-6
                    loss = evidential_loss(
                        img, gamma, nu, alpha, beta_ev,
                        reg_coeff=args.evidence_reg_coeff,
                    )
                    mi = gamma
                else:
                    # beta_nll: output has 2 channels (mu, logvar)
                    mu = output[..., 0]
                    logvar = output[..., 1]
                    logvar = ClampWithGrad.apply(logvar, -15.0, 15.0)
                    if args.output_activation == "sigmoid":
                        mu = torch.sigmoid(mu)
                    elif args.output_activation == "hardsigmoid":
                        mu = F.hardsigmoid(mu)
                    loss = beta_nll_loss(img, mu, logvar)
                    mi = mu

            loss_scaled = loss / args.gradient_accumulation_steps
            scaler.scale(loss_scaled).backward()

            if ((batch_idx + 1) % args.gradient_accumulation_steps == 0) or \
               ((batch_idx + 1) == len(train_loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                # Log per-step training metrics
                with torch.no_grad():
                    log_dict = {
                        "loss": loss.item(),
                        "lr": scheduler.get_last_lr()[0],
                        "mu": mi.mean().item(),
                        "mae": torch.abs(img - mi).mean().item(),
                        "mse": torch.square(img - mi).mean().item(),
                        "step": step,
                    }
                    if args.uncertainty_method == "evidential":
                        log_dict["aleatoric_unc"] = (beta_ev / (alpha - 1.0 + 1e-8)).mean().item()
                        log_dict["evidence_nu"] = nu.mean().item()
                        log_dict["alpha"] = alpha.mean().item()
                        log_training_metrics(**log_dict)
                    else:
                        log_dict["logvar"] = logvar.mean().item()
                        log_training_metrics(**log_dict)

                step += 1

            train_loss_accum += loss.item()
            num_batches += 1

        avg_train_loss = train_loss_accum / max(num_batches, 1)

        # ---- Validation ----
        model.eval()
        val_loss_accum = 0.0
        val_mse_accum = 0.0
        val_mae_accum = 0.0
        val_batches = 0
        val_latents = []

        # Select random images for plotting
        plot_indices = set(np.random.choice(
            np.arange(len(val_loader)), size=min(num_val_plots, len(val_loader)), replace=False
        ))

        with torch.no_grad():
            for idx, (img, channel_ids, ds_name, img_path) in enumerate(
                tqdm(val_loader, desc=f"Val Epoch {epoch}")
            ):
                img = img.to(device, dtype=torch.float32)
                channel_ids = channel_ids.to(device, dtype=torch.long)

                img, channel_ids, masked_img, active_ch_ids = apply_channel_masking_simple(
                    img, channel_ids, min_frac=args.min_channels_frac,
                )

                with autocast("cuda", dtype=torch.bfloat16):
                    # ROZDZIELAMY ENCODE I DECODE ABY ZEBRAĆ LATENT:
                    enc_out = model.encode(masked_img, active_ch_ids)
                    latent = enc_out["output"]
                    val_latents.append(latent.mean(dim=(2, 3)).float().cpu())
                    output = model.decode(latent, channel_ids)

                    if args.uncertainty_method == "evidential":
                        gamma = torch.sigmoid(output[:, :, :, :, 0])
                        nu = F.softplus(output[:, :, :, :, 1]) + 1e-6
                        alpha = F.softplus(output[:, :, :, :, 2]) + 1.0
                        beta_ev = F.softplus(output[:, :, :, :, 3]) + 1e-6
                        loss = evidential_loss(
                            img, gamma, nu, alpha, beta_ev,
                            reg_coeff=args.evidence_reg_coeff,
                        )
                        mi = gamma
                    else:
                        mu = output[..., 0]
                        logvar = output[..., 1]
                        logvar = ClampWithGrad.apply(logvar, -15.0, 15.0)
                        if args.output_activation == "sigmoid":
                            mu = torch.sigmoid(mu)
                        elif args.output_activation == "hardsigmoid":
                            mu = F.hardsigmoid(mu)
                        loss = beta_nll_loss(img, mu, logvar)
                        mi = mu

                val_loss_accum += loss.item()
                val_mse_accum += torch.square(img - mi).mean().item()
                val_mae_accum += torch.abs(img - mi).mean().item()
                val_batches += 1

                # Log reconstruction images for selected validation samples
                if idx in plot_indices:
                    # Determine which channels were masked
                    unactive_channels = [
                        i for i in channel_ids[0] if i not in active_ch_ids[0]
                    ]
                    masked_channels_names = " | ".join(
                        [marker_names_map.get(i.item(), "?") for i in unactive_channels]
                    )

                    # Create a dummy pixel_mask (no spatial masking → all False)
                    pixel_mask = torch.zeros_like(img, dtype=torch.bool)

                    fig = plot_reconstructs_with_masks(
                        img,
                        mi.float(),
                        pixel_mask,
                        channel_ids,
                        [c.item() if hasattr(c, 'item') else c for c in unactive_channels],
                        markers_names_map=marker_names_map,
                        ncols=9,
                    )
                    log_validation_images(
                        fig=fig,
                        panel_idx=0,
                        img_path=img_path[0] if isinstance(img_path, (list, tuple)) else str(img_path),
                        epoch=epoch,
                        masked_channels_names=masked_channels_names,
                        img_idx=idx,
                    )
                    plt.close("all")

        avg_val_loss = val_loss_accum / max(val_batches, 1)
        avg_val_mse = val_mse_accum / max(val_batches, 1)
        avg_val_mae = val_mae_accum / max(val_batches, 1)

        all_latents = torch.cat(val_latents, dim=0)
        s = torch.linalg.svdvals(all_latents)
        p = (s / s.sum()) + 1e-7
        rankme_val = torch.exp(-(p * torch.log(p)).sum()).item()

        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MSE: {avg_val_mse:.6f} | RankMe: {rankme_val:.2f}")

        log_validation_metrics(
            val_loss=avg_val_loss,
            val_mae=avg_val_mae,
            val_mse=avg_val_mse,
            latent_rankme=rankme_val,
            epoch=epoch,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": avg_val_loss,
            }, os.path.join(args.checkpoints_dir, f"best_model-{run_name}.pth"))

        if (epoch + 1) % args.save_checkpoint_freq == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": avg_val_loss,
            }, os.path.join(args.checkpoints_dir, f"checkpoint-{run_name}-epoch_{epoch}.pth"))

    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs - 1,
    }, os.path.join(args.checkpoints_dir, f"final_model-{run_name}.pth"))

    print("Training complete!")
    finish_experiment()


if __name__ == "__main__":
    main()
