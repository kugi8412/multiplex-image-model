#!/usr/bin/env python
# -*- coding: utf-8 -*-
# train_pixelcnn.py


"""
Train a PixelCNN on frozen ImmuVis encoder latent representations.

Step 1: Load a trained ImmuVis autoencoder and freeze it.
Step 2: Extract latent features Z = Encoder(X) for the training set.
Step 3: Train a PixelCNN to model p(Z) autoregressively.

The trained PixelCNN can then be used for:
    - Latent-space anomaly detection (surprise maps)
    - Conditional sampling for counterfactual reasoning
    - Marker attribution via conditional likelihood ratios

Usage::

    python train_pixelcnn.py configs/train_mambaswin_config.yaml \\
        --encoder-checkpoint checkpoints/final_model.pth \\
        --distribution evidential \\
        --epochs 100 \\
        --pixelcnn-hidden 256 \\
        --pixelcnn-layers 8
"""


import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from ruamel.yaml import YAML
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler, TestCrop
from multiplex_model.modules import MultiplexAutoencoder
from multiplex_model.modules.pixelcnn import LatentPixelCNN
from multiplex_model.utils.configuration import TrainingConfig
from multiplex_model.utils import init_experiment, finish_experiment, get_run_name


# ------------------------------------------------------------------
# Phase 1: Extract latent features
# ------------------------------------------------------------------

@torch.no_grad()
def extract_latents(
    model: MultiplexAutoencoder,
    dataloader: DataLoader,
    device: str,
) -> list[torch.Tensor]:
    """Extract latent feature maps from a frozen encoder.

    Returns a list of tensors, each (D, H', W').
    """
    model.eval()
    latents = []
    for img, channel_ids, panel_idx, img_path in tqdm(dataloader, desc="Extracting latents"):
        img = img.to(device, dtype=torch.float32)
        channel_ids = channel_ids.to(device, dtype=torch.long)

        with autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
            z = model.encode(img, channel_ids)["output"]  # (B, D, H', W')

        for b in range(z.shape[0]):
            latents.append(z[b].float().cpu())

    return latents


# ------------------------------------------------------------------
# Phase 2: Train PixelCNN
# ------------------------------------------------------------------

def train_pixelcnn(
    pixelcnn: LatentPixelCNN,
    train_latents: torch.Tensor,
    val_latents: torch.Tensor,
    device: str,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 3e-4,
    final_lr: float = 1e-5,
    weight_decay: float = 0.01,
    save_dir: str = "checkpoints",
    save_freq: int = 10,
):
    """Train the PixelCNN on extracted latents."""
    pixelcnn = pixelcnn.to(device)
    pixelcnn.train()

    train_ds = TensorDataset(train_latents)
    val_ds = TensorDataset(val_latents)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = optim.AdamW(pixelcnn.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = len(train_loader) * epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=final_lr,
    )

    scaler = GradScaler()
    os.makedirs(save_dir, exist_ok=True)
    run_name = get_run_name()
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # --- Train ---
        pixelcnn.train()
        train_loss_acc = 0.0
        for (z_batch,) in tqdm(train_loader, desc=f"PixelCNN Epoch {epoch}"):
            z_batch = z_batch.to(device)

            with autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                loss = pixelcnn.loss(z_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(pixelcnn.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            train_loss_acc += loss.item()

        avg_train = train_loss_acc / len(train_loader)

        # Validate
        pixelcnn.eval()
        val_loss_acc = 0.0
        with torch.no_grad():
            for (z_batch,) in val_loader:
                z_batch = z_batch.to(device)
                with autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                    loss = pixelcnn.loss(z_batch)
                val_loss_acc += loss.item()

        avg_val = val_loss_acc / len(val_loader)

        print(f"Epoch {epoch} | Train NLL: {avg_train:.4f} | Val NLL: {avg_val:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "pixelcnn_state_dict": pixelcnn.state_dict(),
                "epoch": epoch,
                "val_nll": avg_val,
            }, f"{save_dir}/pixelcnn_best-{run_name}.pth")

        # Periodic save
        if (epoch + 1) % save_freq == 0:
            torch.save({
                "pixelcnn_state_dict": pixelcnn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
            }, f"{save_dir}/pixelcnn-{run_name}-epoch_{epoch}.pth")

    # Final save
    torch.save({
        "pixelcnn_state_dict": pixelcnn.state_dict(),
    }, f"{save_dir}/pixelcnn_final-{run_name}.pth")

    return pixelcnn


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train PixelCNN on frozen ImmuVis latents")
    parser.add_argument("config", help="ImmuVis training config YAML")
    parser.add_argument("--encoder-checkpoint", required=True,
                        help="Path to trained ImmuVis model checkpoint")
    parser.add_argument("--distribution", default="evidential",
                        choices=["gaussian", "discretized_logistic_mixture", "evidential"])
    parser.add_argument("--pixelcnn-hidden", type=int, default=256)
    parser.add_argument("--pixelcnn-layers", type=int, default=8)
    parser.add_argument("--pixelcnn-kernel", type=int, default=3)
    parser.add_argument("--n-mixtures", type=int, default=5,
                        help="Number of logistic mixture components (DLM only)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--final-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--evidence-reg-coeff", type=float, default=0.01)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--save-freq", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-latents", default=None,
                        help="Path to cache extracted latents (.pt file)")
    args = parser.parse_args()

    yaml_loader = YAML(typ="safe")
    with open(args.config) as f:
        raw_config = yaml_loader.load(f)
    config = TrainingConfig(**raw_config)

    device = args.device

    PANEL_CONFIG = YAML().load(open(config.panel_config))
    TOKENIZER = YAML().load(open(config.tokenizer_config))
    num_channels = len(TOKENIZER)

    # Load and freeze ImmuVis encoder
    decoder_cfg = config.decoder_config.model_dump()
    decoder_cfg["num_outputs"] = 4 if config.uncertainty_method == "evidential" else 2
    model = MultiplexAutoencoder(
        num_channels=num_channels,
        encoder_config=config.encoder_config.model_dump(),
        decoder_config=decoder_cfg,
    )
    ckpt = torch.load(args.encoder_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print(f"Loaded ImmuVis encoder from {args.encoder_checkpoint}")

    # Check for cached latents
    if args.cache_latents and os.path.exists(args.cache_latents):
        print(f"Loading cached latents from {args.cache_latents}")
        cache = torch.load(args.cache_latents, map_location="cpu", weights_only=True)
        train_latents = cache["train"]
        val_latents = cache["val"]
    else:
        # Extract latents
        SIZE = config.input_image_size
        test_transform = TestCrop(SIZE[0])

        train_dataset = DatasetFromTIFF(
            panels_config=PANEL_CONFIG, split="train",
            marker_tokenizer=TOKENIZER, transform=test_transform,
            use_preprocessing=False, use_butterworth_filter=True,
            use_clip_normalization=True, file_extension="npy",
        )
        val_dataset = DatasetFromTIFF(
            panels_config=PANEL_CONFIG, split="test",
            marker_tokenizer=TOKENIZER, transform=test_transform,
            use_preprocessing=False, use_butterworth_filter=True,
            use_clip_normalization=True, file_extension="npy",
        )

        train_sampler = PanelBatchSampler(train_dataset, config.batch_size)
        val_sampler = PanelBatchSampler(val_dataset, config.batch_size, shuffle=False)

        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                                  num_workers=config.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                                num_workers=config.num_workers, pin_memory=True)

        train_latent_list = extract_latents(model, train_loader, device)
        train_latents = torch.stack(train_latent_list)
        print(f"  Train: {train_latents.shape}")

        val_latent_list = extract_latents(model, val_loader, device)
        val_latents = torch.stack(val_latent_list)
        print(f"  Val:   {val_latents.shape}")

        if args.cache_latents:
            os.makedirs(os.path.dirname(args.cache_latents) or ".", exist_ok=True)
            torch.save({"train": train_latents, "val": val_latents}, args.cache_latents)
            print(f"Cached latents to {args.cache_latents}")

    latent_dim = train_latents.shape[1]
    print(f"Latent dim: {latent_dim}, spatial: {train_latents.shape[2]}×{train_latents.shape[3]}")

    # Build PixelCNN
    pixelcnn = LatentPixelCNN(
        latent_dim=latent_dim,
        hidden_dim=args.pixelcnn_hidden,
        n_layers=args.pixelcnn_layers,
        kernel_size=args.pixelcnn_kernel,
        distribution=args.distribution,
        n_mixtures=args.n_mixtures,
        evidence_reg_coeff=args.evidence_reg_coeff,
    )

    n_params = sum(p.numel() for p in pixelcnn.parameters())
    print(f"PixelCNN: {n_params:,} parameters, distribution={args.distribution}")

    # Train
    init_experiment({
        **raw_config,
        "pixelcnn_hidden": args.pixelcnn_hidden,
        "pixelcnn_layers": args.pixelcnn_layers,
        "pixelcnn_distribution": args.distribution,
    })

    train_pixelcnn(
        pixelcnn, train_latents, val_latents,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        final_lr=args.final_lr,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        save_freq=args.save_freq,
    )

    finish_experiment()
    print("PixelCNN training complete!")


if __name__ == "__main__":
    main()
