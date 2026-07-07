"""Train a sparse autoencoder on latent representations from a frozen model.

Supports two source models:

1. **ImmuVis** (``multiplex_model.modules.MultiplexAutoencoder``)
   - Encoder output: ``(B, D, H', W')`` spatial latent, D = 768
   - Flattened to ``(B*H'*W', D)`` for SAE training

2. **VirTues** (``modules.multiplex_virtues.MultiplexVirtues``)
   - Encoder output: ``MultiplexEncoderOutput`` with per-channel token
     tensors ``List[(C_i, H, W, D)]`` and summary ``List[(H, W, D)]``
   - ``patch_summary`` tokens flattened to ``(H*W, D)`` for SAE training
   - Alternatively, ``encoded_multiplex`` gives per-channel tokens

Usage::

    # Train SAE on ImmuVis encoder latents
    python -m cross_sae.train_sae \\
        --source immuvis \\
        --config configs/train_vit_config.yaml \\
        --checkpoint checkpoints/final_model.pth \\
        --sae-variant topk --k 32 --expansion 8 \\
        --epochs 30 --output-dir sae_models/immuvis_vit/

    # Train SAE on VirTues encoder latents
    python -m cross_sae.train_sae \\
        --source virtues \\
        --virtues-checkpoint path/to/virtues.pth \\
        --virtues-embeddings path/to/marker_embeddings/ \\
        --data-dir path/to/dataset/ \\
        --sae-variant topk --k 32 --expansion 8 \\
        --epochs 30 --output-dir sae_models/virtues/
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from cross_sae.sparse_autoencoder import SAEOutput, build_sae


# ===================================================================
# Latent extraction — ImmuVis
# ===================================================================

@torch.no_grad()
def extract_immuvis_latents(
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda",
    max_batches: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Extract encoder latents from a frozen ImmuVis model.

    Returns:
        latents: (N, D) tensor of flattened spatial latent vectors.
        latent_dim: D.
    """
    from ruamel.yaml import YAML

    from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler, TestCrop
    from multiplex_model.modules import MultiplexAutoencoder
    from multiplex_model.utils.configuration import TrainingConfig

    yaml_loader = YAML(typ="safe")
    with open(config_path) as f:
        raw_config = yaml_loader.load(f)
    config = TrainingConfig(**raw_config)

    PANEL_CONFIG = YAML().load(open(config.panel_config))
    TOKENIZER = YAML().load(open(config.tokenizer_config))
    SIZE = config.input_image_size

    dataset = DatasetFromTIFF(
        panels_config=PANEL_CONFIG,
        split="train",
        marker_tokenizer=TOKENIZER,
        transform=TestCrop(SIZE[0]),
        use_preprocessing=False,
        use_butterworth_filter=True,
        use_clip_normalization=True,
    )
    sampler = PanelBatchSampler(dataset, batch_size=4, shuffle=False)
    dataloader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=config.num_workers,
        pin_memory=True,
    )

    decoder_cfg = config.decoder_config.model_dump()
    decoder_cfg["num_outputs"] = 4 if config.uncertainty_method == "evidential" else 2

    model = MultiplexAutoencoder(
        num_channels=len(TOKENIZER),
        encoder_config=config.encoder_config.model_dump(),
        decoder_config=decoder_cfg,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()

    all_latents = []
    for batch_idx, (img, channel_ids, _panel, _path) in enumerate(
        tqdm(dataloader, desc="Extracting ImmuVis latents")
    ):
        if max_batches is not None and batch_idx >= max_batches:
            break
        img = img.to(device, dtype=torch.float32)
        channel_ids = channel_ids.to(device, dtype=torch.long)

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            z = model.encode(img, channel_ids)["output"]  # (B, D, H', W')

        # Flatten spatial dims: (B, D, H', W') → (B*H'*W', D)
        B, D, Hp, Wp = z.shape
        z = z.permute(0, 2, 3, 1).reshape(-1, D).float().cpu()
        all_latents.append(z)

    latents = torch.cat(all_latents, dim=0)
    print(f"Extracted {latents.shape[0]} latent vectors of dim {latents.shape[1]}")
    return latents, latents.shape[1]


# ===================================================================
# Latent extraction — VirTues
# ===================================================================

@torch.no_grad()
def extract_virtues_latents(
    checkpoint_path: str,
    embeddings_dir: str,
    data_dir: str,
    device: str = "cuda",
    max_batches: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Extract encoder latents from a frozen VirTues model.

    Requires VirTues to be installed (``pip install -e .`` from its repo).
    Uses ``patch_summary_tokens`` as the latent representation, which gives
    spatial latent vectors that are analogous to ImmuVis encoder outputs.

    Returns:
        latents: (N, D) tensor of flattened spatial latent vectors.
        latent_dim: D.
    """
    try:
        from modules.multiplex_virtues import MultiplexVirtues
    except ImportError:
        raise ImportError(
            "VirTues not found. Clone https://github.com/bunnelab/virtues "
            "and install it (pip install -e .) to use this extractor."
        )

    import pandas as pd
    from pathlib import Path

    # Load marker embeddings
    channels_csv = Path(data_dir) / "channels.csv"
    channels_df = pd.read_csv(channels_csv)
    uniprot_ids = channels_df["protein_id"].tolist()

    emb_list = []
    for uid in uniprot_ids:
        emb_path = Path(embeddings_dir) / f"{uid}.pt"
        if emb_path.exists():
            emb_list.append(torch.load(emb_path, map_location="cpu", weights_only=True))
        else:
            print(f"Warning: embedding not found for {uid}, using zeros")
            emb_list.append(torch.zeros(640))  # ESM-2 t30 dim
    prior_embeddings = torch.stack(emb_list)

    # Build model with checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = ckpt.get("config", {})
    model = MultiplexVirtues(
        use_default_config=True,
        prior_bias_embeddings=prior_embeddings.to(device),
        prior_bias_embedding_type=config.get("prior_bias_embedding_type", "esm"),
        **{k: v for k, v in config.items() if k not in (
            "prior_bias_embeddings", "prior_bias_embedding_type", "use_default_config"
        )},
    ).to(device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()
    model_dim = model.encoder.model_dim

    # Load data via numpy crops
    crops_dir = Path(data_dir) / "crops"
    crop_files = sorted(crops_dir.glob("*.npy"))
    if not crop_files:
        raise FileNotFoundError(f"No .npy crop files found in {crops_dir}")

    channel_ids_tensor = torch.arange(len(uniprot_ids), device=device)

    all_latents = []
    for i, crop_path in enumerate(tqdm(crop_files, desc="Extracting VirTues latents")):
        if max_batches is not None and i >= max_batches:
            break
        crop = torch.from_numpy(np.load(crop_path)).float().to(device)  # (C, H, W)

        enc_out = model.encoder.forward_list(
            [crop], [channel_ids_tensor],
        )
        # patch_summary_tokens: List[(H', W', D)] — use as spatial latent
        ps = enc_out.patch_summary_tokens[0]  # (H', W', D)
        Hp, Wp, D = ps.shape
        z = ps.reshape(-1, D).float().cpu()
        all_latents.append(z)

    latents = torch.cat(all_latents, dim=0)
    print(f"Extracted {latents.shape[0]} latent vectors of dim {latents.shape[1]}")
    return latents, model_dim


# ===================================================================
# Latent extraction — ImmunoKRONOS (ImmuvisDINO)
# ===================================================================

@torch.no_grad()
def extract_kronos_latents(
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda",
    max_batches: int | None = None,
    split: str = "train",
) -> tuple[torch.Tensor, int]:
    """Extract CLS-token latents from a frozen ImmunoKRONOS v2 model.

    Returns:
        latents: (N, D) tensor of CLS token embeddings.
        latent_dim: D.
    """
    import sys
    from ruamel.yaml import YAML

    from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler, TestCrop

    yaml_loader = YAML(typ="safe")
    with open(config_path) as f:
        raw_config = yaml_loader.load(f)

    panel_config_path = raw_config.get("panel_config", raw_config.get("panel_config_path"))
    tokenizer_path = raw_config.get("tokenizer_config", raw_config.get("tokenizer_config_path",
                                    "configs/all_markers_tokenizer.yaml"))
    PANEL_CONFIG = YAML().load(open(panel_config_path))
    TOKENIZER = YAML().load(open(tokenizer_path))

    img_size = raw_config.get("global_crops_size", 128)
    if isinstance(img_size, list):
        img_size = img_size[0]

    dataset = DatasetFromTIFF(
        panels_config=PANEL_CONFIG,
        split=split,
        marker_tokenizer=TOKENIZER,
        transform=TestCrop(img_size),
        use_preprocessing=False,
        use_butterworth_filter=True,
        use_clip_normalization=True,
        file_extension=raw_config.get("file_extension", "npy"),
    )
    sampler = PanelBatchSampler(dataset, batch_size=4, shuffle=False)
    dataloader = DataLoader(
        dataset, batch_sampler=sampler,
        num_workers=raw_config.get("num_workers", 4),
        pin_memory=True,
    )

    # Build ImmuvisDINO model
    sys.path.insert(0, os.getcwd())
    from train_kronos_immuvis_v2 import ImmuvisDINO

    num_markers = len(TOKENIZER)
    patch_size = raw_config.get("patch_size", 8)
    out_dim = raw_config.get("out_dim", 65536)

    model = ImmuvisDINO(
        num_markers=num_markers, patch_size=patch_size, out_dim=out_dim,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "student_state_dict" in ckpt:
        model.load_state_dict(ckpt["student_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    all_latents = []
    for batch_idx, (img, channel_ids, _panel, _path) in enumerate(
        tqdm(dataloader, desc=f"Extracting KRONOS latents ({split})")
    ):
        if max_batches is not None and batch_idx >= max_batches:
            break
        img = img.to(device, dtype=torch.float32)
        channel_ids = channel_ids.to(device, dtype=torch.long)

        B, C, H, W = img.shape
        x_hk = img.reshape(B * C, 1, H, W)
        x_enc = model.hyperkernel(x_hk, channel_ids)
        x_enc = x_enc.flatten(2).transpose(1, 2)
        h_p, w_p = H // model.patch_size, W // model.patch_size
        N = h_p * w_p
        cls_tokens = model.cls_token.expand(B, -1, -1)
        x_enc = torch.cat((cls_tokens, x_enc), dim=1)
        x_enc = x_enc + model.pos_embed[:, :N + 1, :]
        for blk in model.blocks:
            x_enc = blk(x_enc)
        x_enc = model.norm(x_enc)

        # CLS token: (B, D)
        cls_out = x_enc[:, 0].float().cpu()
        all_latents.append(cls_out)

    latents = torch.cat(all_latents, dim=0)
    print(f"Extracted {latents.shape[0]} CLS tokens of dim {latents.shape[1]}")
    return latents, latents.shape[1]


# ===================================================================
# SAE training loop
# ===================================================================

def train_sae(
    latents: torch.Tensor,
    sae: torch.nn.Module,
    epochs: int = 30,
    batch_size: int = 4096,
    lr: float = 3e-4,
    device: str = "cuda",
    log_every: int = 100,
) -> dict:
    """Train a sparse autoencoder on pre-extracted latent vectors.

    Args:
        latents: (N, D) tensor of latent vectors.
        sae: An SAE module (VanillaSAE, TopKSAE, or GatedSAE).
        epochs: Number of training epochs.
        batch_size: Batch size for SAE training.
        lr: Learning rate.
        device: Device to train on.
        log_every: Print stats every N batches.

    Returns:
        Dict with training history (losses, L0 stats).
    """
    sae = sae.to(device)
    sae.train()

    # Normalise latents (zero mean, unit variance per dimension)
    mean = latents.mean(dim=0)
    std = latents.std(dim=0).clamp(min=1e-6)
    latents_norm = (latents - mean) / std

    dataset = TensorDataset(latents_norm)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    scaler = GradScaler()

    history = {"loss": [], "mse": [], "sparsity": [], "l0": []}
    global_step = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_mse = 0.0
        n_batches = 0

        for (batch,) in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            batch = batch.to(device)

            with autocast(device_type="cuda", dtype=torch.float16):
                out: SAEOutput = sae(batch)

            scaler.scale(out.loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Normalise decoder columns
            sae.normalise_decoder()

            epoch_loss += out.loss.item()
            epoch_mse += out.mse_loss.item()
            n_batches += 1
            global_step += 1

            if global_step % log_every == 0:
                history["loss"].append(out.loss.item())
                history["mse"].append(out.mse_loss.item())
                history["sparsity"].append(out.sparsity_loss.item())
                history["l0"].append(out.l0.item())

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_mse = epoch_mse / max(n_batches, 1)
        print(
            f"Epoch {epoch + 1}/{epochs}  |  "
            f"Loss: {avg_loss:.6f}  |  MSE: {avg_mse:.6f}  |  "
            f"L0: {out.l0.item():.1f}"
        )

    # Store normalisation stats in history for inference
    history["latent_mean"] = mean.cpu()
    history["latent_std"] = std.cpu()
    return history


# ===================================================================
# CLI entry point
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train a sparse autoencoder on frozen model latents"
    )
    parser.add_argument(
        "--source", required=True, choices=["immuvis", "virtues", "kronos"],
        help="Source model type",
    )

    # ImmuVis args
    parser.add_argument("--config", default=None, help="ImmuVis training config YAML")
    parser.add_argument("--checkpoint", default=None, help="ImmuVis model checkpoint")

    # VirTues args
    parser.add_argument("--virtues-checkpoint", default=None, help="VirTues checkpoint")
    parser.add_argument("--virtues-embeddings", default=None, help="ESM-2 embeddings dir")
    parser.add_argument("--data-dir", default=None, help="VirTues dataset directory")

    # SAE args
    parser.add_argument(
        "--sae-variant", default="topk", choices=["vanilla", "topk", "gated"],
        help="SAE architecture variant (default: topk)",
    )
    parser.add_argument("--expansion", type=int, default=8, help="Hidden dim = expansion × latent_dim")
    parser.add_argument("--l1-coeff", type=float, default=1e-3, help="L1 coefficient (vanilla/gated)")
    parser.add_argument("--k", type=int, default=32, help="Top-k sparsity (topk variant)")
    parser.add_argument("--aux-k", type=int, default=128, help="Auxiliary top-k (topk variant)")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4096, help="SAE training batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max-batches", type=int, default=None, help="Max batches for latent extraction")
    parser.add_argument("--output-dir", default="sae_models/", help="Output directory")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Extract latents
    if args.source == "immuvis":
        if not args.config or not args.checkpoint:
            parser.error("--config and --checkpoint required for ImmuVis")
        latents, latent_dim = extract_immuvis_latents(
            args.config, args.checkpoint, device=device,
            max_batches=args.max_batches,
        )
    elif args.source == "kronos":
        if not args.config or not args.checkpoint:
            parser.error("--config and --checkpoint required for KRONOS")
        latents, latent_dim = extract_kronos_latents(
            args.config, args.checkpoint, device=device,
            max_batches=args.max_batches,
        )
    else:
        if not args.virtues_checkpoint or not args.virtues_embeddings or not args.data_dir:
            parser.error(
                "--virtues-checkpoint, --virtues-embeddings, and --data-dir "
                "required for VirTues"
            )
        latents, latent_dim = extract_virtues_latents(
            args.virtues_checkpoint, args.virtues_embeddings,
            args.data_dir, device=device, max_batches=args.max_batches,
        )

    # Build SAE
    hidden_dim = latent_dim * args.expansion
    sae_kwargs = {}
    if args.sae_variant in ("vanilla", "gated"):
        sae_kwargs["l1_coeff"] = args.l1_coeff
    if args.sae_variant == "topk":
        sae_kwargs["k"] = args.k
        sae_kwargs["aux_k"] = args.aux_k

    sae = build_sae(args.sae_variant, latent_dim, hidden_dim, **sae_kwargs)
    print(f"SAE: {args.sae_variant} | input_dim={latent_dim} | hidden_dim={hidden_dim}")
    print(f"  Parameters: {sum(p.numel() for p in sae.parameters()):,}")

    # Train
    history = train_sae(
        latents, sae, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, device=device,
    )

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "sae_checkpoint.pth")
    torch.save({
        "sae_state_dict": sae.state_dict(),
        "sae_variant": args.sae_variant,
        "input_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "sae_kwargs": sae_kwargs,
        "source": args.source,
        "latent_mean": history["latent_mean"],
        "latent_std": history["latent_std"],
        "history": {k: v for k, v in history.items() if k not in ("latent_mean", "latent_std")},
    }, save_path)
    print(f"Saved SAE checkpoint to {save_path}")


if __name__ == "__main__":
    main()
