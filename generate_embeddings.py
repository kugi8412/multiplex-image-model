#!/usr/bin/env python3
# generate_embeddings.py
# -*- coding: utf-8 -*-

"""
Universal embedding generator for all ImmuVis model types.

Supported model types (--model-type):
  immuvis          ImmuVis Hyperkernel autoencoder (encoder only)
  kronos_dino      KRONOS DINO V2 (train_kronos_dino.py checkpoint)
  kronos_dinov2v3  KRONOS DINOv2/v3 (train_kronos_dinov2v3.py checkpoint)
  kronos_immuvis_v2  ImmunoKronos V2 (train_kronos_immuvis_v2.py checkpoint)
  kronos_immuvis_v3  ImmunoKronos V3 (train_kronos_immuvis_v3.py checkpoint)
  kronos_pretrained  Official KRONOS HuggingFace weights
  dino             DINOv2/v3 backbone via timm (from train_masked_model.py with dino encoder)
  mamba            Mamba/MambaSwin encoder (from train_masked_model.py with vim encoder)
  dino_finetune    Finetuned DINO encoder (from train_masked_model.py with dino encoder)
  virtual_staining Virtual staining model (MultiplexAutoencoder, encoder embeddings)

Usage examples:
  python generate_embeddings.py --model-type immuvis \\
      --config configs/train_vit_config.yaml \\
      --checkpoint checkpoints/final_model.pth \\
      --input_dir data/test/dataset1/imgs \\
      --output_dir embeddings/immuvis/

  python generate_embeddings.py --model-type kronos_dinov2v3 \\
      --config configs/train_kronos_dinov2_config.yaml \\
      --checkpoint checkpoints/kronos_dinov2-final.pth \\
      --input_dir data/test/dataset1/imgs \\
      --output_dir embeddings/kronos/

  python generate_embeddings.py --model-type kronos_pretrained \\
      --checkpoint hf_hub:MahmoodLab/kronos \\
      --input_dir data/test/dataset1/imgs \\
      --output_dir embeddings/kronos_pretrained/
"""

import os
import gc
import sys
import glob
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from ruamel.yaml import YAML


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_tokenizer(config):
    tokenizer_path = config.get(
        "tokenizer_config",
        config.get("tokenizer_config_path", "configs/all_markers_tokenizer.yaml"),
    )
    if os.path.exists(tokenizer_path):
        tokenizer = YAML().load(open(tokenizer_path))
    elif os.path.exists("configs/all_markers_tokenizer.yaml"):
        tokenizer = YAML().load(open("configs/all_markers_tokenizer.yaml"))
    else:
        raise FileNotFoundError(
            f"Tokenizer not found at '{tokenizer_path}' or 'configs/all_markers_tokenizer.yaml'"
        )
    return tokenizer


def _resolve_panel_config(config, override_path=None):
    """Load panel config from explicit path or from model config keys."""
    yaml = YAML(typ="safe")
    if override_path is not None:
        path = override_path
    else:
        path = config.get(
            "panel_config",
            config.get("panel_config_path", None),
        )
    if path is None:
        return None
    if not os.path.exists(path):
        print(f"[WARN] Panel config not found at '{path}'")
        return None
    with open(path, "r") as f:
        return yaml.load(f)


def _normalize_encoder_config(raw_encoder):
    """Normalize raw YAML encoder config to the dict format MultiplexAutoencoder expects.

    Adds Pydantic-style defaults and remaps 'hyperkernel' → 'hyperkernel_config'.
    """
    enc = dict(raw_encoder)
    enc.setdefault("ma_layers_blocks", [])
    enc.setdefault("ma_embedding_dims", [])
    enc.setdefault("pm_layers_blocks", [])
    enc.setdefault("pm_embedding_dims", [])
    enc.setdefault("use_latent_norm", True)
    enc.setdefault("encoder_type", "convnext")
    if "hyperkernel" in enc and "hyperkernel_config" not in enc:
        enc["hyperkernel_config"] = enc.pop("hyperkernel")
    return enc


def _normalize_decoder_config(raw_decoder):
    """Normalize raw YAML decoder config to the dict format MultiplexAutoencoder expects.

    Adds Pydantic-style defaults and remaps 'hyperkernel' → 'hyperkernel_config'.
    """
    dec = dict(raw_decoder)
    dec.setdefault("num_outputs", 2)
    dec.setdefault("block_type", "convnext")
    if "hyperkernel" in dec and "hyperkernel_config" not in dec:
        dec["hyperkernel_config"] = dec.pop("hyperkernel")
    return dec


def load_immuvis(config, checkpoint_path, device):
    from multiplex_model.modules.immuvis import MultiplexAutoencoder

    tokenizer = _load_tokenizer(config)
    num_channels = len(tokenizer)
    input_size = tuple(config.get("input_image_size", [128, 128]))

    encoder_cfg = _normalize_encoder_config(config["encoder"])
    decoder_cfg = _normalize_decoder_config(config["decoder"])

    model = MultiplexAutoencoder(
        num_channels=num_channels,
        encoder_config=encoder_cfg,
        decoder_config=decoder_cfg,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    def extract_fn(x, channel_ids):
        enc_out = model.encode(x, channel_ids)
        latent = enc_out["output"]
        if latent.ndim == 4:
            return latent.mean(dim=(2, 3))
        elif latent.ndim == 3:
            return latent.mean(dim=1)
        return latent

    return extract_fn, input_size


def load_kronos_dino(config, checkpoint_path, device):
    from multiplex_model.kronos.vision_transformer import vit_small, vit_base, vit_large

    model_name = config.get("model_name", "vits16")
    patch_size = config.get("patch_size", 8)
    num_markers = len(_load_tokenizer(config))
    img_size = config.get("global_crops_size", [128, 128])
    if isinstance(img_size, list):
        img_size = img_size[0]

    backbone_kwargs = dict(patch_size=patch_size, num_markers=num_markers, img_size=img_size)
    if model_name in ("vits16", "vit_small"):
        backbone = vit_small(**backbone_kwargs)
    elif model_name in ("vitb16", "vit_base"):
        backbone = vit_base(**backbone_kwargs)
    elif model_name in ("vitl16", "vit_large"):
        backbone = vit_large(**backbone_kwargs)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "student_state_dict" in ckpt:
        raw_sd = ckpt["student_state_dict"]
        sd = {k.replace("backbone.", ""): v for k, v in raw_sd.items()
              if not k.startswith("head.") and not k.startswith("dino_head.")}
        backbone.load_state_dict(sd, strict=False)
    else:
        backbone.load_state_dict(ckpt, strict=False)

    backbone = backbone.to(device).eval()
    input_size = (img_size, img_size)

    def extract_fn(x, channel_ids):
        ids_list = [channel_ids]
        feat = backbone.forward_features(x, masks=None, marker_ids=ids_list)
        return feat["x_norm_clstoken"]

    return extract_fn, input_size


def load_kronos_dinov2v3(config, checkpoint_path, device):
    from multiplex_model.kronos.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

    model_name = config.get("model_name", "vitb16")
    patch_size = config.get("patch_size", 8)
    num_markers = len(_load_tokenizer(config))
    img_size = config.get("global_crops_size", [128, 128])
    if isinstance(img_size, list):
        img_size = img_size[0]

    num_register_tokens = config.get("num_register_tokens", 0)
    ffn_layer = config.get("ffn_layer", "mlp")
    init_values = config.get("init_values", None)

    factories = {
        "vits16": vit_small, "vit_small": vit_small,
        "vitb16": vit_base, "vit_base": vit_base,
        "vitl16": vit_large, "vit_large": vit_large,
        "vitg14": vit_giant2, "vit_giant2": vit_giant2,
    }
    factory = factories.get(model_name)
    if factory is None:
        raise ValueError(f"Unknown model_name: {model_name}")

    backbone = factory(
        patch_size=patch_size, num_markers=num_markers, img_size=img_size,
        drop_path_rate=0.0, num_register_tokens=num_register_tokens,
        ffn_layer=ffn_layer, init_values=init_values,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "student_state_dict" in ckpt:
        raw_sd = ckpt["student_state_dict"]
        sd = {k.replace("backbone.", ""): v for k, v in raw_sd.items()
              if "dino_head." not in k and "ibot_head." not in k}
        backbone.load_state_dict(sd, strict=False)
    else:
        backbone.load_state_dict(ckpt, strict=False)

    backbone = backbone.to(device).eval()
    input_size = (img_size, img_size)

    def extract_fn(x, channel_ids):
        ids_list = [channel_ids]
        feat = backbone.forward_features(x, masks=None, marker_ids=ids_list)
        return feat["x_norm_clstoken"]

    return extract_fn, input_size


def load_kronos_immuvis_v2(config, checkpoint_path, device):
    sys.path.insert(0, os.getcwd())
    from train_kronos_immuvis_v2 import ImmuvisDINO

    num_markers = len(_load_tokenizer(config))
    patch_size = config.get("patch_size", 8)
    out_dim = config.get("out_dim", 65536)

    model = ImmuvisDINO(
        num_markers=num_markers, patch_size=patch_size, out_dim=out_dim,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "student_state_dict" in ckpt:
        model.load_state_dict(ckpt["student_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    img_size = config.get("global_crops_size", 128)
    if isinstance(img_size, list):
        img_size = img_size[0]
    input_size = (img_size, img_size)

    def extract_fn(x, channel_ids):
        B, C, H, W = x.shape
        x_hk = x.reshape(B * C, 1, H, W)
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
        return x_enc[:, 0]

    return extract_fn, input_size


def load_kronos_immuvis_v3(config, checkpoint_path, device):
    sys.path.insert(0, os.getcwd())
    from train_kronos_immuvis_v3 import ImmuvisDINOv3

    num_markers = len(_load_tokenizer(config))
    embed_dim = config.get("embed_dim", 768)
    depth = config.get("depth", 12)
    num_heads = config.get("num_heads", 12)
    patch_size = config.get("patch_size", 8)
    out_dim = config.get("out_dim", 65536)
    num_register_tokens = config.get("num_register_tokens", 4)
    ibot_out_dim = config.get("ibot_out_dim", 8192)
    init_values = config.get("init_values", None)
    mask_strategy = config.get("mask_strategy", "learnable")

    model = ImmuvisDINOv3(
        num_markers=num_markers, embed_dim=embed_dim, depth=depth,
        num_heads=num_heads, patch_size=patch_size, out_dim=out_dim,
        drop_path_rate=0.0, num_register_tokens=num_register_tokens,
        ibot_out_dim=ibot_out_dim, init_values=init_values,
        mask_strategy=mask_strategy,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "student_state_dict" in ckpt:
        model.load_state_dict(ckpt["student_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    img_size = config.get("global_crops_size", 128)
    if isinstance(img_size, list):
        img_size = img_size[0]
    input_size = (img_size, img_size)

    def extract_fn(x, channel_ids):
        feats = model.forward_features(x, channel_ids, mask=None)
        return feats["cls_token"]

    return extract_fn, input_size


def load_kronos_pretrained(config, checkpoint_path, device):
    from multiplex_model.kronos.inference import create_model_from_pretrained

    model, _, _ = create_model_from_pretrained(
        checkpoint_path=checkpoint_path,
        cache_dir=config.get("cache_dir", "./model_assets") if config else "./model_assets",
    )
    model = model.to(device).eval()
    input_size = (224, 224)

    def extract_fn(x, channel_ids):
        ids_list = [channel_ids]
        feat = model.forward_features(x, masks=None, marker_ids=ids_list)
        return feat["x_norm_clstoken"]

    return extract_fn, input_size


def load_dino_encoder(config, checkpoint_path, device):
    from multiplex_model.modules.immuvis import MultiplexAutoencoder

    tokenizer = _load_tokenizer(config)
    num_channels = len(tokenizer)
    input_size = tuple(config.get("input_image_size", [128, 128]))

    encoder_cfg = _normalize_encoder_config(config["encoder"])
    decoder_cfg = _normalize_decoder_config(config["decoder"])

    model = MultiplexAutoencoder(
        num_channels=num_channels,
        encoder_config=encoder_cfg,
        decoder_config=decoder_cfg,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    def extract_fn(x, channel_ids):
        enc_out = model.encode(x, channel_ids)
        latent = enc_out["output"]
        if latent.ndim == 4:
            return latent.mean(dim=(2, 3))
        elif latent.ndim == 3:
            return latent.mean(dim=1)
        return latent

    return extract_fn, input_size


def load_masked_model(config, checkpoint_path, device):
    """Load a MultiplexAutoencoder trained via train_masked_model.py.

    Works for any encoder type (vim/Mamba, dino, vit) and both uncertainty
    methods (evidential num_outputs=4, beta_nll num_outputs=2).
    """
    from multiplex_model.modules.immuvis import MultiplexAutoencoder

    tokenizer = _load_tokenizer(config)
    num_channels = len(tokenizer)
    input_size = tuple(config.get("input_image_size", [128, 128]))

    encoder_cfg = _normalize_encoder_config(config["encoder"])
    decoder_cfg = _normalize_decoder_config(config["decoder"])
    uncertainty_method = config.get("uncertainty_method", "beta_nll")
    if uncertainty_method == "evidential":
        decoder_cfg["num_outputs"] = 4

    model = MultiplexAutoencoder(
        num_channels=num_channels,
        encoder_config=encoder_cfg,
        decoder_config=decoder_cfg,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    def extract_fn(x, channel_ids):
        enc_out = model.encode(x, channel_ids)
        latent = enc_out["output"]
        if latent.ndim == 4:
            return latent.mean(dim=(2, 3))
        elif latent.ndim == 3:
            return latent.mean(dim=1)
        return latent

    return extract_fn, input_size


MODEL_LOADERS = {
    "immuvis": load_immuvis,
    "kronos_dino": load_kronos_dino,
    "kronos_dinov2v3": load_kronos_dinov2v3,
    "kronos_immuvis_v2": load_kronos_immuvis_v2,
    "kronos_immuvis_v3": load_kronos_immuvis_v3,
    "kronos_pretrained": load_kronos_pretrained,
    "dino": load_dino_encoder,
    "mamba": load_masked_model,
    "dino_finetune": load_masked_model,
    "virtual_staining": load_masked_model,
}


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def process_batch(extract_fn, batch_imgs, batch_ids, device, target_size):
    """Process a batch of images. Accepts both numpy arrays and torch tensors."""
    if isinstance(batch_imgs, np.ndarray):
        x = torch.from_numpy(batch_imgs).float().to(device)
    else:
        x = batch_imgs.float().to(device)

    if x.shape[-2:] != target_size:
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

    B, C = x.shape[0], x.shape[1]

    if batch_ids is not None:
        if isinstance(batch_ids, np.ndarray):
            channel_ids = torch.from_numpy(batch_ids).long().to(device)
        else:
            channel_ids = batch_ids.long().to(device)
        if channel_ids.ndim == 1:
            channel_ids = channel_ids.unsqueeze(0).expand(B, -1)
    else:
        channel_ids = torch.arange(C, device=device).unsqueeze(0).expand(B, -1)

    with torch.no_grad():
        embeddings = extract_fn(x, channel_ids)

    return embeddings.cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings from any ImmuVis / KRONOS / DINO model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-type", type=str, required=True,
        choices=list(MODEL_LOADERS.keys()),
        help="Model type to load.",
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to training config YAML (not needed for kronos_pretrained).")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (or hf_hub:... for kronos_pretrained).")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--input_dir", type=str, nargs="+", default=None,
                        help="Directories containing .npy / .npz input files (legacy mode).")
    parser.add_argument("--panel-config", type=str, default=None,
                        help="Path to panel config YAML (e.g. configs/all_panels_config.yaml). "
                             "If not given, auto-detected from --config.")
    parser.add_argument("--split", type=str, default="test",
                        help="Data split to use when loading via panel config (default: test).")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--file-extension", type=str, default=None,
                        help="File extension for data loading (npy or tiff). Auto-detected from config.")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device} | Model type: {args.model_type}")

    config = {}
    if args.config is not None:
        yaml = YAML(typ="safe")
        with open(args.config, "r") as f:
            config = yaml.load(f)

    loader_fn = MODEL_LOADERS[args.model_type]
    try:
        extract_fn, target_size = loader_fn(config, args.checkpoint, device)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    print(f"Target input size: {target_size}")

    panel_config = _resolve_panel_config(config, args.panel_config)

    if panel_config is not None:
        # Panel-config mode: use DatasetFromTIFF + PanelBatchSampler
        _run_panel_config_mode(
            extract_fn, target_size, config, panel_config,
            args, device,
        )
    elif args.input_dir is not None:
        # Legacy mode: raw .npy/.npz files from --input_dir
        _run_legacy_mode(extract_fn, target_size, args, device)
    else:
        print("[ERROR] No data source. Provide --panel-config (or set panel_config in "
              "model config), or provide --input_dir for legacy .npy loading.")
        sys.exit(1)


def _run_panel_config_mode(extract_fn, target_size, config, panel_config, args, device):
    """Load data via DatasetFromTIFF + PanelBatchSampler and generate embeddings."""
    from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler
    from torchvision.transforms import CenterCrop

    tokenizer = _load_tokenizer(config)
    file_extension = args.file_extension or config.get("file_extension", "npy")

    test_transform = CenterCrop(target_size) 

    dataset = DatasetFromTIFF(
        panels_config=panel_config,
        split=args.split,
        marker_tokenizer=tokenizer,
        transform=test_transform,
        use_preprocessing=False,
        use_butterworth_filter=True,
        use_clip_normalization=True,
        file_extension=file_extension,
    )
    print(f"Dataset: {len(dataset)} images from {len(dataset.channel_ids)} panels "
          f"(split={args.split})")

    if len(dataset) == 0:
        print("[WARN] No images found. Check paths and file_extension.")
        return

    sampler = PanelBatchSampler(dataset, batch_size=args.batch_size, shuffle=False)
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config.get("num_workers", 4),
        pin_memory=False,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    all_embeddings = []
    all_paths = []
    all_datasets = []

    for imgs, channel_ids, dataset_names, img_paths in tqdm(dataloader, desc="Batches"):
        emb = process_batch(extract_fn, imgs, channel_ids, device, target_size)
        all_embeddings.append(emb)
        all_paths.extend(img_paths)
        all_datasets.extend(dataset_names)

    if all_embeddings:
        final_embeddings = np.concatenate(all_embeddings, axis=0)
        save_path = os.path.join(args.output_dir, f"embeddings_{args.split}.npz")
        np.savez(
            save_path,
            embeddings=final_embeddings,
            paths=np.array(all_paths, dtype=object),
            datasets=np.array(all_datasets, dtype=object),
        )
        print(f"Done. Saved {final_embeddings.shape[0]} embeddings "
              f"({final_embeddings.shape[1]}d) to {save_path}")
    else:
        print("[WARN] No embeddings generated.")

    gc.collect()


def _run_legacy_mode(extract_fn, target_size, args, device):
    """Legacy mode: load raw .npy/.npz files from --input_dir."""
    files_to_process = []
    for d in args.input_dir:
        files_to_process.extend(glob.glob(os.path.join(d, "*.npy")))
        files_to_process.extend(glob.glob(os.path.join(d, "*.npz")))

    files_to_process = [f for f in files_to_process if not os.path.basename(f).startswith("emb_")]
    print(f"Found {len(files_to_process)} files to process.")
    os.makedirs(args.output_dir, exist_ok=True)

    for file_path in tqdm(files_to_process, desc="Files"):
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        save_path = os.path.join(args.output_dir, f"emb_{base_name}.npy")

        if os.path.exists(save_path):
            continue

        try:
            data_raw = np.load(file_path, allow_pickle=True)

            if isinstance(data_raw, np.ndarray) and data_raw.ndim == 0:
                data = data_raw.item()
            elif isinstance(data_raw, np.lib.npyio.NpzFile):
                data = data_raw
            else:
                data = data_raw

            if isinstance(data, dict) and "patches" in data:
                patches = data["patches"]
                labels = data.get("labels", None)
                channel_ids = data.get("channel_ids", None)
            elif isinstance(data, np.ndarray):
                if data.ndim == 3:
                    patches = data[np.newaxis]
                elif data.ndim == 4:
                    patches = data
                else:
                    continue
                labels = None
                channel_ids = None
            else:
                continue

            if patches.ndim == 4:
                if patches.shape[-1] < patches.shape[-2] and patches.shape[-1] < patches.shape[-3]:
                    patches = patches.transpose(0, 3, 1, 2)

            num_samples = patches.shape[0]
            if num_samples == 0:
                continue

            file_embeddings = []
            for i in range(0, num_samples, args.batch_size):
                batch_imgs = patches[i : i + args.batch_size]
                emb = process_batch(extract_fn, batch_imgs, channel_ids, device, target_size)
                file_embeddings.append(emb)

            if file_embeddings:
                final_embeddings = np.concatenate(file_embeddings, axis=0)
                output_data = {"embeddings": final_embeddings}
                if labels is not None:
                    output_data["labels"] = labels
                np.save(save_path, output_data)

            del patches, file_embeddings
        except Exception as e:
            print(f"Error in {file_name}: {e}")

        gc.collect()

    print(f"Done. Embeddings saved to {args.output_dir}")


if __name__ == "__main__":
    main()
