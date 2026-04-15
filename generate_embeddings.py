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


def load_immuvis(config, checkpoint_path, device):
    from multiplex_model.modules.immuvis import MultiplexAutoencoder

    tokenizer = _load_tokenizer(config)
    num_channels = len(tokenizer)
    input_size = tuple(config.get("input_image_size", [128, 128]))

    model = MultiplexAutoencoder(
        num_channels=num_channels,
        encoder_config=config["encoder"],
        decoder_config=config["decoder"],
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
    patch_size = config.get("patch_size", 16)
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
    patch_size = config.get("patch_size", 16)
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
    patch_size = config.get("patch_size", 16)
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
    patch_size = config.get("patch_size", 16)
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

    model, _, embed_dim = create_model_from_pretrained(
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

    model = MultiplexAutoencoder(
        num_channels=num_channels,
        encoder_config=config["encoder"],
        decoder_config=config["decoder"],
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
}


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def process_batch(extract_fn, batch_imgs, batch_ids, device, target_size):
    x = torch.from_numpy(batch_imgs).float().to(device)
    if x.shape[-2:] != target_size:
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

    B, C = x.shape[0], x.shape[1]

    if batch_ids is not None:
        channel_ids = torch.from_numpy(batch_ids).long().to(device)
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
    parser.add_argument("--input_dir", type=str, nargs="+", required=True,
                        help="Directories containing .npy / .npz input files.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
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
