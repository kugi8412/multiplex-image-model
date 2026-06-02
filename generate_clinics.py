#!/usr/bin/env python3
# generate_clinical_universal.py
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from torchvision.transforms.functional import center_crop

from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler

sys.path.insert(0, os.getcwd())
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

# ============================================================================
# 1. FUNKCJE POMOCNICZE (Wycinki i zapis)
# ============================================================================

def get_all_patches(img, patch_size=128):
    if img.ndim == 3: img = img.unsqueeze(0)
    
    H, W = img.shape[2:]
    patches = []
    coords = []
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            if i + patch_size > H or j + patch_size > W:
                continue

            patch = img[:, :, i : i + patch_size, j : j + patch_size]
            patches.append(patch)
            coords.append([(i, j), (i + patch_size, j + patch_size)])

    return patches, coords

def save_batch(embeddings, metadata, outpath, split_name, model_name, batch_idx):
    if not embeddings:
        return None
    
    emb_arr = np.stack(embeddings)
    save_base = f"ImmuVis-{model_name}_{split_name}_patches"
    
    npy_path = os.path.join(outpath, f'{save_base}_embeddings_batch_{batch_idx}.npy')
    csv_path = os.path.join(outpath, f'{save_base}_metadata_batch_{batch_idx}.csv')
    
    np.save(npy_path, emb_arr)
    
    df = pd.DataFrame(metadata, columns=['img_path', 'panel', 'coords0', 'coords1', 'augmentation'])
    df.to_csv(csv_path, index=False)


# ============================================================================
# 2. LOADERY MODELI (Z zaadaptowanego skryptu uniwersalnego)
# ============================================================================

def _load_tokenizer(config):
    tokenizer_path = config.get("tokenizer_config", config.get("tokenizer_config_path", "configs/all_markers_tokenizer.yaml"))
    if os.path.exists(tokenizer_path): return YAML().load(open(tokenizer_path))
    elif os.path.exists("configs/all_markers_tokenizer.yaml"): return YAML().load(open("configs/all_markers_tokenizer.yaml"))
    else: raise FileNotFoundError("Tokenizer not found.")

def _normalize_encoder_config(raw_encoder):
    enc = dict(raw_encoder)
    for k in ["ma_layers_blocks", "ma_embedding_dims", "pm_layers_blocks", "pm_embedding_dims"]:
        enc.setdefault(k, [])
    enc.setdefault("use_latent_norm", True)
    enc.setdefault("encoder_type", "convnext")
    if "hyperkernel" in enc and "hyperkernel_config" not in enc:
        enc["hyperkernel_config"] = enc.pop("hyperkernel")
    return enc

def _normalize_decoder_config(raw_decoder):
    dec = dict(raw_decoder)
    dec.setdefault("num_outputs", 2)
    dec.setdefault("block_type", "convnext")
    if "hyperkernel" in dec and "hyperkernel_config" not in dec:
        dec["hyperkernel_config"] = dec.pop("hyperkernel")
    return dec

def load_masked_model(config, checkpoint_path, device):
    """Loader dla ViT Baseline (virtual_staining / dino_finetune)"""
    from multiplex_model.modules.immuvis import MultiplexAutoencoder

    num_channels = len(_load_tokenizer(config))
    encoder_cfg = _normalize_encoder_config(config["encoder"])
    decoder_cfg = _normalize_decoder_config(config["decoder"])
    if config.get("uncertainty_method", "beta_nll") == "evidential":
        decoder_cfg["num_outputs"] = 4

    model = MultiplexAutoencoder(num_channels=num_channels, encoder_config=encoder_cfg, decoder_config=decoder_cfg)
    
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    def extract_fn(x, channel_ids):
        enc_out = model.encode(x, channel_ids)
        latent = enc_out["output"]
        if latent.ndim == 4: return latent.mean(dim=(2, 3))
        elif latent.ndim == 3: return latent.mean(dim=1)
        return latent

    return extract_fn

def load_immukronos_finetuned(config, checkpoint_path, device, kronos_config_path=None, kronos_version="v2"):
    """Loader dla ImmunoKronosa"""
    from finetune_immukronos_decoder import ImmukronosAutoencoder, build_decoder, load_kronos_checkpoint

    yaml_loader = YAML(typ="safe")
    with open(kronos_config_path) as f:
        kronos_config = yaml_loader.load(f)

    kronos_model, num_markers, _ = load_kronos_checkpoint(kronos_config, checkpoint_path, device, version=kronos_version)
    embed_dim = kronos_config.get("embed_dim", 768)
    patch_size = kronos_config.get("patch_size", 8)

    dec_cfg = dict(config.get("decoder", config))
    dec_cfg.setdefault("num_outputs", 2)
    if "hyperkernel" in dec_cfg and "hyperkernel_config" not in dec_cfg:
        dec_cfg["hyperkernel_config"] = dec_cfg.pop("hyperkernel")

    decoder = build_decoder(dec_cfg, num_channels=num_markers, embed_dim=embed_dim)
    model = ImmukronosAutoencoder(
        kronos_model=kronos_model, decoder=decoder, num_channels=num_markers, 
        patch_size=patch_size, embed_dim=embed_dim, version=kronos_version, freeze_encoder=True
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    def extract_fn(x, channel_ids):
        enc_out = model.encode(x, channel_ids)
        latent = enc_out["output"]
        if latent.ndim == 4: return latent.mean(dim=(2, 3))
        elif latent.ndim == 3: return latent.mean(dim=1)
        return latent

    return extract_fn


def load_kronos_immuvis_v2(config, checkpoint_path, device):
    """Loader dla bazowego enkodera ImmunoKronos V2"""
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
        # Zwracamy [CLS] token reprezentujący cały wycinek
        return x_enc[:, 0]

    return extract_fn

MODEL_LOADERS = {
    "virtual_staining": load_masked_model,
    "immukronos_finetuned": load_immukronos_finetuned,
    "kronos_immuvis_v2": load_kronos_immuvis_v2,
}


# ============================================================================
# 3. GŁÓWNA PĘTLA PRZETWARZANIA (TTA + Embeddingi)
# ============================================================================

def process_dataset(extract_fn, dataloader, device, patch_size, outpath, split_name, model_name, target_subsets):
    BATCH_SAVE_FREQ = 500
    embeddings = []
    metadata = []    
    batch_idx = 0
    images_processed_count = 0

    aug_labels_template = [
        'orig', 'rot90', 'rot180', 'rot270', 
        'flip', 'flip_rot90', 'flip_rot180', 'flip_rot270'
    ]

    for img, channel_ids, panel_idx, img_path_tuple in tqdm(dataloader, desc=split_name):
        img_path = img_path_tuple[0]
        
        # Filtrowanie datasetów po nazwie w ścieżce
        if not any(subset in img_path for subset in target_subsets):
            continue

        _, _, H, W = img.shape
        if H < patch_size or W < patch_size:
            continue

        channel_ids = channel_ids.to(device)
        patches, coords = get_all_patches(img, patch_size)
        
        if len(patches) == 0:
            continue

        inference_batch_size = 8 # 8 patchy na raz (po 8 augmentacji każdy = 64 wejściowe obrazy)
        
        for i in range(0, len(patches), inference_batch_size):
            batch_patches = patches[i:i+inference_batch_size]
            batch_coords = coords[i:i+inference_batch_size]
            
            aug_patches = []
            aug_coords = []
            aug_types = []

            for patch, coord in zip(batch_patches, batch_coords):
                p0 = patch
                p90 = torch.rot90(patch, 1, [2, 3])
                p180 = torch.rot90(patch, 2, [2, 3])
                p270 = torch.rot90(patch, 3, [2, 3])
                pf0 = torch.flip(p0, [3])
                pf90 = torch.flip(p90, [3])
                pf180 = torch.flip(p180, [3])
                pf270 = torch.flip(p270, [3])

                aug_patches.extend([p0, p90, p180, p270, pf0, pf90, pf180, pf270])
                aug_coords.extend([coord] * 8)
                aug_types.extend(aug_labels_template)
            
            x = torch.cat(aug_patches, dim=0).float().to(device)
            curr_B = x.shape[0]
            batch_c_ids = channel_ids.repeat(curr_B, 1)

            with torch.no_grad():
                # ZAMIENIONO: Wykorzystanie uniwersalnego extract_fn zamiast model.encode
                latent = extract_fn(x, batch_c_ids)
                latent_np = latent.cpu().numpy().astype(np.float16)
                
            for k in range(curr_B):
                embeddings.append(latent_np[k])
                c0, c1 = aug_coords[k]
                c0_str = f"({c0[0]}, {c0[1]})"
                c1_str = f"({c1[0]}, {c1[1]})"
                aug_name = aug_types[k]            
                
                if isinstance(panel_idx, torch.Tensor):
                    p_idx = panel_idx[0].item() if panel_idx.ndim > 0 else panel_idx.item()
                elif isinstance(panel_idx, (list, tuple)):
                    p_idx = panel_idx[0]
                else:
                    p_idx = panel_idx

                metadata.append((img_path, p_idx, c0_str, c1_str, aug_name))

        images_processed_count += 1

        if images_processed_count % BATCH_SAVE_FREQ == 0:
            save_batch(embeddings, metadata, outpath, split_name, model_name, batch_idx)
            batch_idx += 1
            embeddings = []
            metadata = []

    if len(embeddings) > 0:
        save_batch(embeddings, metadata, outpath, split_name, model_name, batch_idx)


# ============================================================================
# 4. START PROGRAMU
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Clinical embedding extractor with TTA.")
    parser.add_argument('--model-type', type=str, required=True, choices=list(MODEL_LOADERS.keys()))
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True, help="Short name for output files (e.g., ViT-Base or KRONOS-V2)")
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--panel_config', type=str, required=True)
    
    # Argumenty dla ImmunoKronosa
    parser.add_argument("--kronos-config", type=str, default=None, help="Wymagane dla model-type: immukronos_finetuned")
    parser.add_argument("--kronos-version", type=str, default="v2", choices=["v2", "v3"])
    
    # Subkohorty do przefiltrowania
    parser.add_argument('--target-subsets', nargs='+', default=['danenberg', 'cords'], help="List of cohort names to match in paths.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Model Type: {args.model_type}")

    model_output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # 1. Załadowanie konfiguracji głównej
    yaml = YAML(typ="safe")
    with open(args.config, "r") as f:
        config = yaml.load(f)

    # 2. Załadowanie Extract Function na podstawie typu modelu
    # 2. Załadowanie Extract Function na podstawie typu modelu
    if args.model_type == "immukronos_finetuned":
        if not args.kronos_config:
            raise ValueError("Flaga --kronos-config jest WYMAGANA dla immukronos_finetuned!")
        extract_fn = load_immukronos_finetuned(config, args.checkpoint, device, args.kronos_config, args.kronos_version)
    elif args.model_type == "kronos_immuvis_v2":
        extract_fn = load_kronos_immuvis_v2(config, args.checkpoint, device)
    else:
        extract_fn = load_masked_model(config, args.checkpoint, device)

    # 3. Przygotowanie Dataloaderów
    tokenizer_path = config.get("tokenizer_config_path", "configs/all_markers_tokenizer.yaml")
    tokenizer = YAML().load(open(tokenizer_path))
    panel_config = YAML().load(open(args.panel_config))

    # Test
    test_dataset = DatasetFromTIFF(
        panels_config=panel_config, split='test', marker_tokenizer=tokenizer,
        use_median_denoising=False, use_butterworth_filter=True,
        use_minmax_normalization=False, use_global_clip_limits=False, use_clip_normalization=True,
        file_extension="npy"  # <--- DODANO TO
    )
    test_loader = DataLoader(test_dataset, batch_sampler=PanelBatchSampler(test_dataset, 1, shuffle=False), num_workers=8)
    # Train
    train_dataset = DatasetFromTIFF(
        panels_config=panel_config, split='train', marker_tokenizer=tokenizer,
        use_median_denoising=False, use_butterworth_filter=True,
        use_minmax_normalization=False, use_global_clip_limits=False, use_clip_normalization=True,
        file_extension="npy"  # <--- DODANO TO
    )
    train_loader = DataLoader(train_dataset, batch_sampler=PanelBatchSampler(train_dataset, 1, shuffle=False), num_workers=8)

    # 4. Uruchomienie ekstrakcji
    print(f"Rozpoczynam ekstrakcję dla kohort: {args.target_subsets}")
    process_dataset(extract_fn, test_loader, device, 128, model_output_dir, 'test', args.model_name, args.target_subsets)
    process_dataset(extract_fn, train_loader, device, 128, model_output_dir, 'train', args.model_name, args.target_subsets)

    print("Zakończono generowanie embeddingów klinicznych z TTA!")

if __name__ == '__main__':
    main()
