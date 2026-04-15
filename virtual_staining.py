#!/usr/bin/env python
# -*- coding: utf-8 -*-
# virtual_staining.py


"""
Leave-one-out Pearson correlation evaluation.

For each image in the test set and for each non-structural marker, removes
that marker from the input, runs reconstruction through the autoencoder,
and computes the Pearson correlation between the predicted (virtual stain)
and the original marker channel.

Usage::

    python virtual_staining.py configs/train_vit_config.yaml \\
        --checkpoint /raid_encrypted/immucan/models/ViTM200.pth \\
        --output immv_recons_ViTM.csv
"""


import gc
import os
import random
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ruamel.yaml import YAML
from scipy.stats import pearsonr
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler, TestCrop
from multiplex_model.losses import get_output_activation, LearnableOutputActivation
from multiplex_model.modules import MultiplexAutoencoder
from multiplex_model.utils.configuration import TrainingConfig

# Reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)


def main():
    parser = argparse.ArgumentParser(description="Virtual staining rebuttal evaluation")
    parser.add_argument("config", help="Training config YAML (e.g. configs/train_vit_config.yaml)")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output", default="immv_recons_ViTM.csv", help="Output CSV path")
    parser.add_argument("--save-recons", action="store_true", help="Save per-marker .npy reconstructions")
    parser.add_argument("--recons-dir", default="recons", help="Directory for .npy outputs")
    parser.add_argument("--skip-markers", nargs="*", default=["DNA1", "DNA2"],
                        help="Structural markers to skip (default: DNA1 DNA2)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # Load config
    yaml_loader = YAML(typ="safe")
    with open(args.config) as f:
        raw_config = yaml_loader.load(f)
    config = TrainingConfig(**raw_config)

    device = args.device or config.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"Using device: {device}")

    PANEL_CONFIG = YAML().load(open(config.panel_config))
    TOKENIZER = YAML().load(open(config.tokenizer_config))
    INV_TOKENIZER = {v: k for k, v in TOKENIZER.items()}
    num_channels = len(TOKENIZER)

    # Dataset
    SIZE = config.input_image_size
    test_transform = TestCrop(SIZE[0])

    test_dataset = DatasetFromTIFF(
        panels_config=PANEL_CONFIG,
        split="test",
        marker_tokenizer=TOKENIZER,
        transform=test_transform,
        use_preprocessing=False,
        use_butterworth_filter=True,
        use_clip_normalization=True,
    )

    test_batch_sampler = PanelBatchSampler(test_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(
        test_dataset, batch_sampler=test_batch_sampler,
        num_workers=config.num_workers, pin_memory=True,
    )

    # Model
    decoder_cfg = config.decoder_config.model_dump()
    decoder_cfg["num_outputs"] = 4 if config.uncertainty_method == "evidential" else 2

    model = MultiplexAutoencoder(
        num_channels=num_channels,
        encoder_config=config.encoder_config.model_dump(),
        decoder_config=decoder_cfg,
    ).to(device)

    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(
        state_dict.get("model_state_dict", state_dict), strict=False,
    )
    model.eval()
    print(f"Loaded model from {args.checkpoint}")

    # Activation
    if config.learnable_activation_params and "activation_state_dict" in state_dict:
        activation_fn = LearnableOutputActivation(
            name=config.output_activation,
            beta=config.activation_beta,
            window=config.activation_window,
            learnable=True,
        ).to(device)
        activation_fn.load_state_dict(state_dict["activation_state_dict"])
        activation_fn.eval()
        print(f"Loaded learnable activation: {activation_fn}")
    else:
        activation_fn = get_output_activation(
            config.output_activation,
            beta=config.activation_beta,
            window=getattr(config, "activation_window", 0.5),
        )

    # Evaluation loop
    skip_set = set(args.skip_markers)
    ids, markers_list, pearson_list = [], [], []

    if args.save_recons:
        os.makedirs(args.recons_dir, exist_ok=True)

    for _, (img, channel_ids, dataset_name, img_path) in enumerate(tqdm(test_dataloader)):
        clip_limit = PANEL_CONFIG["clip_limits"].get(dataset_name[0], 5.0)
        if isinstance(clip_limit, (list, tuple)):
            clip_limit = clip_limit[0]

        img_denorm = float(clip_limit) * img

        _, num_channels_img, H, W = img.shape
        img = img.to(device, dtype=torch.float32)
        channel_ids = channel_ids.to(device, dtype=torch.long)

        # Iterate over each marker channel in this image
        for c in range(num_channels_img):
            marker_token = channel_ids[0, c].item()
            marker_name = INV_TOKENIZER.get(marker_token, f"ch{marker_token}")

            if marker_name in skip_set:
                continue

            # Remove channel c from input (leave-one-out)
            keep_mask = torch.ones(num_channels_img, dtype=torch.bool)
            keep_mask[c] = False
            masked_img = img[:, keep_mask]                      # (1, C-1, H, W)
            active_channel_ids = channel_ids[:, keep_mask]      # (1, C-1)

            with torch.no_grad(), autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                output = model(masked_img, active_channel_ids, channel_ids)["output"]

                if config.uncertainty_method == "evidential":
                    gamma_raw, nu_raw, alpha_raw, beta_raw = output.unbind(dim=-1)
                    mi = activation_fn(gamma_raw)
                    nu = F.softplus(nu_raw) + 1e-6
                    alpha = F.softplus(alpha_raw) + 1.0 + 1e-6
                    beta_param = F.softplus(beta_raw) + 1e-6
                    alpha_m1 = alpha - 1.0 + 1e-8
                    uncertainty = beta_param * (1.0 + nu) / (nu * alpha_m1)
                else:
                    mi_raw, logvar = output.unbind(dim=-1)
                    mi = activation_fn(mi_raw)
                    uncertainty = torch.exp(logvar)

            mi_np = mi.float().cpu().numpy() * clip_limit
            unc_np = uncertainty.float().cpu().numpy()
            gt_np = img_denorm[0, c].cpu().numpy()
            r = pearsonr(mi_np[0, c].flatten(), gt_np.flatten()).statistic
            pearson_list.append(r)
            markers_list.append(marker_name)

            img_id = os.path.basename(img_path[0]).split(".")[0]
            ids.append(img_id)

            if args.save_recons:
                np.save(f"{args.recons_dir}/{img_id}_{marker_name}_recon.npy", mi_np[0, c])
                np.save(f"{args.recons_dir}/{img_id}_{marker_name}_orig.npy", gt_np)
                np.save(f"{args.recons_dir}/{img_id}_{marker_name}_uncertainty.npy", unc_np[0, c])

        torch.cuda.empty_cache()
        gc.collect()

    df = pd.DataFrame({
        "id": ids,
        "marker": markers_list,
        "pearson": pearson_list,
    })
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")

    # Summary statistics
    summary = df.groupby("marker")["pearson"].agg(["mean", "std", "count"])
    summary = summary.sort_values("mean", ascending=False)
    print("\nPer-marker Pearson (mean ± std):")
    for marker, row in summary.iterrows():
        print(f"  {marker:>25s}: {row['mean']:.4f} ± {row['std']:.4f}  (n={int(row['count'])})")
    print(f"\nOverall mean Pearson: {df['pearson'].mean():.4f}")


if __name__ == "__main__":
    main()
