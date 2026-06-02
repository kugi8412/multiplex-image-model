import argparse
import json
import os
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from multiplex_model.data import MultiplexDataset, TestCrop
from multiplex_model.modules.immuvis import MultiplexAutoencoder
from multiplex_model.utils.configuration import (
    DataConfig,
    DecoderConfig,
    EncoderConfig,
    load_panel_config,
    load_tokenizer_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leave-one-out validation (mask one channel at a time)."
    )
    parser.add_argument(
        "--versions",
        type=int,
        nargs="+",
        default=list(range(0, 19)),
        help="Model versions to evaluate (e.g. --versions 14 15).",
    )
    parser.add_argument(
        "--checkpoint-glob",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional glob pattern(s) for checkpoints inside --models-path. "
            "If provided, --versions is ignored."
        ),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of test images to evaluate per model (default: all).",
    )
    parser.add_argument(
        "--models-path",
        default="/raid_encrypted/immucan/models",
        help="Path to model checkpoints and configs.",
    )
    parser.add_argument(
        "--results-dir",
        default="/raid_encrypted/immucan/results/with_reconstructs",
        help="Where to save CSV outputs.",
    )
    parser.add_argument(
        "--recon-dir",
        default="/storage_ssd_2/immuvis/recons",
        help="Where to save leave-one-out reconstructions (npz).",
    )
    parser.add_argument(
        "--save-reconstructions",
        action="store_true",
        help="Save leave-one-out reconstructions to NPZ files.",
    )
    parser.add_argument(
        "--panel-config",
        default="/home/duchal/ImmuVis_2/multiplex-image-model/configs/all_panels_config.yaml",
    )
    parser.add_argument(
        "--tokenizer-config",
        default="/home/duchal/ImmuVis_2/multiplex-image-model/configs/all_markers_tokenizer.yaml",
    )
    parser.add_argument(
        "--data-config",
        default=None,
        help=(
            "Optional YAML file with DataConfig fields for input preprocessing/scaling. "
            "Overrides values inferred from training config."
        ),
    )
    parser.add_argument(
        "--preprocessing-func",
        choices=["arcsinh", "log1p", "none"],
        default=None,
        help="Override preprocessing function.",
    )
    parser.add_argument(
        "--denoising-func",
        choices=["median", "gaussian", "butterworth", "none"],
        default=None,
        help="Override denoising function.",
    )
    parser.add_argument(
        "--scaling-func",
        choices=["minmax", "percentile", "global_clip", "none"],
        default=None,
        help="Override scaling function.",
    )
    parser.add_argument(
        "--normalization-func",
        choices=["zscore_ds", "none"],
        default=None,
        help="Override normalization function.",
    )
    parser.add_argument(
        "--file-extension",
        choices=["tiff", "npy"],
        default=None,
        help="Override input file extension.",
    )
    parser.add_argument(
        "--global-scaling-bound",
        type=float,
        default=None,
        help="Override global scaling bound for global_clip scaling.",
    )
    parser.add_argument(
        "--operation-order",
        nargs="+",
        default=None,
        help="Override preprocessing operation order.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset names to evaluate (e.g. --datasets hn).",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=128,
        help="Test crop size.",
    )
    parser.add_argument(
        "--mu-activation",
        choices=["sigmoid", "hardsigmoid", "none"],
        default="sigmoid",
        help="Activation applied to predicted mu before metric computation.",
    )
    parser.add_argument(
        "--strict-loading",
        action="store_true",
        help="Use strict=True when loading checkpoint state_dict.",
    )
    return parser.parse_args()


def apply_mu_activation(mu: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "sigmoid":
        return torch.sigmoid(mu)
    if activation == "hardsigmoid":
        return F.hardsigmoid(mu)
    if activation == "none":
        return mu
    raise ValueError(f"Unsupported mu activation: {activation}")


def _load_yaml(path: str) -> dict[str, Any]:
    yaml = YAML(typ="safe")
    with open(path, "r") as file:
        config = yaml.load(file)
    if not isinstance(config, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return config


def _resolve_dataset_setup(
    raw_model_config: dict[str, Any],
    panel_config_path: str,
    tokenizer_config_path: str,
    data_config_path: str | None,
    data_overrides: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int], dict[str, Any]]:
    panel_config_dict = load_panel_config(panel_config_path)
    tokenizer = load_tokenizer_config(tokenizer_config_path)
    data_config = {}

    if data_config_path:
        external_data_cfg = _load_yaml(data_config_path)
        validated_external_data_cfg = DataConfig(**external_data_cfg).model_dump()
        data_config.update(validated_external_data_cfg)

    # Final precedence: explicit CLI overrides.
    cli_data_overrides = {
        key: value for key, value in data_overrides.items() if value is not None
    }
    if cli_data_overrides:
        data_config.update(cli_data_overrides)

    # Validate merged data config for early and clear failures.
    data_config = DataConfig(**data_config).model_dump()

    return panel_config_dict, tokenizer, data_config


def _normalize_optional_name(value: str | None) -> str | None:
    if value is None:
        return None
    if value == "none":
        return None
    return value


def _extract_mu_logvar(output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_outputs = output.shape[-1]
    if num_outputs < 1:
        raise ValueError(f"Model output has invalid last dim: {num_outputs}")

    mu = output[..., 0]
    if num_outputs >= 2:
        logvar = output[..., 1]
    else:
        logvar = torch.full_like(mu, float("nan"))

    if mu.ndim == 4 and mu.shape[1] == 1:
        mu = mu.squeeze(1)
    if logvar.ndim == 4 and logvar.shape[1] == 1:
        logvar = logvar.squeeze(1)

    return mu, logvar


def create_leave_one_out_batch(
    img: torch.Tensor, channel_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create leave-one-out batch for a single image.

    Args:
        img: [C, H, W]
        channel_ids: [C]

    Returns:
        masked_img: [C, C-1, H, W]
        active_channel_ids: [C, C-1]
        output_channel_ids: [C, 1]
        masked_indices: [C]
    """
    num_channels, height, width = img.shape
    keep_mask = ~torch.eye(num_channels, dtype=torch.bool, device=img.device)

    img_expand = img.unsqueeze(0).expand(num_channels, -1, -1, -1)
    masked_img = img_expand[keep_mask].view(num_channels, num_channels - 1, height, width)

    channel_ids_expand = channel_ids.unsqueeze(0).expand(num_channels, -1)
    active_channel_ids = channel_ids_expand[keep_mask].view(num_channels, num_channels - 1)

    output_channel_ids = channel_ids.view(num_channels, 1)
    masked_indices = torch.arange(num_channels, device=img.device)

    return masked_img, active_channel_ids, output_channel_ids, masked_indices


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data_overrides = {
        "preprocessing_func": _normalize_optional_name(args.preprocessing_func),
        "denoising_func": _normalize_optional_name(args.denoising_func),
        "scaling_func": _normalize_optional_name(args.scaling_func),
        "normalization_func": _normalize_optional_name(args.normalization_func),
        "file_extension": args.file_extension,
        "global_scaling_bound": args.global_scaling_bound,
        "operation_order": args.operation_order,
    }

    patterns = args.checkpoint_glob or [f"Immu*-6{v:02d}-*.pth" for v in args.versions]
    model_files: list[str] = []
    for pattern in patterns:
        model_files.extend(glob(f"{args.models_path}/{pattern}"))

    if not model_files:
        raise FileNotFoundError(
            f"No model checkpoints found in {args.models_path} for versions {args.versions}."
        )

    os.makedirs(args.results_dir, exist_ok=True)

    for model_name in sorted(model_files):
        model_checkpoint = os.path.basename(model_name)
        model_idx = model_checkpoint.split("-")[1]
        config_name = f"config.{model_checkpoint.split('.')[0]}.yaml"
        model_config_path = f"{args.models_path}/{config_name}"
        if not os.path.exists(model_config_path):
            print(f"Skipping {model_checkpoint}: model config not found at {model_config_path}")
            continue

        model_config_dict = _load_yaml(model_config_path)
        panel_config_dict, tokenizer, data_config = _resolve_dataset_setup(
            raw_model_config=model_config_dict,
            panel_config_path=args.panel_config,
            tokenizer_config_path=args.tokenizer_config,
            data_config_path=args.data_config,
            data_overrides=data_overrides,
        )

        if args.datasets is not None:
            panel_config_dict = panel_config_dict.copy()
            panel_config_dict["datasets"] = args.datasets

        inv_tokenizer = {v: k for k, v in tokenizer.items()}
        num_channels = len(tokenizer)

        print(f"Number of channels: {num_channels}")
        print(f"Sample markers: {list(tokenizer.keys())[:5]}")
        print(
            "Data preprocessing config: "
            f"pre={data_config['preprocessing_func']}, denoise={data_config['denoising_func']}, "
            f"scale={data_config['scaling_func']}, norm={data_config['normalization_func']}, "
            f"ext={data_config['file_extension']}"
        )

        test_transform = TestCrop(args.crop_size)
        test_dataset = MultiplexDataset(
            panels_config=panel_config_dict,
            split=args.split,
            marker_tokenizer=tokenizer,
            transform=test_transform,
            **data_config,
        )
        print(f"Test dataset size: {len(test_dataset)} images")
        dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        encoder_config = EncoderConfig(**model_config_dict["encoder"])
        decoder_config = DecoderConfig(**model_config_dict["decoder"])
        fallback_model_config = {
            "num_channels": num_channels,
            "encoder_config": encoder_config.model_dump(),
            "decoder_config": decoder_config.model_dump(),
        }

        print(f"Loading model weights from: {model_name}")
        model = MultiplexAutoencoder.load_from_checkpoint(
            checkpoint=model_name,
            map_location="cpu",
            model_config=fallback_model_config,
            strict=args.strict_loading,
        ).to(device)
        model.eval()

        all_mse = []
        all_uncertainties = []
        all_pearson_r = []
        all_channel_ids = []
        all_dataset_names = []
        all_image_paths = []

        recon_dir = Path(args.recon_dir) / f"immuvis_{model_idx}_loo"
        if args.save_reconstructions:
            recon_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for img_idx, (img, channel_ids, ds_name, img_path) in enumerate(
                tqdm(dataloader, desc="Leave-one-out validation")
            ):
                if args.max_images is not None and img_idx >= args.max_images:
                    break

                img = img.squeeze(0).to(device, dtype=torch.float32)
                channel_ids = channel_ids.squeeze(0).to(device, dtype=torch.long)

                (
                    masked_img,
                    active_channel_ids,
                    output_channel_ids,
                    masked_indices,
                ) = create_leave_one_out_batch(img=img, channel_ids=channel_ids)

                output = model(masked_img, active_channel_ids, output_channel_ids)[
                    "output"
                ]

                mi, logvar = _extract_mu_logvar(output)
                mi = apply_mu_activation(mi, args.mu_activation)

                target_channels = img[masked_indices]
                mse = (mi - target_channels).pow(2).mean(dim=(1, 2))

                mi_mean = mi.mean(dim=(1, 2), keepdim=True)
                target_mean = target_channels.mean(dim=(1, 2), keepdim=True)
                pearson_r = ((mi - mi_mean) * (target_channels - target_mean)).mean(
                    dim=(1, 2)
                ) / (mi.std(dim=(1, 2)) * target_channels.std(dim=(1, 2)) + 1e-8)

                all_mse.append(mse.flatten().cpu().numpy())
                all_uncertainties.append(logvar.mean(dim=(1, 2)).flatten().cpu().numpy())
                all_pearson_r.append(pearson_r.flatten().cpu().numpy())
                all_channel_ids.append(channel_ids[masked_indices].flatten().cpu().numpy())

                num_observations = masked_indices.numel()
                all_dataset_names.extend([ds_name[0]] * num_observations)
                all_image_paths.extend([img_path[0]] * num_observations)

                if args.save_reconstructions:
                    masked_channel_ids = channel_ids.detach().cpu().numpy()
                    masked_marker_names = [
                        inv_tokenizer.get(int(cid), "Unknown")
                        for cid in masked_channel_ids.tolist()
                    ]
                    metadata = {
                        "image_index": int(img_idx),
                        "image_path": str(img_path[0]),
                        "dataset_name": str(ds_name[0]),
                        "masked_strategy": "leave_one_out",
                        "num_channels": int(masked_channel_ids.shape[0]),
                        "mu_activation": args.mu_activation,
                    }
                    out_path = recon_dir / f"recn-{img_idx:05d}.npz"
                    np.savez_compressed(
                        out_path,
                        recon=mi.detach().cpu().numpy(),
                        variance=torch.exp(logvar).detach().cpu().numpy(),
                        target=img.detach().cpu().numpy(),
                        channel_ids=masked_channel_ids,
                        marker_names=np.array(masked_marker_names),
                        masked_channel_ids=masked_channel_ids,
                        masked_marker_names=np.array(masked_marker_names),
                        metadata=np.array(json.dumps(metadata)),
                    )

        mses = np.concatenate(all_mse, axis=0)
        uncertainties = np.concatenate(all_uncertainties, axis=0)
        pearson_rs = np.concatenate(all_pearson_r, axis=0)
        masked_channel_ids = np.concatenate(all_channel_ids, axis=0)

        df = pd.DataFrame(
            {
                "mse": mses,
                "logsigma": uncertainties,
                "pearson": pearson_rs,
                "Channel_ID": masked_channel_ids.astype(np.int64),
            }
        )
        df["marker"] = df["Channel_ID"].map(lambda x: inv_tokenizer.get(int(x), "Unknown"))
        df["masked"] = "leave_one_out"
        df["masked_count"] = 1
        df["model"] = f"ImmuVis-{model_idx}"
        df["dataset_name"] = all_dataset_names
        df["image_path"] = all_image_paths
        df["mu_activation"] = args.mu_activation

        output_file = os.path.join(args.results_dir, f"immuvis_{model_idx}_loo.csv")
        print(f"Saving results to: {output_file}")
        df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
