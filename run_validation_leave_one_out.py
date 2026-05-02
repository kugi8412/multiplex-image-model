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

from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler, TestCrop
from multiplex_model.modules.immuvis import MultiplexAutoencoder
from ruamel.yaml import YAML


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leave-one-out validation (mask one channel at a time)."
    )

    # ---- single-model mode (recommended for Exp 5a/5b, 7a/7b/7c) ----
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to training config YAML (e.g. configs/exp5a_finetune_dinov2.yaml). "
            "When provided together with --checkpoint, evaluates that single model "
            "and ignores --versions / --checkpoint-glob / --models-path."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a single model checkpoint (.pth). "
            "Must be used together with --config."
        ),
    )

    # ---- batch-discovery mode (legacy) ----
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
        default="configs/all_panels_config.yaml",
    )
    parser.add_argument(
        "--tokenizer-config",
        default="configs/all_markers_tokenizer.yaml",
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
    parser.add_argument(
        "--model-label",
        type=str,
        default=None,
        help=(
            "Human-readable label for this model in the output CSV 'model' column. "
            "Defaults to checkpoint filename stem."
        ),
    )
    args = parser.parse_args()

    # Validate single-model mode args
    if (args.config is None) != (args.checkpoint is None):
        parser.error("--config and --checkpoint must be specified together.")

    return args


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
    
    yaml = YAML(typ="safe")
    
    # Load Panel Config
    with open(panel_config_path, "r") as f:
        panel_config_dict = yaml.load(f)
        
    # Load Tokenizer Config
    with open(tokenizer_config_path, "r") as f:
        tokenizer = yaml.load(f)

    # Simplified data config handling
    data_config = {}
    if data_config_path:
        with open(data_config_path, "r") as f:
             data_config.update(yaml.load(f))

    # Apply overrides
    cli_data_overrides = {
        key: value for key, value in data_overrides.items() if value is not None
    }
    data_config.update(cli_data_overrides)

    # Ensure required defaults (matching your ImmuVis pipeline)
    data_config.setdefault("file_extension", "npy")

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


def _build_model(
    model_config_dict: dict[str, Any],
    num_channels: int,
    strict: bool,
    checkpoint_path: str,
    device: str,
) -> MultiplexAutoencoder:
    """Instantiate MultiplexAutoencoder from a training config and checkpoint."""
    
    # Normalize raw YAML dicts to the format MultiplexAutoencoder expects.
    # Pydantic TrainingConfig uses aliases (hyperkernel → hyperkernel_config)
    # and provides defaults for empty lists; replicate that here.
    encoder_config = dict(model_config_dict["encoder"])
    encoder_config.setdefault("ma_layers_blocks", [])
    encoder_config.setdefault("ma_embedding_dims", [])
    encoder_config.setdefault("pm_layers_blocks", [])
    encoder_config.setdefault("pm_embedding_dims", [])
    encoder_config.setdefault("use_latent_norm", True)
    encoder_config.setdefault("encoder_type", "convnext")
    if "hyperkernel" in encoder_config and "hyperkernel_config" not in encoder_config:
        encoder_config["hyperkernel_config"] = encoder_config.pop("hyperkernel")

    decoder_config_dict = dict(model_config_dict["decoder"])
    decoder_config_dict.setdefault("block_type", "convnext")
    if "hyperkernel" in decoder_config_dict and "hyperkernel_config" not in decoder_config_dict:
        decoder_config_dict["hyperkernel_config"] = decoder_config_dict.pop("hyperkernel")

    # For masked-model checkpoints (beta_nll / evidential), ensure num_outputs
    # is present so the decoder allocates the right output head.
    uncertainty_method = model_config_dict.get("uncertainty_method", "beta_nll")
    if "num_outputs" not in decoder_config_dict:
        decoder_config_dict["num_outputs"] = (
            4 if uncertainty_method == "evidential" else 2
        )

    model = MultiplexAutoencoder(
        num_channels=num_channels,
        encoder_config=encoder_config,
        decoder_config=decoder_config_dict,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    else:
        state_dict = ckpt
        
    model.load_state_dict(state_dict, strict=strict)
    model = model.to(device)
    model.eval()
    return model


def _discover_models(
    args: argparse.Namespace,
) -> list[tuple[str, str, str]]:
    """Return list of (checkpoint_path, config_path, model_label) tuples.

    Single-model mode (--config + --checkpoint) returns exactly one entry.
    Batch-discovery mode (--versions / --checkpoint-glob) scans --models-path.
    """
    # --- single-model mode ---
    if args.config is not None:
        label = args.model_label or Path(args.checkpoint).stem
        return [(args.checkpoint, args.config, label)]

    # --- batch-discovery mode ---
    patterns = args.checkpoint_glob or [
        f"Immu*-6{v:02d}-*.pth" for v in args.versions
    ]
    model_files: list[str] = []
    for pattern in patterns:
        model_files.extend(glob(f"{args.models_path}/{pattern}"))

    if not model_files:
        raise FileNotFoundError(
            f"No model checkpoints found in {args.models_path} "
            f"for patterns {patterns}."
        )

    entries: list[tuple[str, str, str]] = []
    for model_path in sorted(model_files):
        checkpoint_name = os.path.basename(model_path)
        model_idx = checkpoint_name.split("-")[1]
        config_name = f"config.{checkpoint_name.split('.')[0]}.yaml"
        config_path = f"{args.models_path}/{config_name}"
        if not os.path.exists(config_path):
            print(
                f"Skipping {checkpoint_name}: config not found at {config_path}"
            )
            continue
        label = args.model_label or f"ImmuVis-{model_idx}"
        entries.append((model_path, config_path, label))

    return entries


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

    model_entries = _discover_models(args)
    os.makedirs(args.results_dir, exist_ok=True)

    for checkpoint_path, config_path, model_label in model_entries:
        print(f"\n{'=' * 60}")
        print(f"Model: {model_label}")
        print(f"Config: {config_path}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{'=' * 60}")

        model_config_dict = _load_yaml(config_path)

        # Resolve panel/tokenizer from the training config if not overridden.
        effective_panel = args.panel_config
        effective_tokenizer = args.tokenizer_config
        if "panel_config" in model_config_dict and not os.path.isabs(args.panel_config):
            candidate = model_config_dict["panel_config"]
            if os.path.exists(candidate):
                effective_panel = candidate
        if "tokenizer_config" in model_config_dict and not os.path.isabs(args.tokenizer_config):
            candidate = model_config_dict["tokenizer_config"]
            if os.path.exists(candidate):
                effective_tokenizer = candidate

        panel_config_dict, tokenizer, data_config = _resolve_dataset_setup(
            raw_model_config=model_config_dict,
            panel_config_path=effective_panel,
            tokenizer_config_path=effective_tokenizer,
            data_config_path=args.data_config,
            data_overrides=data_overrides,
        )

        if args.datasets is not None:
            panel_config_dict = panel_config_dict.copy()
            panel_config_dict["datasets"] = args.datasets

        inv_tokenizer = {v: k for k, v in tokenizer.items()}
        num_channels = len(tokenizer)

        test_transform = TestCrop(args.crop_size)
        
        test_dataset = DatasetFromTIFF(
            panels_config=panel_config_dict,
            split=args.split,
            marker_tokenizer=tokenizer,
            transform=test_transform,
            use_preprocessing=False,
            use_butterworth_filter=True,
            use_clip_normalization=True,
            file_extension="npy"
        )

        print(f"Test dataset size: {len(test_dataset)} images")

        test_batch_sampler = PanelBatchSampler(test_dataset, batch_size=1, shuffle=False)
        dataloader = DataLoader(
            test_dataset, 
            batch_sampler=test_batch_sampler,
            num_workers=4, 
            pin_memory=False
        )
        print(f"Test dataset size: {len(test_dataset)} images")

        print(f"Loading model weights from: {checkpoint_path}")
        model = _build_model(
            model_config_dict=model_config_dict,
            num_channels=num_channels,
            strict=args.strict_loading,
            checkpoint_path=checkpoint_path,
            device=device,
        )

        all_mse = []
        all_uncertainties = []
        all_pearson_r = []
        all_channel_ids = []
        all_dataset_names = []
        all_image_paths = []

        safe_label = model_label.replace("/", "_").replace("\\", "_")
        recon_dir = Path(args.recon_dir) / f"{safe_label}_loo"
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
                        "model": model_label,
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
        df["model"] = model_label
        df["dataset_name"] = all_dataset_names
        df["image_path"] = all_image_paths
        df["mu_activation"] = args.mu_activation

        output_file = os.path.join(args.results_dir, f"{safe_label}_loo.csv")
        print(f"Saving results to: {output_file}")
        df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
