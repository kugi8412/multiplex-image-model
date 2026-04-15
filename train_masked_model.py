#!/usr/bin/env python
# -*- coding: utf-8 -*-
# train_masked_model.py

import os
import sys

import comet_ml  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from ruamel.yaml import YAML
from torch.amp import GradScaler, autocast
from torch.nn.functional import normalize
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
    RandomRotation,
)
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler, TestCrop
from multiplex_model.losses import RankMe, beta_nll_loss, nll_loss, evidential_loss, evidential_uncertainty, get_output_activation, LearnableOutputActivation
from multiplex_model.modules import MultiplexAutoencoder
from multiplex_model.utils import (
    ClampWithGrad,
    TrainingConfig,
    apply_channel_masking,
    apply_spatial_masking,
    finish_experiment,
    get_run_name,
    get_scheduler_with_warmup,
    init_experiment,
    log_training_metrics,
    log_validation_images,
    log_validation_metrics,
    plot_reconstructs_with_masks,
)


def train_masked(
    model,
    optimizer,
    scheduler,
    train_dataloader,
    val_dataloader,
    device,
    marker_names_map,
    epochs=10,
    gradient_accumulation_steps=1,
    beta=1.0,
    min_channels_frac=0.75,
    fully_masked_channels_max_frac=0.5,
    spatial_masking_ratio=0.6,
    mask_patch_size=8,
    start_epoch=0,
    save_checkpoint_every=5,
    checkpoints_path="checkpoints",
    uncertainty_method="evidential",
    evidence_reg_coeff=0.01,
    mask_strategy="learnable",
    output_activation="swishoid",
    activation_beta=4.0,
    activation_window=0.5,
    learnable_activation_params=False,
    activation_module=None,
):
    """Train a masked autoencoder with the given parameters.

    Args:
        uncertainty_method: 'evidential' (Normal-Inverse-Gamma, 4 outputs) or
                           'beta_nll' (Gaussian, 2 outputs). Default: 'evidential'.
        evidence_reg_coeff: Regularization coefficient for evidential loss.
            Controls how strongly the model is penalized for being confident
            and wrong. Only used when uncertainty_method='evidential'.
        mask_strategy: 'zero' (fill masked pixels with 0.0),
            'negative' (fill with -1.0), or 'learnable' (learnable mask token
            in encoder feature space). Default: 'learnable'.
        output_activation: Activation for predicted mean: 'sigmoid',
            'hard_sigmoid', 'sigmoswish', or 'swishoid'. Default: 'swishoid'.
        activation_beta: Temperature for sigmoswish/swishoid. Default: 4.0.
        activation_window: Half-width for swishoid. Default: 0.5.
        learnable_activation_params: If True, beta/window are nn.Parameters.
        activation_module: Pre-built LearnableOutputActivation (passed from main).
    """
    model.train()
    scaler = GradScaler()
    run_name = get_run_name()

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path, exist_ok=True)
        print(f"Created checkpoints directory at {checkpoints_path}")

    # Resolve mask strategy
    mask_fill_values = {"zero": 0.0, "negative": -1.0, "learnable": 0.0}
    mask_fill_value = mask_fill_values[mask_strategy]
    use_mask_token = (mask_strategy == "learnable")

    # Resolve output activation
    if activation_module is not None:
        activation_fn = activation_module
    elif learnable_activation_params:
        activation_fn = LearnableOutputActivation(
            name=output_activation, beta=activation_beta,
            window=activation_window, learnable=True,
        ).to(device)
    else:
        activation_fn = get_output_activation(
            output_activation, beta=activation_beta, window=activation_window,
        )

    print(f"Uncertainty method: {uncertainty_method}")
    print(f"Mask strategy: {mask_strategy} (fill_value={mask_fill_value}, learnable_token={use_mask_token})")
    print(f"Output activation: {output_activation} (beta={activation_beta}, window={activation_window}, learnable={learnable_activation_params})")

    step = start_epoch * (len(train_dataloader) // gradient_accumulation_steps)
    for epoch in range(start_epoch, epochs):
        model.train()
        for batch_idx, (img, channel_ids, panel_idx, img_path) in enumerate(
            tqdm(train_dataloader, desc=f"Epoch {epoch}")
        ):
            img = img.to(device, dtype=torch.float32)
            channel_ids = channel_ids.to(device, dtype=torch.long)

            # Apply channel masking with channel subset sampling
            img, channel_ids, masked_img, active_channel_ids = apply_channel_masking(
                img,
                channel_ids,
                min_channels_frac,
                fully_masked_channels_max_frac,
                apply_channel_subset_sampling=True,
            )

            # Apply spatial masking
            masked_img, spatial_mask = apply_spatial_masking(
                masked_img, spatial_masking_ratio, mask_patch_size,
                mask_fill_value=mask_fill_value,
            )

            # Only pass spatial_mask to encoder when using learnable mask token
            encoder_mask = spatial_mask if use_mask_token else None

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(masked_img, active_channel_ids, channel_ids,
                               spatial_mask=encoder_mask)["output"]

                if uncertainty_method == "evidential":
                    # Evidential Deep Regression: 4 outputs per pixel
                    gamma_raw, nu_raw, alpha_raw, beta_raw = output.unbind(dim=-1)
                    mi = activation_fn(gamma_raw)
                    nu = torch.nn.functional.softplus(nu_raw) + 1e-6
                    alpha = torch.nn.functional.softplus(alpha_raw) + 1.0 + 1e-6
                    beta_param = torch.nn.functional.softplus(beta_raw) + 1e-6

                    loss = evidential_loss(
                        img, mi, nu, alpha, beta_param,
                        reg_coeff=evidence_reg_coeff,
                    )

                    # Decompose uncertainty for logging
                    with torch.no_grad():
                        aleatoric, epistemic, total_unc = evidential_uncertainty(
                            nu, alpha, beta_param
                        )
                else:
                    # Legacy beta-NLL: 2 outputs per pixel
                    mi, logvar = output.unbind(dim=-1)
                    mi = activation_fn(mi)
                    logvar = ClampWithGrad.apply(logvar, -15.0, 15.0)
                    loss = beta_nll_loss(img, mi, logvar, beta=beta)

            scaler.scale(loss / gradient_accumulation_steps).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                log_dict = {
                    "loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "mu": mi.mean().item(),
                    "mae": torch.abs(img - mi).mean().item(),
                    "mse": torch.square(img - mi).mean().item(),
                    "step": step,
                }

                if uncertainty_method == "evidential":
                    log_dict["aleatoric_unc"] = aleatoric.mean().item()
                    log_dict["epistemic_unc"] = epistemic.mean().item()
                    log_dict["total_unc"] = total_unc.mean().item()
                    log_dict["evidence_nu"] = nu.mean().item()
                    log_dict["alpha"] = alpha.mean().item()
                    log_training_metrics(**log_dict)
                else:
                    log_dict["logvar"] = logvar.mean().item()
                    log_training_metrics(**log_dict)

                step += 1

        test_masked(
            model,
            val_dataloader,
            device,
            epoch,
            spatial_masking_ratio=spatial_masking_ratio,
            fully_masked_channels_max_frac=fully_masked_channels_max_frac,
            mask_patch_size=mask_patch_size,
            marker_names_map=marker_names_map,
            uncertainty_method=uncertainty_method,
            mask_strategy=mask_strategy,
            output_activation=output_activation,
            activation_beta=activation_beta,
            activation_window=activation_window,
            activation_module=activation_fn if isinstance(activation_fn, LearnableOutputActivation) else None,
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
        }
        if isinstance(activation_fn, LearnableOutputActivation):
            checkpoint["activation_state_dict"] = activation_fn.state_dict()
        if (epoch + 1) % save_checkpoint_every == 0:
            torch.save(
                checkpoint,
                f"{checkpoints_path}/checkpoint-{run_name}-epoch_{epoch}.pth",
            )
        torch.save(checkpoint, f"{checkpoints_path}/last_checkpoint-{run_name}.pth")

    final_model_path = f"{checkpoints_path}/final_model-{run_name}.pth"
    print(f"Training completed. Saving final model at {final_model_path}...")
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    if isinstance(activation_fn, LearnableOutputActivation):
        checkpoint["activation_state_dict"] = activation_fn.state_dict()
    torch.save(checkpoint, final_model_path)


def test_masked(
    model,
    test_dataloader,
    device,
    epoch,
    marker_names_map,
    num_plots=4,
    spatial_masking_ratio=0.6,
    fully_masked_channels_max_frac=0.5,
    mask_patch_size=8,
    uncertainty_method="evidential",
    mask_strategy="learnable",
    output_activation="swishoid",
    activation_beta=4.0,
    activation_window=0.5,
    learnable_activation_params=False,
    activation_module=None,
):
    model.eval()

    # Resolve mask strategy and activation
    mask_fill_values = {"zero": 0.0, "negative": -1.0, "learnable": 0.0}
    mask_fill_value = mask_fill_values[mask_strategy]
    use_mask_token = (mask_strategy == "learnable")
    if activation_module is not None:
        activation_fn = activation_module
    else:
        activation_fn = get_output_activation(
            output_activation, beta=activation_beta, window=activation_window,
        )
    running_loss = 0.0
    running_mae = 0.0
    running_mse = 0.0
    plot_indices = np.random.choice(
        np.arange(len(test_dataloader)), size=num_plots, replace=False
    )
    plot_indices = set(plot_indices)

    all_latents = []
    all_channel_aleatoric = []
    all_channel_epistemic = []
    all_channel_maes = []

    with torch.no_grad():
        for idx, (img, channel_ids, panel_idx, img_path) in enumerate(
            tqdm(test_dataloader, desc=f"Testing epoch {epoch}")
        ):
            img = img.to(device, dtype=torch.float32)
            channel_ids = channel_ids.to(device, dtype=torch.long)

            # Apply channel masking (only full channel masking for validation, no channel dropping)
            _, _, masked_img, active_channel_ids = apply_channel_masking(
                img,
                channel_ids,
                fully_masked_channels_max_frac=fully_masked_channels_max_frac,
                apply_channel_subset_sampling=False,
            )

            # Apply spatial masking
            masked_img, pixel_mask = apply_spatial_masking(
                masked_img, spatial_masking_ratio, mask_patch_size,
                mask_fill_value=mask_fill_value,
            )

            encoder_mask = pixel_mask if use_mask_token else None

            latent = model.encode(masked_img, active_channel_ids,
                                   spatial_mask=encoder_mask)["output"]
            output = model.decode(latent, channel_ids)

            if uncertainty_method == "evidential":
                gamma_raw, nu_raw, alpha_raw, beta_raw = output.unbind(dim=-1)
                mi = activation_fn(gamma_raw)
                nu = torch.nn.functional.softplus(nu_raw) + 1e-6
                alpha = torch.nn.functional.softplus(alpha_raw) + 1.0 + 1e-6
                beta_param = torch.nn.functional.softplus(beta_raw) + 1e-6

                aleatoric, epistemic, total_unc = evidential_uncertainty(
                    nu, alpha, beta_param
                )

                # Use Student-t NLL as validation loss
                error = img - mi
                omega = 2.0 * beta_param * (1.0 + nu)
                val_nll = (
                    0.5 * torch.log(torch.pi / (nu + 1e-8))
                    - alpha * torch.log(omega + 1e-8)
                    + (alpha + 0.5) * torch.log(error.pow(2) * nu + omega + 1e-8)
                    + torch.lgamma(alpha)
                    - torch.lgamma(alpha + 0.5)
                )
                loss = val_nll.mean()

                # Per-channel uncertainty decomposition
                aleatoric_per_channel = aleatoric.mean(dim=(0, 2, 3))
                epistemic_per_channel = epistemic.mean(dim=(0, 2, 3))
                all_channel_aleatoric.append(aleatoric_per_channel.cpu())
                all_channel_epistemic.append(epistemic_per_channel.cpu())
            else:
                mi, logvar = output.unbind(dim=-1)
                mi = activation_fn(mi)
                loss = nll_loss(img, mi, logvar)

                # Legacy: treat exp(logvar) as both aleatoric and epistemic
                variance_per_channel = torch.exp(logvar).mean(dim=(0, 2, 3))
                all_channel_aleatoric.append(variance_per_channel.cpu())

            latent = normalize(latent.mean(dim=(2, 3)), p=2, dim=1)
            all_latents.append(latent.cpu())

            mae_per_channel = torch.abs(img - mi).mean(dim=(0, 2, 3))
            all_channel_maes.append(mae_per_channel.cpu())

            running_loss += loss.item()
            running_mae += torch.abs(img - mi).mean().item()
            running_mse += torch.square(img - mi).mean().item()

            if idx in plot_indices:
                unactive_channels = [
                    i for i in channel_ids[0] if i not in active_channel_ids[0]
                ]
                masked_channels_names = " | ".join(
                    [marker_names_map[i.item()] for i in unactive_channels]
                )

                reconstr_img = plot_reconstructs_with_masks(
                    img,
                    mi,
                    pixel_mask,
                    channel_ids,
                    unactive_channels,
                    markers_names_map=marker_names_map,
                    ncols=9,
                )
                log_validation_images(
                    fig=reconstr_img,
                    panel_idx=panel_idx[0],
                    img_path=img_path[0],
                    epoch=epoch,
                    masked_channels_names=masked_channels_names,
                    img_idx=idx,
                )
                plt.close("all")

    val_loss = running_loss / len(test_dataloader)
    val_mae = running_mae / len(test_dataloader)
    val_mse = running_mse / len(test_dataloader)

    all_latents = torch.cat(all_latents)
    rankme = RankMe(all_latents)

    # Uncertainty-MAE correlations
    all_channel_aleatoric = torch.cat(all_channel_aleatoric)
    all_channel_maes = torch.cat(all_channel_maes)
    aleatoric_mae_corr = torch.corrcoef(
        torch.stack([all_channel_aleatoric.flatten(), all_channel_maes.flatten()])
    )[0, 1].item()

    val_metrics = {
        "val_loss": val_loss,
        "val_mae": val_mae,
        "val_mse": val_mse,
        "latent_rankme": rankme,
        "aleatoric_mae_correlation": aleatoric_mae_corr,
        "epoch": epoch,
    }

    if uncertainty_method == "evidential":
        all_channel_epistemic = torch.cat(all_channel_epistemic)
        epistemic_mae_corr = torch.corrcoef(
            torch.stack([all_channel_epistemic.flatten(), all_channel_maes.flatten()])
        )[0, 1].item()
        val_metrics["epistemic_mae_correlation"] = epistemic_mae_corr
        val_metrics["mean_aleatoric"] = all_channel_aleatoric.mean().item()
        val_metrics["mean_epistemic"] = all_channel_epistemic.mean().item()

    log_validation_metrics(**val_metrics)

    print(f"{'=' * 40} EPOCH {epoch + 1} {'=' * 40}")
    print(f"Loss: {val_loss:.4f}")
    print(f"MAE: {val_mae:.6f}")
    print(f"MSE: {val_mse:.6f}")
    print(f"Aleatoric-MAE Corr: {aleatoric_mae_corr:.4f}")
    if uncertainty_method == "evidential":
        print(f"Epistemic-MAE Corr: {epistemic_mae_corr:.4f}")
        print(f"Aleatoric (mean): {val_metrics['mean_aleatoric']:.6f}")
        print(f"Epistemic (mean): {val_metrics['mean_epistemic']:.6f}")
    print("=" * 90)
    print()

    return val_metrics


if __name__ == "__main__":
    # Load the configuration file
    config_path = sys.argv[1]
    yaml = YAML(typ="safe")
    with open(config_path, "r") as f:
        raw_config = yaml.load(f)

    # Validate configuration using Pydantic model
    config = TrainingConfig(**raw_config)

    device = config.device
    print(f"Using device: {device}")

    SIZE = config.input_image_size
    BATCH_SIZE = config.batch_size
    NUM_WORKERS = config.num_workers

    PANEL_CONFIG = YAML().load(open(config.panel_config))
    TOKENIZER = YAML().load(open(config.tokenizer_config))
    INV_TOKENIZER = {v: k for k, v in TOKENIZER.items()}

    train_transform = Compose(
        [
            RandomRotation(180, interpolation=InterpolationMode.BILINEAR),
            RandomCrop(SIZE),
            RandomHorizontalFlip(),
        ]
    )

    test_transform = TestCrop(SIZE[0])

    train_dataset = DatasetFromTIFF(
        panels_config=PANEL_CONFIG,
        split="train",
        marker_tokenizer=TOKENIZER,
        transform=train_transform,
        use_preprocessing=False,  # saved data is already preprocessed
        use_median_denoising=False,
        use_butterworth_filter=True,
        use_minmax_normalization=False,
        use_clip_normalization=True,
        file_extension="npy",
    )

    test_dataset = DatasetFromTIFF(
        panels_config=PANEL_CONFIG,
        split="test",
        marker_tokenizer=TOKENIZER,
        transform=test_transform,
        use_preprocessing=False,  # saved data is already preprocessed
        use_median_denoising=False,
        use_butterworth_filter=True,
        use_minmax_normalization=False,
        use_clip_normalization=True,
        file_extension="npy",
    )

    train_batch_sampler = PanelBatchSampler(train_dataset, BATCH_SIZE)
    test_batch_sampler = PanelBatchSampler(test_dataset, BATCH_SIZE, shuffle=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_sampler=test_batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # Build model configuration
    num_channels = len(TOKENIZER)

    # Set num_outputs based on uncertainty method:
    # evidential = 4 (gamma, nu, alpha, beta), beta_nll = 2 (mu, logvar)
    decoder_cfg = config.decoder_config.model_dump()
    if config.uncertainty_method == "evidential":
        decoder_cfg["num_outputs"] = 4
        print("Evidential Deep Regression: decoder outputs 4 per pixel (gamma, nu, alpha, beta)")
    else:
        decoder_cfg["num_outputs"] = 2
        print("Beta-NLL: decoder outputs 2 per pixel (mu, logvar)")

    model = MultiplexAutoencoder(
        num_channels=num_channels,
        encoder_config=config.encoder_config.model_dump(),
        decoder_config=decoder_cfg,
    ).to(device)

    # Build learnable activation module (if configured)
    activation_module = None
    if config.learnable_activation_params:
        activation_module = LearnableOutputActivation(
            name=config.output_activation,
            beta=config.activation_beta,
            window=config.activation_window,
            learnable=True,
        ).to(device)
        print(f"Learnable activation: {activation_module}")

    # Setup optimizer and scheduler — include activation params if learnable
    total_steps = (
        len(train_dataloader) * config.epochs // config.gradient_accumulation_steps
    )
    num_warmup_steps = int(total_steps * config.frac_warmup_steps)
    num_annealing_steps = total_steps - num_warmup_steps

    param_groups = list(model.parameters())
    if activation_module is not None:
        param_groups += list(activation_module.parameters())

    optimizer = optim.AdamW(
        param_groups, lr=config.peak_lr, weight_decay=config.weight_decay
    )
    scheduler = get_scheduler_with_warmup(
        optimizer,
        num_warmup_steps,
        num_annealing_steps,
        final_lr=config.final_lr,
        peak_lr=config.peak_lr,
        type="cosine",
    )

    # Initialize Comet.ml experiment
    comet_config = config.model_dump()
    init_experiment(comet_config)

    # Load checkpoint if specified
    start_epoch = 0
    if config.resolve_checkpoint():
        print(f"Loading model from checkpoint: {config.from_checkpoint}")
        checkpoint = torch.load(config.from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        if activation_module is not None and "activation_state_dict" in checkpoint:
            activation_module.load_state_dict(checkpoint["activation_state_dict"])
            print(f"Loaded learnable activation state: {activation_module}")

    # Train the model
    train_masked(
        model,
        optimizer,
        scheduler,
        train_dataloader,
        test_dataloader,
        device,
        marker_names_map=INV_TOKENIZER,
        epochs=config.epochs,
        start_epoch=start_epoch,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        min_channels_frac=config.min_channels_frac,
        spatial_masking_ratio=config.spatial_masking_ratio,
        fully_masked_channels_max_frac=config.fully_masked_channels_max_frac,
        mask_patch_size=config.mask_patch_size,
        save_checkpoint_every=config.save_checkpoint_freq,
        checkpoints_path=config.checkpoints_dir,
        beta=config.beta,
        uncertainty_method=config.uncertainty_method,
        evidence_reg_coeff=config.evidence_reg_coeff,
        mask_strategy=config.mask_strategy,
        output_activation=config.output_activation,
        activation_beta=config.activation_beta,
        activation_window=config.activation_window,
        learnable_activation_params=config.learnable_activation_params,
        activation_module=activation_module,
    )

    finish_experiment()
