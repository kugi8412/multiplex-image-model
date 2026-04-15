#!/usr/bin/env python
# -*- coding: utf-8 -*-
# counterfactual.py


"""Counterfactual perturbation for ImmuVis multiplex image models.

Given a trained autoencoder, removes one or more target markers from the
input and then iteratively perturbs the remaining (context) marker pixel
values via gradient ascent/descent to maximise or minimise the model's
predicted intensity for the target markers.

Key improvements over the original notebook prototype:
    1. **Uncertainty weighting** — gradients are scaled by the inverse of
       the model's predicted uncertainty (evidential or beta-NLL) so that
       perturbations focus on confident predictions and ignore noisy regions.
    2. **Proximity regularisation** — an L2 penalty keeps the perturbed
       image close to the original, preventing adversarial drift.
    3. **Butterworth-filtered gradients** — optional low-pass filtering on
       the gradient itself removes high-frequency noise that creates
       unrealistic pixel-level artefacts.
    4. **Configurable activation** — uses the same output activation the
       model was trained with (sigmoid / hard_sigmoid / sigmoswish).
    5. **Works with any target marker(s)** — not hardcoded to PD-L1.

Usage (CLI)::

    python -m xai.counterfactual \\
        --config configs/train_mambaswin_config.yaml \\
        --checkpoint checkpoints/final_model.pth \\
        --target-markers PDL1 PD1 \\
        --steps 300 \\
        --output-dir xai_results/

Usage (API)::

    from xai.counterfactual import CounterfactualPerturbation
    cf = CounterfactualPerturbation(model, tokenizer, ...)
    result = cf.run(image, channel_ids, target_markers=["PDL1"])
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torchvision.transforms.functional import gaussian_blur

from multiplex_model.losses import get_output_activation


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------

@dataclass
class CounterfactualResult:
    """Stores the output of a single counterfactual run."""

    original_img: np.ndarray           # (C_ctx, H, W) original context pixels
    perturbed_img: np.ndarray          # (C_ctx, H, W) perturbed context pixels
    delta: np.ndarray                  # (C_ctx, H, W) perturbed - original
    accumulated_grad: np.ndarray       # (C_ctx, H, W) summed gradient signal
    target_predictions_pre: np.ndarray  # (C_tgt, H, W) predicted target before
    target_predictions_post: np.ndarray # (C_tgt, H, W) predicted target after
    target_uncertainty_pre: np.ndarray | None  # (C_tgt, H, W) uncertainty before
    target_uncertainty_post: np.ndarray | None # (C_tgt, H, W) uncertainty after
    context_marker_names: list[str] = field(default_factory=list)
    target_marker_names: list[str] = field(default_factory=list)
    loss_curve: list[float] = field(default_factory=list)


# ------------------------------------------------------------------
# Core engine
# ------------------------------------------------------------------

class CounterfactualPerturbation:
    """Gradient-based counterfactual perturbation engine.

    Parameters
    ----------
    model : MultiplexAutoencoder
        Trained model (will be frozen — no parameter updates).
    tokenizer : dict[str, int]
        Marker name → token ID mapping.
    output_activation : str
        Must match what the model was trained with.
    activation_beta : float
        Temperature for sigmoswish.
    uncertainty_method : str
        ``'evidential'`` or ``'beta_nll'``.
    device : str
        Torch device.
    """

    def __init__(
        self,
        model,
        tokenizer: dict[str, int],
        output_activation: str = "sigmoswish",
        activation_beta: float = 2.0,
        uncertainty_method: str = "evidential",
        device: str = "cuda",
    ):
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.tokenizer = tokenizer
        self.inv_tokenizer = {v: k for k, v in tokenizer.items()}
        self.activation_fn = get_output_activation(output_activation, beta=activation_beta)
        self.uncertainty_method = uncertainty_method
        self.device = device

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _decode_output(self, output: torch.Tensor):
        """Unpack decoder output into mean prediction and uncertainty.

        Returns (prediction, uncertainty) where uncertainty is total
        predictive uncertainty for evidential, or exp(logvar) for beta-NLL.
        Both have shape matching the prediction tensor.
        """
        if self.uncertainty_method == "evidential":
            gamma_raw, nu_raw, alpha_raw, beta_raw = output.unbind(dim=-1)
            pred = self.activation_fn(gamma_raw)
            nu = F.softplus(nu_raw) + 1e-6
            alpha = F.softplus(alpha_raw) + 1.0 + 1e-6
            beta = F.softplus(beta_raw) + 1e-6
            # Total uncertainty = β(1+ν) / (ν(α-1))
            alpha_m1 = alpha - 1.0 + 1e-8
            uncertainty = beta * (1.0 + nu) / (nu * alpha_m1)
        else:
            mi_raw, logvar = output.unbind(dim=-1)
            pred = self.activation_fn(mi_raw)
            uncertainty = torch.exp(logvar)

        return pred, uncertainty

    def _remove_target_channels(
        self,
        img: torch.Tensor,
        channel_ids: torch.Tensor,
        target_token_ids: set[int],
    ):
        """Remove target markers from input, return (context_img, context_ids, full_ids)."""
        B, C, H, W = img.shape
        keep_mask = torch.tensor(
            [cid.item() not in target_token_ids for cid in channel_ids[0]],
            dtype=torch.bool,
        )
        ctx_img = img[:, keep_mask]
        ctx_ids = channel_ids[:, keep_mask]
        return ctx_img, ctx_ids

    # ------------------------------------------------------------------
    # Main perturbation loop
    # ------------------------------------------------------------------

    def run(
        self,
        img: torch.Tensor,
        channel_ids: torch.Tensor,
        target_markers: list[str],
        steps: int = 300,
        lr: float = 1e-3,
        proximity_weight: float = 0.1,
        gradient_topk_quantile: float = 0.9,
        gradient_blur_kernel: int = 5,
        gradient_blur_sigma: float = 1.5,
        maximise: bool = True,
        uncertainty_weight: float = 1.0,
    ) -> CounterfactualResult:
        """Run counterfactual perturbation.

        Parameters
        ----------
        img : Tensor (1, C, H, W)
            Single multiplex image (batch size 1).
        channel_ids : Tensor (1, C)
            Marker token IDs for each channel.
        target_markers : list[str]
            Names of markers to perturb toward (e.g. ``["PDL1"]``).
        steps : int
            Number of optimisation iterations.
        lr : float
            Learning rate for the input perturbation.
        proximity_weight : float
            L2 regularisation weight toward original image.
        gradient_topk_quantile : float
            Only keep gradients above this quantile per channel.
        gradient_blur_kernel : int
            Gaussian blur kernel size for gradient smoothing.
        gradient_blur_sigma : float
            Gaussian blur sigma for gradient smoothing.
        maximise : bool
            If True, maximise target prediction. If False, minimise.
        uncertainty_weight : float
            How strongly to weight by inverse uncertainty.
            0 = ignore uncertainty, 1 = full weighting.

        Returns
        -------
        CounterfactualResult
        """
        assert img.shape[0] == 1, "Counterfactual operates on single images (B=1)"
        img = img.to(self.device, dtype=torch.float32)
        channel_ids = channel_ids.to(self.device)

        # Identify target token IDs
        target_token_ids = set()
        target_names = []
        for m in target_markers:
            if m not in self.tokenizer:
                raise ValueError(f"Marker '{m}' not found in tokenizer")
            target_token_ids.add(self.tokenizer[m])
            target_names.append(m)

        # Get target channel indices in the full image for reading predictions
        target_channel_indices = [
            i for i in range(channel_ids.shape[1])
            if channel_ids[0, i].item() in target_token_ids
        ]
        if not target_channel_indices:
            raise ValueError(
                f"Target markers {target_markers} not present in this image's channels"
            )

        # Remove target channels from input → context
        ctx_img, ctx_ids = self._remove_target_channels(
            img, channel_ids, target_token_ids
        )
        context_marker_names = [
            self.inv_tokenizer[cid.item()] for cid in ctx_ids[0]
        ]

        # Compute pre-perturbation predictions
        with torch.no_grad(), autocast(device_type=self.device.split(":")[0], dtype=torch.bfloat16):
            pre_output = self.model(ctx_img, ctx_ids, channel_ids)["output"]
            pre_pred, pre_unc = self._decode_output(pre_output)

        target_pred_pre = pre_pred[0, target_channel_indices].detach().cpu().numpy()
        target_unc_pre = pre_unc[0, target_channel_indices].detach().cpu().numpy()

        # Prepare differentiable context image
        x = ctx_img.clone().to(torch.bfloat16)
        x.requires_grad = True
        x_orig = ctx_img.clone().to(torch.bfloat16).detach()

        optimizer = torch.optim.AdamW([x], lr=lr, weight_decay=0.0)

        accumulated_grad = torch.zeros_like(x, dtype=torch.float32)
        loss_curve = []
        sign = 1.0 if maximise else -1.0

        for step in range(steps):
            optimizer.zero_grad()

            with autocast(device_type=self.device.split(":")[0], dtype=torch.bfloat16):
                output = self.model(x, ctx_ids, channel_ids)["output"]
                pred, unc = self._decode_output(output)

                # Target score: mean over target channels and spatial dims
                target_pred = pred[0, target_channel_indices]  # (C_tgt, H, W)

                if uncertainty_weight > 0:
                    target_unc = unc[0, target_channel_indices]
                    # Weight by inverse uncertainty (confident regions matter more)
                    inv_unc = 1.0 / (1.0 + uncertainty_weight * target_unc)
                    score = (target_pred * inv_unc).mean()
                else:
                    score = target_pred.mean()

                # Proximity regularisation
                prox_loss = proximity_weight * F.mse_loss(x, x_orig)

                loss = -sign * score + prox_loss

            # Compute gradient w.r.t. input
            grad = torch.autograd.grad(loss, x, retain_graph=False)[0]

            # Per-channel top-k masking
            if gradient_topk_quantile > 0:
                for c in range(grad.shape[1]):
                    threshold = torch.quantile(
                        grad[0, c].abs().float(), gradient_topk_quantile
                    )
                    mask = grad[0, c].abs() < threshold
                    grad[0, c][mask] = 0

            # Gaussian blur on gradient for spatial smoothness
            if gradient_blur_kernel > 0:
                grad = gaussian_blur(
                    grad.float(),
                    kernel_size=gradient_blur_kernel,
                    sigma=gradient_blur_sigma,
                ).to(grad.dtype)

            # Apply gradient as optimizer step
            x_pre = x.data.clone()
            x.grad = grad
            optimizer.step()
            x.data = torch.clamp(x.data, 0, 1)

            accumulated_grad += (x.data.float() - x_pre.float())
            loss_curve.append(loss.item())

        # Compute post-perturbation predictions
        with torch.no_grad(), autocast(device_type=self.device.split(":")[0], dtype=torch.bfloat16):
            post_output = self.model(x, ctx_ids, channel_ids)["output"]
            post_pred, post_unc = self._decode_output(post_output)

        target_pred_post = post_pred[0, target_channel_indices].detach().float().cpu().numpy()
        target_unc_post = post_unc[0, target_channel_indices].detach().float().cpu().numpy()

        return CounterfactualResult(
            original_img=x_orig[0].float().cpu().numpy(),
            perturbed_img=x[0].detach().float().cpu().numpy(),
            delta=(x[0].detach().float() - x_orig[0].float()).cpu().numpy(),
            accumulated_grad=accumulated_grad[0].cpu().numpy(),
            target_predictions_pre=target_pred_pre,
            target_predictions_post=target_pred_post,
            target_uncertainty_pre=target_unc_pre,
            target_uncertainty_post=target_unc_post,
            context_marker_names=context_marker_names,
            target_marker_names=target_names,
            loss_curve=loss_curve,
        )

    # ------------------------------------------------------------------
    # Per-cell analysis
    # ------------------------------------------------------------------

    def cell_level_analysis(
        self,
        result: CounterfactualResult,
        cell_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Aggregate perturbation results per cell.

        Parameters
        ----------
        result : CounterfactualResult
            Output from ``run()``.
        cell_mask : np.ndarray (H, W)
            Integer mask where each value is a cell ID (0 = background).

        Returns
        -------
        dict with keys:
            ``'cell_ids'`` — unique cell IDs (excluding 0)
            ``'pre_expression'`` — (N_cells, C_ctx) per-cell means before
            ``'post_expression'`` — (N_cells, C_ctx) per-cell means after
            ``'fold_change'`` — (N_cells, C_ctx) post / pre ratio
            ``'importance'`` — (N_cells, C_ctx) per-cell mean |gradient|
            ``'target_pre'`` — (N_cells, C_tgt) target prediction before
            ``'target_post'`` — (N_cells, C_tgt) target prediction after
        """
        cell_ids = np.unique(cell_mask)
        cell_ids = cell_ids[cell_ids > 0]

        n_cells = len(cell_ids)
        n_ctx = result.original_img.shape[0]
        n_tgt = result.target_predictions_pre.shape[0]

        pre_expr = np.full((n_cells, n_ctx), np.nan)
        post_expr = np.full((n_cells, n_ctx), np.nan)
        importance = np.full((n_cells, n_ctx), np.nan)
        target_pre = np.full((n_cells, n_tgt), np.nan)
        target_post = np.full((n_cells, n_tgt), np.nan)

        for i, cid in enumerate(cell_ids):
            mask = cell_mask == cid
            for c in range(n_ctx):
                pre_expr[i, c] = result.original_img[c, mask].mean()
                post_expr[i, c] = result.perturbed_img[c, mask].mean()
                importance[i, c] = np.abs(result.accumulated_grad[c, mask]).mean()
            for t in range(n_tgt):
                target_pre[i, t] = result.target_predictions_pre[t, mask].mean()
                target_post[i, t] = result.target_predictions_post[t, mask].mean()

        fold_change = np.where(
            np.abs(pre_expr) > 1e-8,
            post_expr / pre_expr,
            np.nan,
        )

        return {
            "cell_ids": cell_ids,
            "pre_expression": pre_expr,
            "post_expression": post_expr,
            "fold_change": fold_change,
            "importance": importance,
            "target_pre": target_pre,
            "target_post": target_post,
            "context_markers": result.context_marker_names,
            "target_markers": result.target_marker_names,
        }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Counterfactual perturbation for ImmuVis models",
    )
    parser.add_argument("--config", required=True, help="Training config YAML")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint .pth")
    parser.add_argument(
        "--target-markers", nargs="+", required=True,
        help="Marker names to perturb toward (e.g. PDL1 PD1)",
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--proximity-weight", type=float, default=0.1)
    parser.add_argument("--uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--gradient-topk", type=float, default=0.9)
    parser.add_argument("--maximise", action="store_true", default=True)
    parser.add_argument("--minimise", action="store_true")
    parser.add_argument("--output-dir", default="xai_results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-images", type=int, default=None,
                        help="Max images to process (None = all)")
    args = parser.parse_args()

    from ruamel.yaml import YAML
    from torch.utils.data import DataLoader
    from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler, TestCrop
    from multiplex_model.modules import MultiplexAutoencoder
    from multiplex_model.utils.configuration import TrainingConfig

    yaml = YAML(typ="safe")
    with open(args.config) as f:
        raw_config = yaml.load(f)
    config = TrainingConfig(**raw_config)

    PANEL_CONFIG = YAML().load(open(config.panel_config))
    TOKENIZER = YAML().load(open(config.tokenizer_config))

    test_dataset = DatasetFromTIFF(
        panels_config=PANEL_CONFIG,
        split="test",
        marker_tokenizer=TOKENIZER,
        transform=TestCrop(config.input_image_size[0]),
        use_preprocessing=False,
        use_butterworth_filter=True,
        use_clip_normalization=True,
        file_extension="npy",
    )
    test_sampler = PanelBatchSampler(test_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_sampler=test_sampler, num_workers=4,
    )

    # Build model
    num_channels = len(TOKENIZER)
    decoder_cfg = config.decoder_config.model_dump()
    decoder_cfg["num_outputs"] = 4 if config.uncertainty_method == "evidential" else 2
    model = MultiplexAutoencoder(
        num_channels=num_channels,
        encoder_config=config.encoder_config.model_dump(),
        decoder_config=decoder_cfg,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)

    cf = CounterfactualPerturbation(
        model=model,
        tokenizer=TOKENIZER,
        output_activation=config.output_activation,
        activation_beta=config.activation_beta,
        uncertainty_method=config.uncertainty_method,
        device=args.device,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    maximise = not args.minimise

    from tqdm import tqdm
    processed = 0
    for batch_idx, (img, channel_ids, panel_idx, img_path) in enumerate(
        tqdm(test_loader, desc="Counterfactual perturbation")
    ):
        if args.num_images is not None and processed >= args.num_images:
            break

        # Check if target markers exist in this panel
        panel_markers = PANEL_CONFIG["markers"][panel_idx[0]]
        if not all(m in panel_markers for m in args.target_markers):
            continue

        result = cf.run(
            img, channel_ids,
            target_markers=args.target_markers,
            steps=args.steps,
            lr=args.lr,
            proximity_weight=args.proximity_weight,
            uncertainty_weight=args.uncertainty_weight,
            gradient_topk_quantile=args.gradient_topk,
            maximise=maximise,
        )

        # Save results
        stem = Path(img_path[0]).stem
        np.savez_compressed(
            out_dir / f"{stem}_counterfactual.npz",
            original=result.original_img,
            perturbed=result.perturbed_img,
            delta=result.delta,
            grad=result.accumulated_grad,
            target_pre=result.target_predictions_pre,
            target_post=result.target_predictions_post,
            target_unc_pre=result.target_uncertainty_pre,
            target_unc_post=result.target_uncertainty_post,
            context_markers=result.context_marker_names,
            target_markers=result.target_marker_names,
            loss_curve=np.array(result.loss_curve),
        )
        processed += 1

    print(f"Processed {processed} images → {out_dir}")


if __name__ == "__main__":
    main()
