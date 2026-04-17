#!/usr/bin/env python
# -*- coding: utf-8 -*-
# attribution.py


"""
Marker attribution for ImmuVis multiplex image models.

Computes per-marker, per-pixel saliency maps showing which input channels
and spatial locations drive the model's prediction for a given target
marker.  Three methods are implemented:

    1. **Gradient * Input** — fast, single backward pass. Shows which
       pixels have both high input value AND high gradient. Noisy but cheap.

    2. Integrated Gradients (IG)

    3. Uncertainty-weighted Integrated Gradients — same as IG but
       weights the final attribution by the model's predicted confidence
       (inverse uncertainty). Regions where the model is uncertain get
       attenuated, focusing analysis on trustworthy attributions.

All methods support the configurable output activation (sigmoid /
hard_sigmoid / sigmoswish) and evidential / beta-NLL uncertainty.

Usage (CLI)::

    python -m xai.attribution \\
        --config configs/train_mambaswin_config.yaml \\
        --checkpoint checkpoints/final_model.pth \\
        --target-marker PDL1 \\
        --method integrated_gradients \\
        --output-dir xai_attributions/

Usage (API)::

    from xai.attribution import MarkerAttribution
    attr = MarkerAttribution(model, tokenizer, ...)
    maps = attr.attribute(image, channel_ids, target_marker="PDL1")
"""


from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from tqdm import tqdm
from multiplex_model.losses import get_output_activation

from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler, TestCrop
from multiplex_model.modules import MultiplexAutoencoder
from multiplex_model.utils.configuration import TrainingConfig


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------

@dataclass
class AttributionResult:
    """Stores attribution maps for a single image."""

    attributions: np.ndarray         # (C_input, H, W) per-channel saliency
    prediction: np.ndarray           # (H, W) predicted target intensity
    uncertainty: np.ndarray | None   # (H, W) predicted uncertainty
    marker_names: list[str] = field(default_factory=list)
    target_marker: str = ""
    method: str = ""

    @property
    def channel_importance(self) -> dict[str, float]:
        """Mean absolute attribution per marker — summarises global importance."""
        return {
            name: float(np.abs(self.attributions[i]).mean())
            for i, name in enumerate(self.marker_names)
        }


# ------------------------------------------------------------------
# Core engine
# ------------------------------------------------------------------

class MarkerAttribution:
    """Per-marker, per-pixel attribution engine.

    Parameters
    ----------
    model : MultiplexAutoencoder
        Trained model (will be set to eval, no param updates).
    tokenizer : dict[str, int]
        Marker name → token ID.
    output_activation : str
        Must match training config.
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
    # Internal decode
    # ------------------------------------------------------------------

    def _decode_output(self, output: torch.Tensor):
        """Unpack decoder output → (prediction, uncertainty)."""
        if self.uncertainty_method == "evidential":
            gamma_raw, nu_raw, alpha_raw, beta_raw = output.unbind(dim=-1)
            pred = self.activation_fn(gamma_raw)
            nu = F.softplus(nu_raw) + 1e-6
            alpha = F.softplus(alpha_raw) + 1.0 + 1e-6
            beta = F.softplus(beta_raw) + 1e-6
            alpha_m1 = alpha - 1.0 + 1e-8
            uncertainty = beta * (1.0 + nu) / (nu * alpha_m1)
        else:
            mi_raw, logvar = output.unbind(dim=-1)
            pred = self.activation_fn(mi_raw)
            uncertainty = torch.exp(logvar)
        return pred, uncertainty

    def _forward_score(
        self,
        x: torch.Tensor,
        channel_ids: torch.Tensor,
        target_idx: int,
    ):
        """Forward pass -> scalar target score and uncertainty."""
        output = self.model(x, channel_ids, channel_ids)["output"]
        pred, unc = self._decode_output(output)
        target_score = pred[0, target_idx].mean()
        target_unc = unc[0, target_idx].mean()
        return target_score, target_unc, pred, unc

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def gradient_x_input(
        self,
        img: torch.Tensor,
        channel_ids: torch.Tensor,
        target_marker: str,
    ) -> AttributionResult:
        """Gradient x Input saliency.

        Fast (single backward) but noisy. Attribution = ∂target/∂x * x.
        """
        img = img.to(self.device, dtype=torch.float32)
        channel_ids = channel_ids.to(self.device)
        target_idx = self._get_target_idx(channel_ids, target_marker)
        marker_names = [self.inv_tokenizer[cid.item()] for cid in channel_ids[0]]

        x = img.clone().requires_grad_(True)

        with autocast(device_type=self.device.split(":")[0], dtype=torch.bfloat16):
            score, unc, pred, unc_map = self._forward_score(
                x, channel_ids, target_idx
            )

        grad = torch.autograd.grad(score, x)[0]  # (1, C, H, W)
        attr = (grad * x.detach()).float()  # gradient × input

        return AttributionResult(
            attributions=attr[0].detach().cpu().numpy(),
            prediction=pred[0, target_idx].detach().float().cpu().numpy(),
            uncertainty=unc_map[0, target_idx].detach().float().cpu().numpy(),
            marker_names=marker_names,
            target_marker=target_marker,
            method="gradient_x_input",
        )

    def integrated_gradients(
        self,
        img: torch.Tensor,
        channel_ids: torch.Tensor,
        target_marker: str,
        n_steps: int = 50,
        baseline: torch.Tensor | None = None,
        uncertainty_weighted: bool = False,
    ) -> AttributionResult:
        """Integrated Gradients (Sundararajan et al., 2017).

        Accumulates gradients along a linear path from baseline to input:

            IG_i = (x_i - b_i) * (1/m) * sum_k dF/dx_i |_{b + k/m * (x-b)}

        Parameters:
        n_steps : int
            Number of interpolation steps (higher = more accurate).
        baseline : Tensor, optional
            Reference input. Default is zeros (black image).
        uncertainty_weighted : bool
            If True, multiplies attribution by inverse uncertainty.
        """
        img = img.to(self.device, dtype=torch.float32)
        channel_ids = channel_ids.to(self.device)
        target_idx = self._get_target_idx(channel_ids, target_marker)
        marker_names = [self.inv_tokenizer[cid.item()] for cid in channel_ids[0]]

        if baseline is None:
            baseline = torch.zeros_like(img)
        else:
            baseline = baseline.to(self.device, dtype=torch.float32)

        # Accumulate gradients along interpolation path
        delta = img - baseline
        grad_sum = torch.zeros_like(img)

        for step in range(1, n_steps + 1):
            alpha = step / n_steps
            x_interp = (baseline + alpha * delta).requires_grad_(True)

            with autocast(device_type=self.device.split(":")[0], dtype=torch.bfloat16):
                score, _, _, _ = self._forward_score(x_interp, channel_ids, target_idx)

            grad = torch.autograd.grad(score, x_interp)[0]
            grad_sum += grad.float()

        # IG = (x - baseline) * mean(gradients)
        attr = delta * grad_sum / n_steps

        # Get final prediction and uncertainty
        with torch.no_grad(), autocast(device_type=self.device.split(":")[0], dtype=torch.bfloat16):
            _, _, pred, unc_map = self._forward_score(
                img, channel_ids, target_idx
            )

        if uncertainty_weighted:
            inv_unc = 1.0 / (1.0 + unc_map[0, target_idx].float())
            attr[0] = attr[0] * inv_unc.unsqueeze(0)

        method_name = "uncertainty_weighted_ig" if uncertainty_weighted else "integrated_gradients"

        return AttributionResult(
            attributions=attr[0].detach().cpu().numpy(),
            prediction=pred[0, target_idx].detach().float().cpu().numpy(),
            uncertainty=unc_map[0, target_idx].detach().float().cpu().numpy(),
            marker_names=marker_names,
            target_marker=target_marker,
            method=method_name,
        )

    def attribute(
        self,
        img: torch.Tensor,
        channel_ids: torch.Tensor,
        target_marker: str,
        method: str = "integrated_gradients",
        **kwargs,
    ) -> AttributionResult:
        """Dispatch to the appropriate attribution method.

        Parameters
        ----------
        method : str
            'gradient_x_input', 'integrated_gradients', or
            'uncertainty_weighted_ig'.
        """
        if method == "gradient_x_input":
            return self.gradient_x_input(img, channel_ids, target_marker)
        elif method == "integrated_gradients":
            return self.integrated_gradients(
                img, channel_ids, target_marker, **kwargs
            )
        elif method == "uncertainty_weighted_ig":
            return self.integrated_gradients(
                img, channel_ids, target_marker,
                uncertainty_weighted=True, **kwargs
            )
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose from: "
                "gradient_x_input, integrated_gradients, uncertainty_weighted_ig"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_target_idx(self, channel_ids: torch.Tensor, target_marker: str) -> int:
        """Find index of target marker in channel_ids."""
        token_id = self.tokenizer[target_marker]
        matches = (channel_ids[0] == token_id).nonzero(as_tuple=True)[0]
        if len(matches) == 0:
            raise ValueError(
                f"Target marker '{target_marker}' (token {token_id}) "
                f"not found in channel_ids"
            )
        return matches[0].item()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Marker attribution maps for ImmuVis models",
    )
    parser.add_argument("--config", required=True, help="Training config YAML")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint .pth")
    parser.add_argument(
        "--target-marker", required=True,
        help="Marker name to compute attribution for (e.g. PDL1)",
    )
    parser.add_argument(
        "--method", default="integrated_gradients",
        choices=["gradient_x_input", "integrated_gradients", "uncertainty_weighted_ig"],
    )
    parser.add_argument("--ig-steps", type=int, default=50)
    parser.add_argument("--output-dir", default="xai_attributions")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-images", type=int, default=None)
    args = parser.parse_args()

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
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=4)

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

    attr_engine = MarkerAttribution(
        model=model,
        tokenizer=TOKENIZER,
        output_activation=config.output_activation,
        activation_beta=config.activation_beta,
        uncertainty_method=config.uncertainty_method,
        device=args.device,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    importance_records = []

    for _, (img, channel_ids, panel_idx, img_path) in enumerate(
        tqdm(test_loader, desc=f"Attribution ({args.method})")
    ):
        if args.num_images is not None and processed >= args.num_images:
            break

        panel_markers = PANEL_CONFIG["markers"][panel_idx[0]]
        if args.target_marker not in panel_markers:
            continue

        result = attr_engine.attribute(
            img, channel_ids,
            target_marker=args.target_marker,
            method=args.method,
            n_steps=args.ig_steps,
        )

        stem = Path(img_path[0]).stem
        np.savez_compressed(
            out_dir / f"{stem}_attr_{args.target_marker}.npz",
            attributions=result.attributions,
            prediction=result.prediction,
            uncertainty=result.uncertainty,
            marker_names=result.marker_names,
        )

        importance_records.append({
            "image": stem,
            "panel": panel_idx[0],
            **result.channel_importance,
        })
        processed += 1

    # Save summary CSV
    if importance_records:
        import pandas as pd
        df = pd.DataFrame(importance_records)
        df.to_csv(out_dir / f"importance_summary_{args.target_marker}.csv", index=False)
        print(f"\nTop contributing markers for {args.target_marker}:")
        mean_imp = df.select_dtypes(include=[np.number]).mean().sort_values(ascending=False)
        for name, val in mean_imp.head(10).items():
            print(f"  {name}: {val:.6f}")

    print(f"Processed {processed} images → {out_dir}")


if __name__ == "__main__":
    main()
