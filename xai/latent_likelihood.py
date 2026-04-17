#!/usr/bin/env python
# -*- coding: utf-8 -*-
# train_kronos_dino.py


"""
PixelCNN-based XAI for ImmuVis multiplex image models.

Uses a trained PixelCNN on the frozen encoder's latent space to provide
explainability methods that operate on the learned latent density p(Z).

1. **Surprise maps** — per-position negative log-likelihood reveals
   spatially anomalous regions where the latent representation deviates
   from the learned distribution. High surprise = unusual tissue structure
   that the model rarely encounters.

2. **Conditional likelihood attribution** — for each input marker channel,
   measure how removing it from the input changes the log-likelihood of
   the resulting latent representation under the PixelCNN. Markers whose
   removal causes the largest drop in log-likelihood are the most
   informative for the tissue's overall latent structure.

3. **Marker dependency graph** — compute pair-wise conditional likelihood
   shifts between all marker pairs and build a directed graph of
   marker-marker dependencies. Edge weight from A -> B means "removing
   marker A changes the latent likelihood of the region associated with
   marker B".

4. **Latent uncertainty decomposition** — (evidential PixelCNN only)
   decompose the PixelCNN's prediction uncertainty into aleatoric
   (inherent randomness in latent values) and epistemic (model's lack of
   knowledge about rare latent configurations).

5. **Counterfactual latent sampling** — given a real latent map Z, mask
   out spatial regions and use the PixelCNN's autoregressive sampling to
   generate plausible alternative latent completions. Decoding these
   gives counterfactual reconstructions.

Usage (CLI)::

    python -m xai.latent_likelihood \\
        --config configs/train_mambaswin_config.yaml \\
        --encoder-checkpoint checkpoints/encoder.pth \\
        --pixelcnn-checkpoint checkpoints/pixelcnn_best.pth \\
        --image path/to/sample.npy \\
        --method surprise \\
        --output-dir xai_latent/

Usage (API)::

    from xai.latent_likelihood import LatentLikelihoodXAI
    xai = LatentLikelihoodXAI(model, pixelcnn, tokenizer, ...)
    result = xai.surprise_map(image, channel_ids)
"""


from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass

import torch
import numpy as np
from torch.amp import autocast

from multiplex_model.modules.pixelcnn import LatentPixelCNN


# ------------------------------------------------------------------
# Result containers
# ------------------------------------------------------------------

@dataclass
class SurpriseResult:
    """Spatial surprise map from latent log-likelihood."""
    surprise_map: np.ndarray           # (H', W') — NLL per position
    log_likelihood_map: np.ndarray     # (H', W') — LL per position
    mean_surprise: float
    channel_surprise: np.ndarray       # (D,) — per-channel mean NLL


@dataclass
class AttributionResult:
    """Per-marker latent likelihood attribution."""
    marker_names: list[str]
    baseline_ll: float                 # LL with all markers
    ablated_ll: dict[str, float]       # marker → LL without that marker
    delta_ll: dict[str, float]         # marker → baseline - ablated (importance)

    @property
    def ranked_markers(self) -> list[tuple[str, float]]:
        """Markers sorted by importance (largest LL drop first)."""
        return sorted(self.delta_ll.items(), key=lambda x: -x[1])


@dataclass
class DependencyEdge:
    source: str
    target: str
    weight: float   # LL change in target region when source is removed


@dataclass
class DependencyGraph:
    """Directed marker dependency graph."""
    marker_names: list[str]
    edges: list[DependencyEdge]
    adjacency: np.ndarray              # (M, M) — adj[i,j] = effect of removing i on j

    def top_edges(self, k: int = 10) -> list[DependencyEdge]:
        return sorted(self.edges, key=lambda e: -abs(e.weight))[:k]


@dataclass
class UncertaintyResult:
    """Latent uncertainty decomposition (evidential PixelCNN only)."""
    aleatoric: np.ndarray              # (D, H', W')
    epistemic: np.ndarray              # (D, H', W')
    total: np.ndarray                  # (D, H', W')
    channel_aleatoric: np.ndarray      # (D,) per-channel mean
    channel_epistemic: np.ndarray      # (D,) per-channel mean


@dataclass
class CounterfactualLatentResult:
    """Counterfactual latent completions."""
    original_latent: np.ndarray        # (D, H', W')
    sampled_latents: list[np.ndarray]  # each (D, H', W')
    mask_positions: np.ndarray         # (H', W') bool
    decoded_originals: np.ndarray | None = None   # (C, H, W) if decoder used
    decoded_samples: list[np.ndarray] | None = None


# ------------------------------------------------------------------
# Core engine
# ------------------------------------------------------------------

class LatentLikelihoodXAI:
    """PixelCNN-based explainability on frozen ImmuVis latent space.

    Parameters
    ----------
    encoder_model : MultiplexAutoencoder
        Trained ImmuVis model (frozen). Used to encode inputs → latents
        and optionally decode latents → reconstructions.
    pixelcnn : LatentPixelCNN
        Trained PixelCNN density model on the latent space.
    tokenizer : dict[str, int]
        Marker name → token ID.
    device : str
        Torch device.
    """

    def __init__(
        self,
        encoder_model,
        pixelcnn: LatentPixelCNN,
        tokenizer: dict[str, int],
        device: str = "cuda",
    ):
        self.model = encoder_model.to(device).eval()
        self.pixelcnn = pixelcnn.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.pixelcnn.parameters():
            p.requires_grad = False

        self.tokenizer = tokenizer
        self.inv_tokenizer = {v: k for k, v in tokenizer.items()}
        self.device = device

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode(
        self, img: torch.Tensor, channel_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a multiplex image → latent feature map (B, D, H', W')."""
        img = img.to(self.device, dtype=torch.float32)
        channel_ids = channel_ids.to(self.device, dtype=torch.long)
        with autocast(device_type=self.device.split(":")[0], dtype=torch.bfloat16):
            z = self.model.encode(img, channel_ids)["output"]
        return z.float()

    def _ll_map(self, z: torch.Tensor) -> torch.Tensor:
        """Compute per-position log-likelihood under the PixelCNN.

        Returns (B, H', W').
        """
        return self.pixelcnn.log_likelihood(z)

    # ------------------------------------------------------------------
    # Method 1: Surprise maps
    # ------------------------------------------------------------------

    @torch.no_grad()
    def surprise_map(
        self,
        img: torch.Tensor,
        channel_ids: torch.Tensor,
    ) -> SurpriseResult:
        """Compute spatial surprise map.

        High-surprise positions have latent representations that are
        unlikely under the learned distribution — indicating unusual
        tissue structure or artefacts.

        Args:
            img: (1, C, H, W) multiplex image.
            channel_ids: (1, C) marker token IDs.

        Returns:
            SurpriseResult with spatial NLL map.
        """
        z = self._encode(img, channel_ids)              # (1, D, H', W')
        ll_map = self._ll_map(z)                         # (1, H', W')
        surprise = -ll_map                               # NLL

        # Also compute per-channel surprise
        params = self.pixelcnn(z)
        if self.pixelcnn.distribution_name == "gaussian":
            mu = params["mu"]
            log_sigma = params["log_sigma"]
            import math
            var = torch.exp(2 * log_sigma) + 1e-8
            channel_ll = -0.5 * ((z - mu) ** 2 / var + 2 * log_sigma + math.log(2 * math.pi))
            channel_nll = -channel_ll[0].mean(dim=(1, 2))  # (D,)
        elif self.pixelcnn.distribution_name == "evidential":
            gamma = params["gamma"]
            channel_nll = (z - gamma).pow(2)[0].mean(dim=(1, 2))  # proxy
        else:
            channel_nll = torch.zeros(z.shape[1])

        return SurpriseResult(
            surprise_map=surprise[0].cpu().numpy(),
            log_likelihood_map=ll_map[0].cpu().numpy(),
            mean_surprise=float(surprise.mean()),
            channel_surprise=channel_nll.cpu().numpy(),
        )

    # ------------------------------------------------------------------
    # Method 2: Conditional likelihood attribution
    # ------------------------------------------------------------------

    @torch.no_grad()
    def conditional_likelihood_attribution(
        self,
        img: torch.Tensor,
        channel_ids: torch.Tensor,
    ) -> AttributionResult:
        """Leave-one-out marker attribution via latent log-likelihood.

        For each marker channel, computes the change in latent LL when
        that marker is ablated (zeroed out) from the input.

        Args:
            img: (1, C, H, W)
            channel_ids: (1, C)

        Returns:
            AttributionResult with per-marker importance.
        """
        # Baseline: full input
        z_full = self._encode(img, channel_ids)
        ll_full = float(self._ll_map(z_full).mean())

        marker_names = [self.inv_tokenizer[cid.item()] for cid in channel_ids[0]]
        C = img.shape[1]

        ablated_ll = {}
        delta_ll = {}

        for c in range(C):
            name = marker_names[c]
            # Zero out channel c
            img_ablated = img.clone()
            img_ablated[:, c] = 0.0

            z_abl = self._encode(img_ablated, channel_ids)
            ll_abl = float(self._ll_map(z_abl).mean())

            ablated_ll[name] = ll_abl
            delta_ll[name] = ll_full - ll_abl  # positive = marker was helpful

        return AttributionResult(
            marker_names=marker_names,
            baseline_ll=ll_full,
            ablated_ll=ablated_ll,
            delta_ll=delta_ll,
        )

    # ------------------------------------------------------------------
    # Method 3: Marker dependency graph
    # ------------------------------------------------------------------

    @torch.no_grad()
    def marker_dependency_graph(
        self,
        img: torch.Tensor,
        channel_ids: torch.Tensor,
    ) -> DependencyGraph:
        """Build a directed marker dependency graph.

        For each pair (source, target), we measure how removing the
        source marker from the input affects the latent LL at spatial
        positions most associated with the target marker.

        "Most associated" is defined using the target marker's per-channel
        latent dimension — we look at the PixelCNN log-likelihood
        restricted to the corresponding latent dimensions.

        Args:
            img: (1, C, H, W)
            channel_ids: (1, C)

        Returns:
            DependencyGraph
        """
        marker_names = [self.inv_tokenizer[cid.item()] for cid in channel_ids[0]]
        C = img.shape[1]

        # Baseline latent & per-channel LL
        z_full = self._encode(img, channel_ids)
        ll_full_map = self._ll_map(z_full)  # (1, H', W')
        ll_full = float(ll_full_map.mean())

        # Cache ablated latents
        ablated_z = {}
        for c in range(C):
            img_abl = img.clone()
            img_abl[:, c] = 0.0
            ablated_z[c] = self._encode(img_abl, channel_ids)

        # Build adjacency: adj[src, tgt] = LL_full(tgt-dims) - LL_ablated_src(tgt-dims)
        D = z_full.shape[1]
        # Approximate: split latent dims evenly across channels
        dims_per_channel = D // C
        remainder = D % C

        adjacency = np.zeros((C, C), dtype=np.float32)
        edges = []

        for src in range(C):
            z_abl = ablated_z[src]
            for tgt in range(C):
                # Slice of latent dims associated with target channel
                tgt_start = tgt * dims_per_channel + min(tgt, remainder)
                tgt_end = tgt_start + dims_per_channel + (1 if tgt < remainder else 0)

                # Compute local LL on these dims
                params_full = self.pixelcnn(z_full)
                params_abl = self.pixelcnn(z_abl)

                if self.pixelcnn.distribution_name == "gaussian":
                    import math
                    # Full
                    mu_f = params_full["mu"][:, tgt_start:tgt_end]
                    ls_f = params_full["log_sigma"][:, tgt_start:tgt_end]
                    z_slice = z_full[:, tgt_start:tgt_end]
                    var_f = torch.exp(2 * ls_f) + 1e-8
                    ll_f = (-0.5 * ((z_slice - mu_f)**2 / var_f + 2 * ls_f
                                    + math.log(2 * math.pi))).mean()

                    # Ablated
                    mu_a = params_abl["mu"][:, tgt_start:tgt_end]
                    ls_a = params_abl["log_sigma"][:, tgt_start:tgt_end]
                    z_abl_slice = z_abl[:, tgt_start:tgt_end]
                    var_a = torch.exp(2 * ls_a) + 1e-8
                    ll_a = (-0.5 * ((z_abl_slice - mu_a)**2 / var_a + 2 * ls_a
                                    + math.log(2 * math.pi))).mean()
                else:
                    # Generic: use overall LL as proxy
                    ll_f = self._ll_map(z_full).mean()
                    ll_a = self._ll_map(z_abl).mean()

                weight = float(ll_f - ll_a)
                adjacency[src, tgt] = weight

                if abs(weight) > 1e-6:
                    edges.append(DependencyEdge(
                        source=marker_names[src],
                        target=marker_names[tgt],
                        weight=weight,
                    ))

        return DependencyGraph(
            marker_names=marker_names,
            edges=edges,
            adjacency=adjacency,
        )

    # ------------------------------------------------------------------
    # Method 4: Latent uncertainty decomposition
    # ------------------------------------------------------------------

    @torch.no_grad()
    def latent_uncertainty(
        self,
        img: torch.Tensor,
        channel_ids: torch.Tensor,
    ) -> UncertaintyResult:
        """Decompose PixelCNN latent uncertainty (evidential head only).

        Aleatoric uncertainty captures inherent variability in the latent
        space at each position. Epistemic uncertainty captures the model's
        lack of knowledge about rare latent configurations.

        Args:
            img: (1, C, H, W)
            channel_ids: (1, C)

        Returns:
            UncertaintyResult with aleatoric / epistemic decomposition.
        """
        z = self._encode(img, channel_ids)
        unc = self.pixelcnn.latent_uncertainty(z)

        aleatoric = unc["aleatoric"][0].cpu().numpy()   # (D, H', W')
        epistemic = unc["epistemic"][0].cpu().numpy()
        total = aleatoric + epistemic

        return UncertaintyResult(
            aleatoric=aleatoric,
            epistemic=epistemic,
            total=total,
            channel_aleatoric=aleatoric.mean(axis=(1, 2)),
            channel_epistemic=epistemic.mean(axis=(1, 2)),
        )

    # ------------------------------------------------------------------
    # Method 5: Counterfactual latent sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def counterfactual_sampling(
        self,
        img: torch.Tensor,
        channel_ids: torch.Tensor,
        mask_region: np.ndarray | None = None,
        n_samples: int = 5,
        decode: bool = False,
    ) -> CounterfactualLatentResult:
        """Generate counterfactual latent completions.

        Masks out a spatial region of the latent map and uses the
        PixelCNN's autoregressive sampling to fill it in with plausible
        alternative values. Optionally decodes back to image space.

        Args:
            img: (1, C, H, W)
            channel_ids: (1, C)
            mask_region: (H', W') bool array — True = regenerate.
                If None, masks the bottom-right quadrant.
            n_samples: Number of counterfactual samples.
            decode: If True, decode latent → image using the full
                autoencoder (requires decoder in the model).

        Returns:
            CounterfactualLatentResult
        """
        z = self._encode(img, channel_ids)  # (1, D, H', W')
        _, D, Hp, Wp = z.shape

        if mask_region is None:
            # Default: mask bottom-right quadrant
            mask_region = np.zeros((Hp, Wp), dtype=bool)
            mask_region[Hp // 2:, Wp // 2:] = True

        original_z = z[0].cpu().numpy()
        sampled_latents = []
        decoded_samples = [] if decode else None

        for _ in range(n_samples):
            z_ctx = z.clone()
            mask_t = torch.from_numpy(mask_region).to(self.device)
            # Set masked positions to NaN — PixelCNN.sample() interprets NaN as "generate"
            z_ctx[0, :, mask_t] = float("nan")
            z_sampled = self.pixelcnn.sample(z_ctx)
            sampled_latents.append(z_sampled[0].cpu().numpy())

            if decode:
                with autocast(device_type=self.device.split(":")[0], dtype=torch.bfloat16):
                    recon = self.model.decode(z_sampled, channel_ids)
                decoded_samples.append(recon[0].float().cpu().numpy())

        decoded_originals = None
        if decode:
            with autocast(device_type=self.device.split(":")[0], dtype=torch.bfloat16):
                decoded_originals = self.model.decode(z, channel_ids)
            decoded_originals = decoded_originals[0].float().cpu().numpy()

        return CounterfactualLatentResult(
            original_latent=original_z,
            sampled_latents=sampled_latents,
            mask_positions=mask_region,
            decoded_originals=decoded_originals,
            decoded_samples=decoded_samples,
        )


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PixelCNN-based latent space XAI for ImmuVis models"
    )
    parser.add_argument("--config", required=True, help="ImmuVis training config YAML")
    parser.add_argument("--encoder-checkpoint", required=True)
    parser.add_argument("--pixelcnn-checkpoint", required=True)
    parser.add_argument("--image", required=True, help="Path to input .npy image")
    parser.add_argument("--method", required=True,
                        choices=["surprise", "attribution", "dependency",
                                 "uncertainty", "counterfactual"])
    parser.add_argument("--distribution", default="evidential",
                        choices=["gaussian", "discretized_logistic_mixture", "evidential"])
    parser.add_argument("--pixelcnn-hidden", type=int, default=256)
    parser.add_argument("--pixelcnn-layers", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Number of counterfactual samples")
    parser.add_argument("--decode", action="store_true",
                        help="Decode counterfactual latents back to image space")
    parser.add_argument("--output-dir", default="xai_latent")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from ruamel.yaml import YAML as RuamelYAML
    from multiplex_model.modules import MultiplexAutoencoder
    from multiplex_model.utils.configuration import TrainingConfig

    yaml_loader = RuamelYAML(typ="safe")
    with open(args.config) as f:
        raw_config = yaml_loader.load(f)
    config = TrainingConfig(**raw_config)

    TOKENIZER = RuamelYAML().load(open(config.tokenizer_config))
    num_channels = len(TOKENIZER)

    # Load encoder
    decoder_cfg = config.decoder_config.model_dump()
    decoder_cfg["num_outputs"] = 4 if config.uncertainty_method == "evidential" else 2
    model = MultiplexAutoencoder(
        num_channels=num_channels,
        encoder_config=config.encoder_config.model_dump(),
        decoder_config=decoder_cfg,
    )
    ckpt = torch.load(args.encoder_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))

    # Load PixelCNN
    latent_dim = config.encoder_config.model_dump().get("pm_embedding_dims", [768])[-1]
    pixelcnn = LatentPixelCNN(
        latent_dim=latent_dim,
        hidden_dim=args.pixelcnn_hidden,
        n_layers=args.pixelcnn_layers,
        distribution=args.distribution,
    )
    pcnn_ckpt = torch.load(args.pixelcnn_checkpoint, map_location="cpu", weights_only=True)
    pixelcnn.load_state_dict(pcnn_ckpt.get("pixelcnn_state_dict", pcnn_ckpt))

    # Build XAI engine
    xai = LatentLikelihoodXAI(model, pixelcnn, TOKENIZER, device=args.device)

    # Load image
    img_np = np.load(args.image)
    if img_np.ndim == 2:
        img_np = img_np[np.newaxis]
    img = torch.from_numpy(img_np).unsqueeze(0).float()

    # Determine channel IDs from tokenizer (assume image channels match panel order)
    panel_markers = list(TOKENIZER.keys())[:img.shape[1]]
    channel_ids = torch.tensor(
        [[TOKENIZER[m] for m in panel_markers]], dtype=torch.long,
    )

    # Run selected method
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "surprise":
        result = xai.surprise_map(img, channel_ids)
        np.save(out_dir / "surprise_map.npy", result.surprise_map)
        np.save(out_dir / "log_likelihood_map.npy", result.log_likelihood_map)
        print(f"Mean surprise: {result.mean_surprise:.4f}")
        print(f"Saved surprise maps to {out_dir}/")

    elif args.method == "attribution":
        result = xai.conditional_likelihood_attribution(img, channel_ids)
        print(f"Baseline LL: {result.baseline_ll:.4f}")
        print("Marker importance (ΔLL, descending):")
        for name, delta in result.ranked_markers:
            print(f"  {name:>20s}: {delta:+.4f}")
        np.savez(out_dir / "attribution.npz",
                 delta_ll={k: v for k, v in result.delta_ll.items()})

    elif args.method == "dependency":
        result = xai.marker_dependency_graph(img, channel_ids)
        np.save(out_dir / "adjacency.npy", result.adjacency)
        print(f"Top dependency edges:")
        for e in result.top_edges(10):
            print(f"  {e.source:>15s} → {e.target:<15s}: {e.weight:+.4f}")

    elif args.method == "uncertainty":
        result = xai.latent_uncertainty(img, channel_ids)
        np.save(out_dir / "aleatoric.npy", result.aleatoric)
        np.save(out_dir / "epistemic.npy", result.epistemic)
        print(f"Mean aleatoric: {result.channel_aleatoric.mean():.4f}")
        print(f"Mean epistemic: {result.channel_epistemic.mean():.4f}")

    elif args.method == "counterfactual":
        result = xai.counterfactual_sampling(
            img, channel_ids,
            n_samples=args.n_samples,
            decode=args.decode,
        )
        for i, s in enumerate(result.sampled_latents):
            np.save(out_dir / f"counterfactual_latent_{i}.npy", s)
        if result.decoded_samples:
            for i, d in enumerate(result.decoded_samples):
                np.save(out_dir / f"counterfactual_decoded_{i}.npy", d)
        print(f"Saved {len(result.sampled_latents)} counterfactual samples to {out_dir}/")

    print("Done.")


if __name__ == "__main__":
    main()
