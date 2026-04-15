"""PixelCNN for autoregressive density estimation on frozen latent spaces.

Models the distribution p(Z) over latent feature maps Z ∈ R^{D×H×W}
produced by a frozen ImmuVis encoder.  Uses gated masked convolutions
(van den Oord et al., 2016) with vertical + horizontal stacks for
proper causal ordering over raster scan.

Three output distribution heads are supported:

    ``gaussian``
        Predicts (μ, log σ) per spatial position per channel.
        Loss = standard Gaussian NLL.

    ``discretized_logistic_mixture``
        Mixture of K logistics per channel (PixelCNN++ style).
        Best density estimation but slower at high K.

    ``evidential``
        Predicts (γ, ν, α, β) per position per channel → NIG prior.
        Provides aleatoric + epistemic uncertainty decomposition.
        Matches the uncertainty framework in the main ImmuVis model.

The model operates on the *spatial* dimensions of the latent map.
At each position (i, j), it predicts the full D-channel vector
conditioned on all previously scanned positions (raster order).

Architecture
------------
::

    Input Z: (B, D, H, W)
        ↓
    Vertical stack (masked conv, sees above)
        +
    Horizontal stack (masked conv, sees left + vertical context)
        ↓  × N residual layers
    Output head → distribution parameters (B, K*D, H, W)
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================================================================
# 1. Masked Convolutions
# ===================================================================

class MaskedConv2d(nn.Conv2d):
    """Conv2d with a causal mask applied to the weight tensor.

    ``mask_type='A'``: strict — the center pixel is excluded (first layer).
    ``mask_type='B'``: relaxed — the center pixel is included (subsequent layers).
    ``direction='vertical'``: masks bottom half of the kernel.
    ``direction='horizontal'``: masks right side of center row.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        direction: Literal["vertical", "horizontal"] = "horizontal",
        mask_type: Literal["A", "B"] = "B",
        **kwargs,
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        kwargs["padding"] = (kernel_size[0] // 2, kernel_size[1] // 2)
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

        self.direction = direction
        self.mask_type = mask_type

        # Build mask
        mask = torch.ones_like(self.weight.data)
        kH, kW = kernel_size
        cH, cW = kH // 2, kW // 2

        if direction == "vertical":
            # See only rows above (and center row excluded for type A)
            mask[:, :, cH + 1:, :] = 0
            if mask_type == "A":
                mask[:, :, cH, :] = 0
        else:  # horizontal
            # See only current row, positions to the left
            mask[:, :, cH + 1:, :] = 0
            mask[:, :, :cH, :] = 0
            mask[:, :, cH, cW + 1:] = 0
            if mask_type == "A":
                mask[:, :, cH, cW] = 0

        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask
        return super().forward(x)


# ===================================================================
# 2. Gated Residual Block
# ===================================================================

class GatedResidualBlock(nn.Module):
    """Gated PixelCNN block with vertical + horizontal stacks.

    Vertical stack sees everything above the current row.
    Horizontal stack sees everything to the left in the current row,
    plus a feed-forward connection from the vertical stack.
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        # Vertical stack: 2× channels for tanh/sigmoid gating
        self.v_conv = MaskedConv2d(
            channels, 2 * channels, kernel_size,
            direction="vertical", mask_type="B",
        )
        self.v_to_h = nn.Conv2d(2 * channels, 2 * channels, kernel_size=1)

        # Horizontal stack: 1×K kernel (only current row)
        self.h_conv = MaskedConv2d(
            channels, 2 * channels, (1, kernel_size),
            direction="horizontal", mask_type="B",
        )

        # Residual projection
        self.h_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(
        self,
        v_input: torch.Tensor,
        h_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            v_input: (B, C, H, W) vertical stack input
            h_input: (B, C, H, W) horizontal stack input

        Returns:
            (v_out, h_out) — both (B, C, H, W)
        """
        # Vertical stack
        v = self.v_conv(v_input)
        v_tanh, v_sig = v.chunk(2, dim=1)
        v_out = torch.tanh(v_tanh) * torch.sigmoid(v_sig)

        # Horizontal stack with vertical context
        h = self.h_conv(h_input) + self.v_to_h(v)
        h_tanh, h_sig = h.chunk(2, dim=1)
        h_out = torch.tanh(h_tanh) * torch.sigmoid(h_sig)

        # Residual
        h_out = self.h_proj(h_out) + h_input

        return v_out, h_out


# ===================================================================
# 3. Output Distribution Heads
# ===================================================================

class GaussianHead(nn.Module):
    """Predicts (mu, log_sigma) per channel per spatial position."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=1)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.proj(x)
        mu, log_sigma = out.chunk(2, dim=1)
        return {"mu": mu, "log_sigma": log_sigma}

    def nll(self, params: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        """Gaussian negative log-likelihood."""
        mu = params["mu"]
        log_sigma = params["log_sigma"]
        var = torch.exp(2 * log_sigma) + 1e-8
        nll = 0.5 * ((target - mu) ** 2 / var + 2 * log_sigma + math.log(2 * math.pi))
        return nll.mean()

    def sample(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        mu = params["mu"]
        sigma = torch.exp(params["log_sigma"])
        return mu + sigma * torch.randn_like(mu)


class DiscretizedLogisticMixtureHead(nn.Module):
    """Mixture of K discretized logistics per channel (PixelCNN++ style).

    For continuous latent values, we use the logistic CDF difference
    on small bins centred at the target value (bin width 1/255 by default,
    configurable).
    """

    def __init__(self, in_channels: int, out_channels: int, n_mixtures: int = 5,
                 bin_width: float = 0.01):
        super().__init__()
        self.n_mixtures = n_mixtures
        self.out_channels = out_channels
        self.bin_width = bin_width
        # Per mixture: logit_weight + mu + log_scale = 3 params
        self.proj = nn.Conv2d(in_channels, 3 * n_mixtures * out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.proj(x)
        B, _, H, W = out.shape
        K, D = self.n_mixtures, self.out_channels
        out = out.reshape(B, 3, K, D, H, W)
        return {
            "logit_weights": out[:, 0],  # (B, K, D, H, W)
            "mu": out[:, 1],
            "log_scale": out[:, 2],
        }

    def nll(self, params: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        """Discretized logistic mixture NLL."""
        logit_w = params["logit_weights"]  # (B, K, D, H, W)
        mu = params["mu"]
        log_s = params["log_scale"].clamp(min=-7.0)

        # target: (B, D, H, W) → (B, 1, D, H, W)
        t = target.unsqueeze(1)
        inv_s = torch.exp(-log_s)

        half_bin = self.bin_width / 2.0
        cdf_plus = torch.sigmoid((t - mu + half_bin) * inv_s)
        cdf_minus = torch.sigmoid((t - mu - half_bin) * inv_s)
        log_probs = torch.log((cdf_plus - cdf_minus).clamp(min=1e-12))

        # Log-sum-exp over mixture components
        log_w = F.log_softmax(logit_w, dim=1)
        log_p = torch.logsumexp(log_w + log_probs, dim=1)  # (B, D, H, W)
        return -log_p.mean()

    def sample(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        logit_w = params["logit_weights"]
        mu = params["mu"]
        log_s = params["log_scale"]

        # Sample mixture component
        w = F.softmax(logit_w, dim=1)
        B, K, D, H, W = w.shape
        w_flat = w.permute(0, 2, 3, 4, 1).reshape(-1, K)
        idx = torch.multinomial(w_flat, 1).squeeze(-1)

        mu_flat = mu.permute(0, 2, 3, 4, 1).reshape(-1, K)
        ls_flat = log_s.permute(0, 2, 3, 4, 1).reshape(-1, K)

        chosen_mu = mu_flat[torch.arange(len(idx)), idx]
        chosen_ls = ls_flat[torch.arange(len(idx)), idx]
        scale = torch.exp(chosen_ls)

        u = torch.rand_like(chosen_mu).clamp(1e-5, 1 - 1e-5)
        sample = chosen_mu + scale * (torch.log(u) - torch.log(1 - u))
        return sample.reshape(B, D, H, W)


class EvidentialHead(nn.Module):
    """NIG evidential head: predicts (γ, ν, α, β) per channel per position.

    Provides decomposed aleatoric + epistemic uncertainty on the latent space.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, 4 * out_channels, kernel_size=1)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.proj(x)
        gamma, nu_raw, alpha_raw, beta_raw = out.chunk(4, dim=1)
        nu = F.softplus(nu_raw) + 1e-6
        alpha = F.softplus(alpha_raw) + 1.0 + 1e-6
        beta = F.softplus(beta_raw) + 1e-6
        return {"gamma": gamma, "nu": nu, "alpha": alpha, "beta": beta}

    def nll(self, params: dict[str, torch.Tensor], target: torch.Tensor,
            reg_coeff: float = 0.01) -> torch.Tensor:
        """NIG negative log-marginal-likelihood (Student-t) + evidence regularizer."""
        gamma = params["gamma"]
        nu = params["nu"]
        alpha = params["alpha"]
        beta = params["beta"]

        error = target - gamma
        omega = 2.0 * beta * (1.0 + nu)
        nll = (
            0.5 * torch.log(torch.pi / (nu + 1e-8))
            - alpha * torch.log(omega + 1e-8)
            + (alpha + 0.5) * torch.log(error.pow(2) * nu + omega + 1e-8)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )
        reg = error.abs() * (2.0 * nu + alpha)
        return (nll + reg_coeff * reg).mean()

    def sample(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Sample from the NIG predictive (Student-t)."""
        gamma = params["gamma"]
        nu = params["nu"]
        alpha = params["alpha"]
        beta = params["beta"]
        # Student-t(2α) with loc=γ, scale=β(1+ν)/(να)
        alpha_m1 = alpha - 1.0 + 1e-8
        scale = torch.sqrt(beta * (1.0 + nu) / (nu * alpha_m1 + 1e-8))
        # Approximate: sample Gaussian with matched variance
        return gamma + scale * torch.randn_like(gamma)

    def uncertainty(self, params: dict[str, torch.Tensor]):
        """Decompose into aleatoric/epistemic."""
        nu, alpha, beta = params["nu"], params["alpha"], params["beta"]
        alpha_m1 = alpha - 1.0 + 1e-8
        aleatoric = beta / alpha_m1
        epistemic = beta / (nu * alpha_m1)
        return aleatoric, epistemic


# ===================================================================
# 4. PixelCNN Model
# ===================================================================

class LatentPixelCNN(nn.Module):
    """PixelCNN for autoregressive density estimation on latent feature maps.

    Input shape: ``(B, D, H, W)`` — frozen encoder output.
    Output: distribution parameters at each spatial position, conditioned
    on all positions that come before in raster scan order.

    Parameters
    ----------
    latent_dim : int
        Number of channels in the latent feature map (D).
    hidden_dim : int
        Internal channel count for the residual blocks.
    n_layers : int
        Number of gated residual blocks.
    kernel_size : int
        Convolution kernel size (odd).
    distribution : str
        ``'gaussian'``, ``'discretized_logistic_mixture'``, or ``'evidential'``.
    n_mixtures : int
        Number of logistic mixture components (only for DLM).
    evidence_reg_coeff : float
        Regularization for evidential head.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 8,
        kernel_size: int = 3,
        distribution: str = "evidential",
        n_mixtures: int = 5,
        evidence_reg_coeff: float = 0.01,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.distribution_name = distribution
        self.evidence_reg_coeff = evidence_reg_coeff

        # Initial projection (type A mask — excludes center pixel)
        self.v_input = MaskedConv2d(
            latent_dim, hidden_dim, kernel_size,
            direction="vertical", mask_type="A",
        )
        self.h_input = MaskedConv2d(
            latent_dim, hidden_dim, (1, kernel_size),
            direction="horizontal", mask_type="A",
        )

        # Residual blocks
        self.layers = nn.ModuleList([
            GatedResidualBlock(hidden_dim, kernel_size)
            for _ in range(n_layers)
        ])

        # Output projection
        self.out_proj = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ELU(),
        )

        # Distribution head
        if distribution == "gaussian":
            self.head = GaussianHead(hidden_dim, latent_dim)
        elif distribution == "discretized_logistic_mixture":
            self.head = DiscretizedLogisticMixtureHead(
                hidden_dim, latent_dim, n_mixtures=n_mixtures,
            )
        elif distribution == "evidential":
            self.head = EvidentialHead(hidden_dim, latent_dim)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass: compute distribution parameters at each position.

        Args:
            z: (B, D, H, W) latent feature map.

        Returns:
            dict of distribution parameters (shapes depend on head type).
        """
        v = self.v_input(z)
        h = self.h_input(z)

        for layer in self.layers:
            v, h = layer(v, h)

        out = self.out_proj(h)
        return self.head(out)

    def loss(self, z: torch.Tensor) -> torch.Tensor:
        """Compute negative log-likelihood loss.

        Args:
            z: (B, D, H, W) target latent features.
        """
        params = self.forward(z)
        if self.distribution_name == "evidential":
            return self.head.nll(params, z, reg_coeff=self.evidence_reg_coeff)
        return self.head.nll(params, z)

    @torch.no_grad()
    def log_likelihood(self, z: torch.Tensor) -> torch.Tensor:
        """Compute per-position log-likelihood (negative NLL per pixel).

        Returns:
            (B, H, W) log-likelihood map (higher = more expected).
        """
        params = self.forward(z)
        if self.distribution_name == "gaussian":
            mu = params["mu"]
            log_sigma = params["log_sigma"]
            var = torch.exp(2 * log_sigma) + 1e-8
            ll = -0.5 * ((z - mu) ** 2 / var + 2 * log_sigma + math.log(2 * math.pi))
            return ll.mean(dim=1)  # average over D → (B, H, W)
        elif self.distribution_name == "discretized_logistic_mixture":
            logit_w = params["logit_weights"]
            mu = params["mu"]
            log_s = params["log_scale"].clamp(min=-7.0)
            t = z.unsqueeze(1)
            inv_s = torch.exp(-log_s)
            hw = self.head.bin_width / 2.0
            cdf_p = torch.sigmoid((t - mu + hw) * inv_s)
            cdf_m = torch.sigmoid((t - mu - hw) * inv_s)
            log_probs = torch.log((cdf_p - cdf_m).clamp(min=1e-12))
            log_w = F.log_softmax(logit_w, dim=1)
            ll = torch.logsumexp(log_w + log_probs, dim=1)
            return ll.mean(dim=1)
        else:  # evidential
            gamma = params["gamma"]
            nu = params["nu"]
            alpha = params["alpha"]
            beta = params["beta"]
            error = z - gamma
            omega = 2.0 * beta * (1.0 + nu)
            ll = -(
                0.5 * torch.log(torch.pi / (nu + 1e-8))
                - alpha * torch.log(omega + 1e-8)
                + (alpha + 0.5) * torch.log(error.pow(2) * nu + omega + 1e-8)
                + torch.lgamma(alpha)
                - torch.lgamma(alpha + 0.5)
            )
            return ll.mean(dim=1)

    @torch.no_grad()
    def sample(self, z_context: torch.Tensor = None,
               shape: tuple = None, device: str = "cuda") -> torch.Tensor:
        """Autoregressive sampling.

        If ``z_context`` is given, uses it as partial context and re-generates
        the masked positions. If not, generates from scratch.

        Args:
            z_context: (B, D, H, W) optional partial context (NaN = generate).
            shape: (B, D, H, W) shape for unconditional generation.
            device: torch device.

        Returns:
            (B, D, H, W) sampled latent map.
        """
        if z_context is not None:
            z = z_context.clone()
            mask = torch.isnan(z[:, 0])  # (B, H, W) positions to generate
        else:
            assert shape is not None
            z = torch.zeros(shape, device=device)
            mask = torch.ones(shape[0], shape[2], shape[3],
                              dtype=torch.bool, device=device)

        B, D, H, W = z.shape

        for i in range(H):
            for j in range(W):
                if not mask[:, i, j].any():
                    continue
                params = self.forward(z)
                sample = self.head.sample(params)
                # Only fill positions that need generation
                batch_mask = mask[:, i, j]
                z[batch_mask, :, i, j] = sample[batch_mask, :, i, j]

        return z

    @torch.no_grad()
    def latent_uncertainty(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Get per-position uncertainty decomposition (evidential head only).

        Returns dict with 'aleatoric' and 'epistemic', each (B, D, H, W).
        """
        if self.distribution_name != "evidential":
            raise ValueError("Uncertainty decomposition requires evidential head")
        params = self.forward(z)
        aleatoric, epistemic = self.head.uncertainty(params)
        return {"aleatoric": aleatoric, "epistemic": epistemic}
