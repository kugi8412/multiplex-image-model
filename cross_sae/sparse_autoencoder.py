"""Sparse autoencoder architectures for dictionary learning on latent spaces.

Three variants are provided:

1. **VanillaSAE** — standard overcomplete autoencoder with L1 penalty on
   the hidden (feature) activations.  Simple, well-understood, but the
   L1 coefficient requires careful tuning.

2. **TopKSAE** — instead of a soft L1 penalty, keeps only the top-k
   activations per sample and zeros the rest (Gao et al., 2024).
   Sparsity level is exact and controllable.

3. **GatedSAE** — uses a parallel gating network to decide which features
   fire, decoupling the magnitude and selection mechanisms
   (Rajamanoharan et al., 2024).  Better reconstruction at the same
   sparsity level.

All variants share:
    • Linear encoder:  h = act(W_enc @ (x − b_dec) + b_enc)
    • Linear decoder:  x̂ = W_dec @ h + b_dec
    • Decoder columns are unit-normalised after each step.
    • Reconstruction loss = MSE(x, x̂)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEOutput:
    """Container for SAE forward pass results."""
    x_hat: torch.Tensor       # Reconstructed input
    features: torch.Tensor    # Sparse feature activations (hidden layer)
    loss: torch.Tensor        # Total loss (reconstruction + sparsity)
    mse_loss: torch.Tensor    # Reconstruction loss only
    sparsity_loss: torch.Tensor  # Sparsity penalty only
    l0: torch.Tensor          # Mean L0 (number of nonzero features per sample)


class VanillaSAE(nn.Module):
    """Sparse autoencoder with L1 sparsity penalty.

    Args:
        input_dim: Dimensionality of latent vectors (e.g. 768 for ImmuVis).
        hidden_dim: Number of dictionary features (overcomplete, e.g. 768*8).
        l1_coeff: Weight for L1 penalty on feature activations.
        tied_weights: If True, decoder weight = encoder weight transposed
            (reduces parameters by half, but may limit expressiveness).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        l1_coeff: float = 1e-3,
        tied_weights: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.l1_coeff = l1_coeff
        self.tied_weights = tied_weights

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.b_dec = nn.Parameter(torch.zeros(input_dim))

        if not tied_weights:
            self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        else:
            self.decoder = None

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        if self.decoder is not None:
            nn.init.kaiming_uniform_(self.decoder.weight)

    @property
    def W_dec(self) -> torch.Tensor:
        if self.tied_weights:
            return self.encoder.weight.t()
        return self.decoder.weight

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x - self.b_dec))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        if self.tied_weights:
            return F.linear(h, self.encoder.weight.t()) + self.b_dec
        return self.decoder(h) + self.b_dec

    def forward(self, x: torch.Tensor) -> SAEOutput:
        h = self.encode(x)
        x_hat = self.decode(h)

        mse_loss = F.mse_loss(x_hat, x)
        sparsity_loss = self.l1_coeff * h.abs().mean()
        loss = mse_loss + sparsity_loss
        l0 = (h > 0).float().sum(dim=-1).mean()

        return SAEOutput(
            x_hat=x_hat, features=h, loss=loss,
            mse_loss=mse_loss, sparsity_loss=sparsity_loss, l0=l0,
        )

    @torch.no_grad()
    def normalise_decoder(self):
        """Project decoder columns to unit norm (prevents feature shrinkage)."""
        if self.tied_weights:
            return
        w = self.decoder.weight.data
        self.decoder.weight.data = w / (w.norm(dim=0, keepdim=True) + 1e-8)


class TopKSAE(nn.Module):
    """Sparse autoencoder with exact top-k sparsity.

    Instead of an L1 penalty, only the top-k activations are kept per
    sample.  This gives exact control over the sparsity level and avoids
    the L1 coefficient tuning problem.

    Args:
        input_dim: Dimensionality of latent vectors.
        hidden_dim: Number of dictionary features (overcomplete).
        k: Number of features to keep active per sample.
        aux_k: Number of extra features for the auxiliary reconstruction
            loss (prevents dead features). Set to 0 to disable.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        k: int = 32,
        aux_k: int = 128,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.aux_k = aux_k

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(input_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_acts = self.encoder(x - self.b_dec)
        # Keep only top-k activations
        topk_vals, topk_idx = pre_acts.topk(self.k, dim=-1)
        h = torch.zeros_like(pre_acts)
        h.scatter_(-1, topk_idx, F.relu(topk_vals))
        return h

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(h) + self.b_dec

    def forward(self, x: torch.Tensor) -> SAEOutput:
        pre_acts = self.encoder(x - self.b_dec)
        topk_vals, topk_idx = pre_acts.topk(self.k, dim=-1)

        h = torch.zeros_like(pre_acts)
        h.scatter_(-1, topk_idx, F.relu(topk_vals))

        x_hat = self.decode(h)
        mse_loss = F.mse_loss(x_hat, x)

        # Auxiliary loss on dead features to prevent collapse
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.training and self.aux_k > 0:
            # Mask out top-k features, find top-aux_k among the rest
            dead_pre = pre_acts.clone()
            dead_pre.scatter_(-1, topk_idx, float("-inf"))
            aux_topk_vals, aux_topk_idx = dead_pre.topk(self.aux_k, dim=-1)
            h_aux = torch.zeros_like(pre_acts)
            h_aux.scatter_(-1, aux_topk_idx, F.relu(aux_topk_vals))
            x_hat_aux = self.decode(h_aux)
            aux_loss = F.mse_loss(x_hat_aux, x)

        loss = mse_loss + aux_loss
        l0 = (h > 0).float().sum(dim=-1).mean()

        return SAEOutput(
            x_hat=x_hat, features=h, loss=loss,
            mse_loss=mse_loss, sparsity_loss=aux_loss, l0=l0,
        )

    @torch.no_grad()
    def normalise_decoder(self):
        w = self.decoder.weight.data
        self.decoder.weight.data = w / (w.norm(dim=0, keepdim=True) + 1e-8)


class GatedSAE(nn.Module):
    """Gated sparse autoencoder.

    A parallel gating network decides which features fire (binary gate),
    while the magnitude network determines their activation values.  This
    decouples selection from magnitude estimation.

    Args:
        input_dim: Dimensionality of latent vectors.
        hidden_dim: Number of dictionary features (overcomplete).
        l1_coeff: Weight for L1 penalty on the gated pre-activations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        l1_coeff: float = 1e-3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.l1_coeff = l1_coeff

        # Magnitude path
        self.W_mag = nn.Linear(input_dim, hidden_dim)
        # Gate path (shared direction, separate bias)
        self.b_gate = nn.Parameter(torch.zeros(hidden_dim))
        self.r_mag = nn.Parameter(torch.ones(hidden_dim))

        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(input_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_mag.weight)
        nn.init.zeros_(self.W_mag.bias)
        nn.init.kaiming_uniform_(self.decoder.weight)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.b_dec
        # Gate: binary decision which features fire
        gate_pre = F.linear(x_centered, self.W_mag.weight, self.b_gate)
        gate = (gate_pre > 0).float()
        # Magnitude
        mag_pre = self.W_mag(x_centered)
        mag = F.relu(mag_pre * self.r_mag)
        return gate * mag

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(h) + self.b_dec

    def forward(self, x: torch.Tensor) -> SAEOutput:
        x_centered = x - self.b_dec

        gate_pre = F.linear(x_centered, self.W_mag.weight, self.b_gate)
        gate = (gate_pre > 0).float()
        mag_pre = self.W_mag(x_centered)
        mag = F.relu(mag_pre * self.r_mag)
        h = gate * mag

        x_hat = self.decode(h)
        mse_loss = F.mse_loss(x_hat, x)

        # L1 on the gate pre-activations (encourages the gate itself to be sparse)
        sparsity_loss = self.l1_coeff * F.relu(gate_pre).mean()
        loss = mse_loss + sparsity_loss
        l0 = (h > 0).float().sum(dim=-1).mean()

        return SAEOutput(
            x_hat=x_hat, features=h, loss=loss,
            mse_loss=mse_loss, sparsity_loss=sparsity_loss, l0=l0,
        )

    @torch.no_grad()
    def normalise_decoder(self):
        w = self.decoder.weight.data
        self.decoder.weight.data = w / (w.norm(dim=0, keepdim=True) + 1e-8)


# ===================================================================
# Factory
# ===================================================================

SAE_REGISTRY = {
    "vanilla": VanillaSAE,
    "topk": TopKSAE,
    "gated": GatedSAE,
}


def build_sae(
    variant: str,
    input_dim: int,
    hidden_dim: int,
    **kwargs,
) -> nn.Module:
    """Build a sparse autoencoder from a variant name.

    Args:
        variant: One of 'vanilla', 'topk', 'gated'.
        input_dim: Latent vector dimensionality.
        hidden_dim: Dictionary size (overcomplete, typically 4–16× input_dim).
        **kwargs: Variant-specific parameters (l1_coeff, k, aux_k, etc.).

    Returns:
        An nn.Module with .encode(), .decode(), and forward() → SAEOutput.
    """
    if variant not in SAE_REGISTRY:
        raise ValueError(
            f"Unknown SAE variant '{variant}'. Choose from: {list(SAE_REGISTRY)}"
        )
    cls = SAE_REGISTRY[variant]
    return cls(input_dim=input_dim, hidden_dim=hidden_dim, **kwargs)
