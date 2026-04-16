#!/usr/bin/env python
# -*- coding: utf-8 -*-
# losses.py


import torch
import torch.nn as nn


def nll_loss(x, mi, logvar):
    return torch.mean((x - mi) ** 2 / (torch.exp(logvar) + 1e-8) + logvar)


def beta_nll_loss(x, mi, logvar, beta=1.0):
    sg_var_beta = logvar.detach().exp().pow(beta)
    nll = (x - mi) ** 2 / (torch.exp(logvar) + 1e-8) + logvar
    beta_nll = sg_var_beta * nll
    return torch.mean(beta_nll)


# -------------------------------------------------------------------
# Evidential Deep Regression (Normal-Inverse-Gamma)
# The model predicts 4 parameters per pixel:
#   (gamma)  — predicted mean
#   (nu)     — evidence (virtual observation count, > 0)
#   (alpha)  — Inverse-Gamma shape (> 1 for finite variance)
#   (beta)   — Inverse-Gamma rate (> 0)
#
# These parameterize a Normal-Inverse-Gamma prior:
#   sigma^2 ~ Inv-Gamma(alpha, beta)           — aleatoric uncertainty
#   mu | sigma^2 ~ Normal(gamma, sigma^2/nu)       — epistemic uncertainty
#
# Posterior predictive is Student-t(2alpha) with:
#   mean = gamma
#   var  = beta(1 + nu) / (nu*alpha)
#
# Uncertainty decomposition:
#   Aleatoric  = beta / (alpha - 1)          — irreducible data noise
#   Epistemic  = beta / (nu(alpha - 1))       — model uncertainty (shrinks with evidence)
#   Total      = Aleatoric + Epistemic = beta(1 + nu) / (nu(alpha - 1))
#
# -------------------------------------------------------------------


def evidential_loss(x, gamma, nu, alpha, beta, reg_coeff=0.01):
    """Normal-Inverse-Gamma evidential regression loss.

    Combines the negative log-marginal-likelihood of the Student-t predictive
    distribution with an evidence regularizer that penalizes high evidence
    (confidence) on incorrect predictions.

    Args:
        x: Ground truth values. Shape: (B, C, H, W)
        gamma: Predicted mean. Shape: (B, C, H, W)
        nu: Evidence (virtual observation count, > 0). Shape: (B, C, H, W)
        alpha: Inverse-Gamma shape (> 1). Shape: (B, C, H, W)
        beta: Inverse-Gamma rate (> 0). Shape: (B, C, H, W)
        reg_coeff: Weight for the evidence regularizer. Controls how strongly
            the model is penalized for being confident and wrong. Default: 0.01

    Returns:
        Scalar loss value.

    Notes:
        The NLL term is the negative log-marginal of a Student-t(2*alpha) distribution
        with location gamma and scale β(1+nu)/(nu*alpha). The regularizer is
        |x - gamma| * (2*nu + alpha), which penalizes high evidence on residuals.
    """
    # Residual
    error = x - gamma

    # Negative log-marginal-likelihood of Student-t predictive
    # p(x | γ, ν, α, β) = St(x; γ, β(1+ν)/(να), 2α)
    omega = 2.0 * beta * (1.0 + nu)
    nll = (
        0.5 * torch.log(torch.pi / (nu + 1e-8))
        - alpha * torch.log(omega + 1e-8)
        + (alpha + 0.5) * torch.log(error.pow(2) * nu + omega + 1e-8)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    # Evidence regularizer: penalize confidence on wrong predictions
    # When error is large, this pushes nu and alpha down (less evidence).
    # When error is small, this term vanishes — the model can be confident.
    reg = error.abs() * (2.0 * nu + alpha)

    return torch.mean(nll + reg_coeff * reg)


def evidential_uncertainty(nu, alpha, beta):
    """Decompose uncertainty from NIG parameters.

    Args:
        nu: Evidence (> 0)
        alpha: Shape (> 1)
        beta: Rate (> 0)

    Returns:
        aleatoric: Irreducible data noise — β / (α - 1)
        epistemic: Model uncertainty — β / (ν(α - 1))
        total: Full predictive variance — β(1 + ν) / (ν(α - 1))
    """
    alpha_m1 = alpha - 1.0 + 1e-8  # numerical safety
    aleatoric = beta / alpha_m1
    epistemic = beta / (nu * alpha_m1)
    total = aleatoric + epistemic
    return aleatoric, epistemic, total


def RankMe(features):
    U, S, V = torch.linalg.svd(features)
    p = S / (S.sum() + 1e-7)
    entropy = -torch.sum(p * torch.log(p + 1e-7))
    rank_me = torch.exp(entropy)
    return rank_me


def get_output_activation(name: str, beta: float = 1.0, window: float = 0.5):
    """Factory for output activation functions.

    All activations map to approximately [0, 1] to match normalized
    pixel values.

    Args:
        name: Activation name. One of:
            ``'sigmoid'``
                Standard sigmoid sigma(x). Output strictly in (0, 1).
                Problem: never reaches 0, vanishing gradients for large |x|.
            ``'hard_sigmoid'``
                Piecewise-linear approximation: clamp(0.2x + 0.5, 0, 1).
                Reaches 0 and 1 exactly. No vanishing gradient in the
                linear region, but non-smooth at the kink points.
            ``'sigmoswish'``
                x * sigmoid(βx) clamped to [0, 1]. f(0) = 0 with non-zero
                gradient. β controls ramp sharpness. Has gradient
                discontinuity at clamp boundaries.
            ``'swishoid'``
                (x+w)·sigmoid(β(x+w)) gated by sigmoid(β(x-w)/(2w)). No clamp.
                Smooth everywhere with non-zero gradient. **Recommended.**
        beta: Temperature/sharpness parameter. Higher β =
            faster saturation. Default 1.0; recommended 4.0 for swishoid,
            2.0 for sigmoswish.
        window: Half-width for swishoid zero-crossing control.
            Default 0.5 (matches [0, 1] data range). Ignored by other
            activations.

    Returns:
        Callable that maps raw predictions to ≈[0, 1].
    """
    if name == "sigmoid":
        return torch.sigmoid

    if name == "hard_sigmoid":
        def _hard_sigmoid(x: torch.Tensor) -> torch.Tensor:
            return torch.clamp(0.2 * x + 0.5, 0.0, 1.0)
        return _hard_sigmoid

    if name == "sigmoswish":
        def _sigmoswish(x: torch.Tensor) -> torch.Tensor:
            return torch.clamp(x * torch.sigmoid(beta * x), 0.0, 1.0)
        return _sigmoswish

    if name == "swishoid":
        def _swishoid(x: torch.Tensor) -> torch.Tensor:
            upper = (x + window) * torch.sigmoid(beta * (x + window))
            lower = (x - window) / (2.0 * window)
            return torch.sigmoid(beta * lower) * upper
        return _swishoid

    raise ValueError(
        f"Unknown output activation '{name}'. "
        "Choose from: sigmoid, hard_sigmoid, sigmoswish, swishoid"
    )


# --------------------------------------------------------------------
# Learnable Output Activation (nn.Module)
# --------------------------------------------------------------------

class LearnableOutputActivation(nn.Module):
    """Output activation with optionally learnable parameters.

    When 'learnable=True', 'beta' and 'window' become
    ``nn.Parameter`` tensors that the optimiser will update alongside
    the rest of the model.  This lets the network itself discover the
    optimal activation shape — analogous to how learnable mask tokens
    replaced fixed mask values.

    The module's parameters are included in 'model.parameters()'
    automatically, so no special optimiser handling is needed.

    Args:
        name: Activation type (same options as 'get_output_activation').
        beta: Initial beta value.
        window: Initial window value (used by swishoid only).
        learnable: If 'True', beta (and window for swishoid) become
            trainable 'nn.Parameter' tensors.
    """

    def __init__(
        self,
        name: str = "swishoid",
        beta: float = 4.0,
        window: float = 0.5,
        learnable: bool = False,
    ):
        super().__init__()
        self.name = name
        self.learnable = learnable

        if learnable:
            # Store in log-space so the raw parameter is unconstrained
            self._log_beta = nn.Parameter(torch.tensor(float(beta)).log())
            if name == "swishoid":
                self._log_window = nn.Parameter(torch.tensor(float(window)).log())
            else:
                self.register_buffer("_log_window", torch.tensor(float(window)).log())
        else:
            self.register_buffer("_log_beta", torch.tensor(float(beta)).log())
            self.register_buffer("_log_window", torch.tensor(float(window)).log())

    @property
    def beta(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self._log_beta) + 0.01

    @property
    def window(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self._log_window) + 0.01

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beta = self.beta
        window = self.window

        if self.name == "sigmoid":
            return torch.sigmoid(x)

        if self.name == "hard_sigmoid":
            return torch.clamp(0.2 * x + 0.5, 0.0, 1.0)

        if self.name == "sigmoswish":
            return torch.clamp(x * torch.sigmoid(beta * x), 0.0, 1.0)

        if self.name == "swishoid":
            upper = (x + window) * torch.sigmoid(beta * (x + window))
            lower = (x - window) / (2.0 * window)
            return torch.sigmoid(beta * lower) * upper

        raise ValueError(f"Unknown activation '{self.name}'")

    def extra_repr(self) -> str:
        return (
            f"name={self.name}, beta={self.beta.item():.3f}, "
            f"window={self.window.item():.3f}, learnable={self.learnable}"
        )
