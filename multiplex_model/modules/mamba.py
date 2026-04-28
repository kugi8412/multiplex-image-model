"""
Hierarchical 4-Way Mamba x Swin Transformer — pure-PyTorch implementation.

No dependency on mamba_ssm or causal-conv1d. The Selective State Space Model
(S6 / Mamba core) is implemented from scratch using standard PyTorch ops.

Architecture:
    1. SelectiveSSM       — pure-PyTorch S6 scan (discretized continuous SSM)
    2. MambaLayer         — Conv1D -> SSM -> SiLU gate (one direction)
    3. FourWayScan        — 4 MambaLayers scanning H<->, V|^, fused via linear proj
    4. WindowAttention    — local window self-attention (Swin-style, shifted windows)
    5. MambaSwinBlock     — WindowAttention + FourWayScan per block, residual
    6. RotInvMambaSwinBlock  — applies MambaSwinBlock at 0/90/180/270 deg -> averages
    7. MambaSwinEncoder   — hierarchical encoder with patch merging (registered)

Rotation invariance: exact over the discrete C4 group (90 deg multiples).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_modules import Block, Encoder, LayerNorm
from .registry import BLOCK_REGISTRY, ENCODER_REGISTRY


# ===================================================================
# 1. Pure-PyTorch Selective State Space Model (S6)
# ===================================================================

class SelectiveSSM(nn.Module):
    """Selective State Space Model — the core of Mamba, implemented in pure PyTorch.

    Discretizes a continuous SSM (A, B, C, Delta) per-token such that each position
    in the sequence can selectively decide how much to remember or forget.

    Given input x of shape (B, L, D) and parameters projected from x:
        Delta  — step size         (B, L, D)
        B_bar  — input projection  (B, L, N)
        C_bar  — output projection (B, L, N)

    The continuous SSM A in (D, N) is discretized as:
        A_bar = exp(Delta * A)            — diagonal state transition
        B_tilde = Delta * B_bar           — input matrix
        h(t) = A_bar * h(t-1) + B_tilde * x(t) — hidden state recurrence
        y(t) = C_bar * h(t)              — output
    """

    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int = None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, d_model // 16)

        # Continuous A matrix — initialized as negative log-spaced (HiPPO-inspired)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))  # (D, N) — learned in log-space

        # Projections from input to SSM parameters
        self.proj_dt = nn.Linear(d_model, self.dt_rank, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)  # rank -> full dim
        self.proj_B = nn.Linear(d_model, d_state, bias=False)
        self.proj_C = nn.Linear(d_model, d_state, bias=False)

        # D skip connection (like a residual)
        self.D = nn.Parameter(torch.ones(d_model))

        # Initialize dt bias to ensure positive Delta after softplus
        with torch.no_grad():
            dt_init = torch.exp(
                torch.rand(d_model) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            )
            inv_softplus = dt_init + torch.log(-torch.expm1(-dt_init))
            self.dt_proj.bias.copy_(inv_softplus)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            y: (B, L, D)
        """
        B, L, D = x.shape

        # Project input to SSM parameters
        dt = F.softplus(self.dt_proj(self.proj_dt(x)))  # (B, L, D) — step sizes > 0
        B_bar = self.proj_B(x)                           # (B, L, N)
        C_bar = self.proj_C(x)                           # (B, L, N)

        # Discretize A: A_bar = exp(Delta * A) where A = -exp(A_log)
        A = -torch.exp(self.A_log)  # (D, N) — negative for stability
        # dt: (B, L, D), A: (D, N) -> dA: (B, L, D, N)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))

        # Discretize B: B_tilde = Delta * B_bar
        # dt: (B, L, D) -> (B, L, D, 1), B_bar: (B, L, N) -> (B, L, 1, N)
        dB = dt.unsqueeze(-1) * B_bar.unsqueeze(2)  # (B, L, D, N)

        # Sequential scan (the core recurrence)
        y = self._sequential_scan(x, dA, dB, C_bar)

        # Skip connection
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)

        return y

    def _sequential_scan(self, x, dA, dB, C):
        """Sequential SSM scan.

        Args:
            x:  (B, L, D)
            dA: (B, L, D, N) — discretized state transition
            dB: (B, L, D, N) — discretized input matrix
            C:  (B, L, N)    — output projection

        Returns:
            y:  (B, L, D)
        """
        B, L, D = x.shape
        N = self.d_state

        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)  # (B, D, N)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)           # (B, D)
            ys.append(y_t)

        return torch.stack(ys, dim=1)  # (B, L, D)


# ===================================================================
# 2. Mamba Layer (Conv1D -> SSM -> SiLU Gate)
# ===================================================================

class MambaLayer(nn.Module):
    """One-directional Mamba layer: input projection -> conv1d -> SSM -> gated output.

    Architecture:
        x -> Linear(D, E) -> [branch_ssm, branch_gate]
        branch_ssm: Conv1D -> SiLU -> SSM
        branch_gate: SiLU
        output: branch_ssm * branch_gate -> Linear(E, D)

    where E = D * expand.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        d_inner = d_model * expand

        # Input projection: D -> 2*E (split into SSM branch + gate branch)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)

        # Depthwise Conv1D for local context
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True,
        )

        # Selective SSM
        self.ssm = SelectiveSSM(d_inner, d_state=d_state)

        # Output projection: E -> D
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            y: (B, L, D)
        """
        # Project and split
        xz = self.in_proj(x)  # (B, L, 2*E)
        x_ssm, z = xz.chunk(2, dim=-1)  # each (B, L, E)

        # Conv1D branch
        x_ssm = x_ssm.transpose(1, 2)  # (B, E, L)
        x_ssm = self.conv1d(x_ssm)[:, :, :x.shape[1]]  # causal-trim
        x_ssm = x_ssm.transpose(1, 2)  # (B, L, E)
        x_ssm = F.silu(x_ssm)

        # SSM
        x_ssm = self.ssm(x_ssm)  # (B, L, E)

        # Gated output
        y = x_ssm * F.silu(z)
        y = self.out_proj(y)  # (B, L, D)

        return y


# ===================================================================
# 3. Four-Way Cross-Scan Mamba
# ===================================================================

class FourWayScan(nn.Module):
    """4-directional Mamba scanning for 2D spatial features.

    Scans the spatial grid in 4 directions:
        1. Row-major forward  (left to right, top to bottom)
        2. Row-major backward (right to left, bottom to top)
        3. Column-major forward  (top to bottom, left to right)
        4. Column-major backward (bottom to top, right to left)

    Each direction uses an independent MambaLayer. Results are fused via
    learned linear projection.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.mamba_h_fwd = MambaLayer(d_model, d_state, d_conv, expand)
        self.mamba_h_bwd = MambaLayer(d_model, d_state, d_conv, expand)
        self.mamba_v_fwd = MambaLayer(d_model, d_state, d_conv, expand)
        self.mamba_v_bwd = MambaLayer(d_model, d_state, d_conv, expand)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) — feature map
        Returns:
            y: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Row-major: flatten (H,W) -> L = H*W
        x_row = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        out_h_fwd = self.mamba_h_fwd(x_row)
        out_h_bwd = torch.flip(self.mamba_h_bwd(torch.flip(x_row, [1])), [1])

        # Column-major: transpose H,W then flatten
        x_col = x.transpose(2, 3).flatten(2).transpose(1, 2)  # (B, W*H, C)
        out_v_fwd = self.mamba_v_fwd(x_col)
        out_v_bwd = torch.flip(self.mamba_v_bwd(torch.flip(x_col, [1])), [1])

        # Reshape back to 2D
        out_h_fwd = out_h_fwd.transpose(1, 2).reshape(B, C, H, W)
        out_h_bwd = out_h_bwd.transpose(1, 2).reshape(B, C, H, W)
        out_v_fwd = out_v_fwd.transpose(1, 2).reshape(B, C, W, H).transpose(2, 3)
        out_v_bwd = out_v_bwd.transpose(1, 2).reshape(B, C, W, H).transpose(2, 3)

        # Fuse all 4 directions
        fused = out_h_fwd + out_h_bwd + out_v_fwd + out_v_bwd
        fused = fused.flatten(2).transpose(1, 2)  # (B, H*W, C)
        fused = self.proj(fused)
        return fused.transpose(1, 2).reshape(B, C, H, W)


# ===================================================================
# 4. Window Self-Attention (Swin-style)
# ===================================================================

class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with optional cyclic shift.

    Swin-style: partitions the feature map into non-overlapping windows,
    applies self-attention within each window, then un-partitions.
    Alternating blocks use shifted windows for cross-window connectivity.
    """

    def __init__(self, dim: int, num_heads: int, window_size: int = 7,
                 shift: bool = False, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift = shift
        self.shift_size = window_size // 2 if shift else 0
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords, coords, indexing="ij")).flatten(1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            y: (B, C, H, W)
        """
        B, C, H, W = x.shape
        ws = self.window_size

        # Pad to multiples of window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x.shape

        # Cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))

        # Partition into windows: (B, C, Hp, Wp) -> (B*nW, ws*ws, C)
        x = x.reshape(B, C, Hp // ws, ws, Wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, ws * ws, C)  # (B*nW, ws^2, C)

        # Self-attention
        nW_total = x.shape[0]
        qkv = self.qkv(x).reshape(nW_total, ws * ws, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (nW_total, heads, ws^2, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(ws * ws, ws * ws, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(nW_total, ws * ws, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Un-partition: (B*nW, ws^2, C) -> (B, C, Hp, Wp)
        nH, nW_dim = Hp // ws, Wp // ws
        x = x.reshape(B, nH, nW_dim, ws, ws, C).permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, C, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]

        return x


# ===================================================================
# 5. MambaSwin Block — Window Attention + 4-Way Mamba + FFN
# ===================================================================

class MambaSwinFFN(nn.Module):
    """ConvNeXt-style FFN with depthwise conv + GELU."""

    def __init__(self, dim: int, expand: int = 4):
        super().__init__()
        hidden = dim * expand
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        self.norm = LayerNorm(dim, data_format="channels_first")
        self.pw1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.pw2 = nn.Conv2d(hidden, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_conv(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = F.gelu(x)
        x = self.pw2(x)
        return x


@BLOCK_REGISTRY.register("mambaswin")
class MambaSwinBlock(Block):
    """Combined Mamba + Swin block: local window attention -> global 4-way Mamba -> FFN.

    Each block applies:
        1. LayerNorm -> WindowAttention (local, shifted windows alternate)
        2. LayerNorm -> FourWayScan Mamba (global, all spatial positions)
        3. LayerNorm -> FFN with depthwise conv

    This captures both local detail (attention) and global context (SSM).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 7,
        shift: bool = False,
        d_state: int = 16,
        d_conv: int = 4,
        ssm_expand: int = 2,
        ffn_expand: int = 4,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm_attn = LayerNorm(dim, data_format="channels_first")
        self.attn = WindowAttention(
            dim, num_heads=num_heads, window_size=window_size, shift=shift,
        )

        self.norm_mamba = LayerNorm(dim, data_format="channels_first")
        self.mamba = FourWayScan(dim, d_state=d_state, d_conv=d_conv, expand=ssm_expand)

        self.norm_ffn = LayerNorm(dim, data_format="channels_first")
        self.ffn = MambaSwinFFN(dim, expand=ffn_expand)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Window attention (local)
        x = x + self.drop_path(self.attn(self.norm_attn(x)))
        # 4-way Mamba (global)
        x = x + self.drop_path(self.mamba(self.norm_mamba(x)))
        # FFN
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))
        return x


# ===================================================================
# 6. Rotation-Invariant Wrapper (C4 Equivariant)
# ===================================================================

@BLOCK_REGISTRY.register("rotinv_mambaswin")
class RotInvMambaSwinBlock(Block):
    """Rotation-invariant MambaSwin block via C4 group averaging.

    Applies the inner MambaSwinBlock to the input at 4 rotations
    (0, 90, 180, 270 degrees), rotates each output back to the canonical
    orientation, and averages. This gives exact equivariance under the
    discrete C4 rotation group with no additional parameters.

    The key insight: Mamba's sequential scanning is direction-dependent,
    so the same spatial pattern produces different hidden states when
    scanned at different orientations. By explicitly averaging over all
    4 rotations, we make the representation invariant to 90 degree rotations.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.block = MambaSwinBlock(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply block at 4 rotations and average
        out = self.block(x)

        x_90 = torch.rot90(x, k=1, dims=(2, 3))
        out = out + torch.rot90(self.block(x_90), k=-1, dims=(2, 3))

        x_180 = torch.rot90(x, k=2, dims=(2, 3))
        out = out + torch.rot90(self.block(x_180), k=-2, dims=(2, 3))

        x_270 = torch.rot90(x, k=3, dims=(2, 3))
        out = out + torch.rot90(self.block(x_270), k=-3, dims=(2, 3))

        return out * 0.25


# ===================================================================
# 7. Drop Path (Stochastic Depth)
# ===================================================================

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask


# ===================================================================
# 8. Hierarchical MambaSwin Encoder
# ===================================================================

@ENCODER_REGISTRY.register("mambaswin")
class MambaSwinEncoder(Encoder):
    """Hierarchical 4-Way Mamba x Swin Transformer encoder.

    Architecture per stage:
        PatchMerging -> N x RotInvMambaSwinBlock (alternating shifted windows)

    Patch merging uses stride-2 convolution for 2x spatial downsampling
    and channel expansion at each stage boundary.
    """

    def __init__(
        self,
        input_channels: int,
        layers_blocks: list[int],
        embedding_dims: list[int],
        stem: bool = True,
        patch_size: int = 2,
        block_parameters: dict | None = None,
    ):
        """
        Args:
            input_channels: Number of input channels.
            layers_blocks: Number of blocks per stage. E.g. [2, 4, 4, 2].
            embedding_dims: Channel dims per stage. E.g. [96, 192, 384, 768].
            stem: Whether to apply initial patch embedding.
            patch_size: Patch size for stem embedding.
            block_parameters: Dict with keys passed to RotInvMambaSwinBlock:
                num_heads, window_size, d_state, d_conv, ssm_expand, ffn_expand,
                drop_path, rotation_invariant (bool, default True).
        """
        super().__init__()
        bp = block_parameters.copy() if block_parameters else {}
        rotation_invariant = bp.pop("rotation_invariant", True)

        # Build patch embeddings (stem + downsampling)
        self.patch_embeds = nn.ModuleList()
        if stem:
            self.patch_embeds.append(nn.Sequential(
                nn.Conv2d(input_channels, embedding_dims[0],
                          kernel_size=patch_size, stride=patch_size),
                LayerNorm(embedding_dims[0], data_format="channels_first"),
            ))
        else:
            self.patch_embeds.append(nn.Identity())

        for i, out_dim in enumerate(embedding_dims[1:]):
            in_dim = embedding_dims[i]
            self.patch_embeds.append(nn.Sequential(
                LayerNorm(in_dim, data_format="channels_first"),
                nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2),
            ))

        # Build stages with stochastic depth schedule
        total_blocks = sum(layers_blocks)
        dp_rates = [x.item() for x in torch.linspace(0, bp.get("drop_path", 0.0), total_blocks)]
        block_idx = 0

        BlockClass = RotInvMambaSwinBlock if rotation_invariant else MambaSwinBlock

        self.stages = nn.ModuleList()
        for stage_i, (n_blocks, dim) in enumerate(zip(layers_blocks, embedding_dims)):
            stage = []
            num_heads = bp.get("num_heads", max(1, dim // 32))
            for i in range(n_blocks):
                blk_kwargs = {
                    "dim": dim,
                    "num_heads": num_heads,
                    "window_size": bp.get("window_size", 7),
                    "shift": (i % 2 == 1),
                    "d_state": bp.get("d_state", 16),
                    "d_conv": bp.get("d_conv", 4),
                    "ssm_expand": bp.get("ssm_expand", 2),
                    "ffn_expand": bp.get("ffn_expand", 4),
                    "drop_path": dp_rates[block_idx],
                }
                stage.append(BlockClass(**blk_kwargs))
                block_idx += 1
            self.stages.append(nn.Sequential(*stage))

    def forward(self, x: torch.Tensor, return_features: bool = False) -> dict:
        outputs = {}
        features = []

        for patch_embed, stage in zip(self.patch_embeds, self.stages):
            x = patch_embed(x)
            x = stage(x)
            if return_features:
                features.append(x)

        outputs["output"] = x
        if return_features:
            outputs["features"] = features
        return outputs


# ===================================================================
# Backward compatibility aliases
# ===================================================================
# 9. Pure Vision Mamba (ViM) Block — 4-Way Mamba + FFN, no Attention
# ===================================================================

@BLOCK_REGISTRY.register("vim")
class VimBlock(Block):
    """Pure Vision Mamba block: 4-way cross-scan SSM + FFN (no window attention).

    Each block applies:
        1. LayerNorm -> FourWayScan Mamba (global, all spatial positions)
        2. LayerNorm -> FFN with depthwise conv

    Unlike MambaSwinBlock, this block does NOT include window attention,
    making it a pure state-space model for isotropic ViM architectures.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        ffn_expand: int = 4,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm_mamba = LayerNorm(dim, data_format="channels_first")
        self.mamba = FourWayScan(dim, d_state=d_state, d_conv=d_conv, expand=expand)

        self.norm_ffn = LayerNorm(dim, data_format="channels_first")
        self.ffn = MambaSwinFFN(dim, expand=ffn_expand)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.mamba(self.norm_mamba(x)))
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))
        return x


# ===================================================================
# 10. Vision Mamba (ViM) Encoder — Isotropic or Hierarchical
# ===================================================================

@ENCODER_REGISTRY.register("vim")
class VimEncoder(Encoder):
    """Pure Vision Mamba encoder using 4-way cross-scan SSM blocks.

    Supports both isotropic (single-stage) and hierarchical (multi-stage)
    configurations via layers_blocks/embedding_dims lists.
    Does NOT use window attention — purely SSM-based.
    """

    def __init__(
        self,
        input_channels: int,
        layers_blocks: list[int],
        embedding_dims: list[int],
        stem: bool = True,
        patch_size: int = 2,
        block_parameters: dict | None = None,
    ):
        super().__init__()
        bp = block_parameters.copy() if block_parameters else {}

        # Build patch embeddings (stem + downsampling)
        self.patch_embeds = nn.ModuleList()
        if stem:
            self.patch_embeds.append(nn.Sequential(
                nn.Conv2d(input_channels, embedding_dims[0],
                          kernel_size=patch_size, stride=patch_size),
                LayerNorm(embedding_dims[0], data_format="channels_first"),
            ))
        else:
            self.patch_embeds.append(nn.Identity())

        for i, out_dim in enumerate(embedding_dims[1:]):
            in_dim = embedding_dims[i]
            self.patch_embeds.append(nn.Sequential(
                LayerNorm(in_dim, data_format="channels_first"),
                nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2),
            ))

        # Build stages with stochastic depth schedule
        total_blocks = sum(layers_blocks)
        dp_rates = [x.item() for x in torch.linspace(0, bp.get("drop_path", 0.0), total_blocks)]
        block_idx = 0

        self.stages = nn.ModuleList()
        for stage_i, (n_blocks, dim) in enumerate(zip(layers_blocks, embedding_dims)):
            stage = []
            for i in range(n_blocks):
                blk_kwargs = {
                    "dim": dim,
                    "d_state": bp.get("d_state", 16),
                    "d_conv": bp.get("d_conv", 4),
                    "expand": bp.get("expand", 2),
                    "ffn_expand": bp.get("ffn_expand", 4),
                    "drop_path": dp_rates[block_idx],
                }
                stage.append(VimBlock(**blk_kwargs))
                block_idx += 1
            self.stages.append(nn.Sequential(*stage))

    def forward(self, x: torch.Tensor, return_features: bool = False) -> dict:
        outputs = {}
        features = []

        for patch_embed, stage in zip(self.patch_embeds, self.stages):
            x = patch_embed(x)
            x = stage(x)
            if return_features:
                features.append(x)

        outputs["output"] = x
        if return_features:
            outputs["features"] = features
        return outputs


# ===================================================================
# Backward compatibility aliases
# ===================================================================

CrossScanMambaBlock = MambaSwinBlock
VisionMambaEncoder = MambaSwinEncoder
