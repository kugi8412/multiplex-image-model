#!/usr/bin/env python
# -*- coding: utf-8 -*-
# train_immukronos_unified.py
#
# Unified ImmuKRONOS training script supporting multiple backbones (ViT, ConvNeXt, Swin)
# with DINOv2 or DINOv3 self-distillation objectives.
#
# The backbone is selected via the `backbone` config key:
#   - "kronos_vit" : ViT blocks (isotropic, token-sequence) — original ImmuKRONOS
#   - "convnext"   : ConvNeXt v2 blocks (hierarchical, spatial CNN)
#   - "swin"       : Swin Transformer blocks (hierarchical, window attention)
#   - "vim"        : Vision Mamba / SSM blocks (isotropic, spatial, no attention)
#
# Preprocessing is selected via the `preprocessing` config key:
#   - "immuvis"  : Butterworth filter + clip normalization (default)
#   - "virtues"  : Gaussian blur + z-standardization (no butterworth/clip)
#
# Usage:
#   python train_immukronos_unified.py configs/immukronos_vit_v2.yaml
#   python train_immukronos_unified.py configs/immukronos_convnext_v3.yaml
#   python train_immukronos_unified.py configs/immukronos_vim_v2.yaml

import argparse
import functools
import os
import random
import math

import comet_ml  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
)
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler
from multiplex_model.utils import init_experiment, finish_experiment, get_run_name
from multiplex_model.modules.immuvis import Hyperkernel
from multiplex_model.kronos.dino_head import DINOHead


class SharedPatchEmbed(nn.Module):
    """Marker-agnostic patch embedding: shared Conv2d across all channels.

    Each marker channel is independently embedded with the same conv kernel,
    then summed across channels. No marker identity information is used.
    This is the 'no hyperkernel' baseline for I-JEPA / Immu-JEPA ablations.
    """

    def __init__(self, embed_dim, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(1, embed_dim, kernel_size=kernel_size,
                              stride=stride, padding=padding)

    def forward(self, x, channel_ids):
        """
        Args:
            x: (B, C, H, W) multi-channel input
            channel_ids: (B, C) marker IDs — ignored (marker-agnostic)
        Returns:
            (B, embed_dim, h_p, w_p)
        """
        B, C, H, W = x.shape
        # Embed each channel independently with the same conv
        x = x.reshape(B * C, 1, H, W)
        x = self.conv(x)  # (B*C, E, h_p, w_p)
        _, E, Hp, Wp = x.shape
        x = x.reshape(B, C, E, Hp, Wp)
        # Sum across channels (like Hyperkernel but without marker-specific kernels)
        return x.sum(dim=1)  # (B, E, h_p, w_p)

# Import registered encoders so they register themselves
import multiplex_model.modules.convnext  # noqa: F401
import multiplex_model.modules.swin      # noqa: F401
import multiplex_model.modules.vit       # noqa: F401
import multiplex_model.modules.mamba     # noqa: F401
from multiplex_model.modules.registry import ENCODER_REGISTRY

# Import KRONOS ViT blocks for the "kronos_vit" backbone option
from multiplex_model.kronos.vision_transformer import Block as KronosBlock, MemEffAttention


# ============================================================
# DINOv3 BUILDING BLOCKS (used by "kronos_vit" backbone in v3 mode)
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


class RotaryPositionEmbedding(nn.Module):
    """2D RoPE for vision transformers."""
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 4 == 0
        self.half_dim = head_dim // 4
        inv_freq = 1.0 / (base ** (torch.arange(0, self.half_dim, dtype=torch.float32) / self.half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _build_freqs(self, h, w, device, dtype):
        y = torch.arange(h, device=device, dtype=torch.float32)
        x = torch.arange(w, device=device, dtype=torch.float32)
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        gy, gx = gy.reshape(-1), gx.reshape(-1)
        inv = self.inv_freq.to(device)
        freqs_y = torch.outer(gy, inv)
        freqs_x = torch.outer(gx, inv)
        return torch.cat([freqs_y, freqs_y, freqs_x, freqs_x], dim=-1).to(dtype)

    def forward(self, q, k, h, w, num_prefix=0):
        freqs = self._build_freqs(h, w, q.device, q.dtype)
        cos_f, sin_f = torch.cos(freqs), torch.sin(freqs)

        def rotate(t):
            t1, t2 = t[..., ::2], t[..., 1::2]
            return torch.stack((-t2, t1), dim=-1).flatten(-2)

        if num_prefix > 0:
            q_pre, q_patch = q[:, :, :num_prefix], q[:, :, num_prefix:]
            k_pre, k_patch = k[:, :, :num_prefix], k[:, :, num_prefix:]
        else:
            q_pre = k_pre = None
            q_patch, k_patch = q, k

        q_patch = q_patch * cos_f + rotate(q_patch) * sin_f
        k_patch = k_patch * cos_f + rotate(k_patch) * sin_f

        if num_prefix > 0:
            return torch.cat([q_pre, q_patch], dim=2), torch.cat([k_pre, k_patch], dim=2)
        return q_patch, k_patch


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=None, drop=0.0, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * x.new_empty(shape).bernoulli_(keep).div_(keep)


class DINOv3Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_bias=True,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope=None, h=None, w=None, num_prefix=0):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if rope is not None and h is not None and w is not None:
            q, k = rope(q, k, h, w, num_prefix=num_prefix)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class DINOv3Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False,
                 proj_bias=True, drop_path=0.0, init_values=None):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = DINOv3Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLUFFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ls1 = nn.Parameter(init_values * torch.ones(dim)) if init_values else None
        self.ls2 = nn.Parameter(init_values * torch.ones(dim)) if init_values else None

    def forward(self, x, rope=None, h=None, w=None, num_prefix=0):
        a = self.attn(self.norm1(x), rope=rope, h=h, w=w, num_prefix=num_prefix)
        if self.ls1 is not None:
            a = a * self.ls1
        x = x + self.drop_path(a)
        m = self.mlp(self.norm2(x))
        if self.ls2 is not None:
            m = m * self.ls2
        x = x + self.drop_path(m)
        return x


# ============================================================
# UNIFIED IMMUKRONOS MODEL
# ============================================================

class ImmuKRONOS(nn.Module):
    """Unified ImmuKRONOS model supporting ViT, ConvNeXt, and Swin backbones.

    Architecture:
        1. Hyperkernel stem — marker-aware patch embedding + channel fusion
        2. Backbone — configurable (ViT / ConvNeXt / Swin)
        3. DINO head (CLS) + optional iBOT head (patches)

    For ViT backbone ("kronos_vit"):
        - Isotropic token-sequence architecture
        - CLS token, optional register tokens
        - DINOv2: LayerNorm + standard MLP (KRONOS Block)
        - DINOv3: RMSNorm + SwiGLU + RoPE (DINOv3Block)

    For ConvNeXt / Swin backbones:
        - Spatial feature maps (B, D, H, W)
        - CLS = global average pooling
        - Patch tokens = flattened spatial features
        - No explicit positional embeddings (handled by conv locality / window attention)
    """

    def __init__(
        self,
        num_markers: int,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        patch_size: int = 8,
        out_dim: int = 65536,
        drop_path_rate: float = 0.0,
        num_register_tokens: int = 0,
        ibot_out_dim: int = None,
        init_values: float = None,
        mask_strategy: str = "zero",
        backbone: str = "kronos_vit",
        dino_version: str = "v2",
        backbone_config: dict = None,
        use_hyperkernel: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_register_tokens = num_register_tokens
        self.mask_strategy = mask_strategy
        self.backbone_type = backbone
        self.dino_version = dino_version
        self._is_spatial_backbone = backbone in ("convnext", "swin", "vim")

        # 1. Stem — Hyperkernel (marker-aware) or SharedPatchEmbed (marker-agnostic)
        if use_hyperkernel:
            self.hyperkernel = Hyperkernel(
                num_channels=num_markers,
                input_dim=1,
                embedding_dim=embed_dim,
                module_type="encoder",
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
            )
        else:
            self.hyperkernel = SharedPatchEmbed(
                embed_dim=embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
            )

        # 2. Backbone
        if backbone == "kronos_vit":
            self._build_vit_backbone(embed_dim, depth, num_heads, drop_path_rate, init_values, num_register_tokens)
        elif backbone in ("convnext", "swin", "vim"):
            self._build_spatial_backbone(backbone, embed_dim, depth, drop_path_rate, backbone_config or {})
        else:
            raise ValueError(f"Unknown backbone: {backbone}. Use 'kronos_vit', 'convnext', 'swin', or 'vim'.")

        # 3. DINO head
        self.dino_head = DINOHead(in_dim=embed_dim, out_dim=out_dim)

        # 4. iBOT head (DINOv3 only)
        self.ibot_head = (
            DINOHead(in_dim=embed_dim, out_dim=ibot_out_dim)
            if ibot_out_dim is not None else None
        )

        # 5. Mask token for iBOT / I-JEPA
        # Maximum patch-grid positions supported by a per-position ("full") mask.
        # ViT pos_embed holds 1024 patch slots; spatial backbones mask at the
        # hyperkernel-output resolution which stays <= this for supported crops.
        self.max_mask_positions = 1024
        if mask_strategy == "learnable":
            # Single shared learnable mask token, broadcast to every masked slot.
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        elif mask_strategy == "learnable_full":
            # Full per-position learnable mask: one distinct token per patch slot.
            self.mask_token = nn.Parameter(torch.zeros(1, self.max_mask_positions, embed_dim))
        elif mask_strategy == "zero":
            self.register_buffer("mask_token", torch.zeros(1, 1, embed_dim))
        elif mask_strategy == "negative":
            self.register_buffer("mask_token", torch.full((1, 1, embed_dim), -1.0))
        else:
            raise ValueError(
                f"Unknown mask_strategy: {mask_strategy}. "
                f"Use 'zero', 'negative', 'learnable', or 'learnable_full'.")

        self._init_weights()

    def _build_vit_backbone(self, embed_dim, depth, num_heads, drop_path_rate, init_values, num_register_tokens):
        """Build ViT (isotropic) backbone — either DINOv2 or DINOv3 blocks."""
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens > 0 else None
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        if self.dino_version == "v3":
            # DINOv3: RMSNorm + RoPE + SwiGLU
            head_dim = embed_dim // num_heads
            self.rope = RotaryPositionEmbedding(head_dim=head_dim)
            self.blocks = nn.ModuleList([
                DINOv3Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0,
                            qkv_bias=False, drop_path=dpr[i], init_values=init_values)
                for i in range(depth)
            ])
            self.norm = RMSNorm(embed_dim)
        else:
            # DINOv2: LayerNorm + standard MLP (KRONOS Block)
            self.rope = None
            self.pos_embed = nn.Parameter(torch.zeros(1, 1024 + 1, embed_dim))
            self.blocks = nn.ModuleList([
                KronosBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0,
                            qkv_bias=True, attn_class=MemEffAttention,
                            drop_path=dpr[i], init_values=init_values)
                for i in range(depth)
            ])
            self.norm = nn.LayerNorm(embed_dim)

    def _build_spatial_backbone(self, backbone_name, embed_dim, depth, drop_path_rate, backbone_config):
        """Build ConvNeXt, Swin, or ViM backbone — spatial feature map architecture."""
        # For spatial backbones, we need blocks that operate on (B, C, H, W) feature maps
        encoder_cls = ENCODER_REGISTRY.get(backbone_name)

        # Build encoder using the registered class
        block_params = backbone_config.get("block_parameters", {})
        layers_blocks = backbone_config.get("layers_blocks", [depth])
        embedding_dims = backbone_config.get("embedding_dims", [embed_dim])

        self.spatial_encoder = encoder_cls(
            input_channels=embed_dim,
            layers_blocks=layers_blocks,
            embedding_dims=embedding_dims,
            stem=False,  # Hyperkernel already does the stem / patchification
            block_parameters=block_params,
        )

        # Final norm: applied on flattened features
        self.norm = nn.LayerNorm(embedding_dims[-1])

        # No CLS token / register tokens for spatial backbones — use GAP
        self.cls_token = None
        self.register_tokens = None
        self.rope = None

        # If the encoder changes dimensionality, add a projection
        if embedding_dims[-1] != embed_dim:
            self.backbone_proj = nn.Linear(embedding_dims[-1], embed_dim)
        else:
            self.backbone_proj = nn.Identity()

    def _init_weights(self):
        if hasattr(self, 'cls_token') and self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        if hasattr(self, 'pos_embed'):
            nn.init.normal_(self.pos_embed, std=0.02)
        if self.mask_strategy in ("learnable", "learnable_full"):
            nn.init.normal_(self.mask_token, std=0.02)
        if hasattr(self, 'register_tokens') and self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)

    @property
    def num_prefix_tokens(self):
        n = 0
        if self.cls_token is not None:
            n += 1
        if self.register_tokens is not None:
            n += self.num_register_tokens
        return n

    def forward_features(self, x, channel_ids, mask=None):
        """Extract features from Hyperkernel + backbone.

        Args:
            x: (B, C_markers, H, W)
            channel_ids: (B, C_markers) marker token IDs
            mask: (B, N_patches) boolean mask for iBOT (True=masked). Only for ViT.
        Returns:
            dict with "cls_token" (B, D) and "patch_tokens" (B, N, D)
        """
        B, C, H, W = x.shape
        h_p, w_p = H // self.patch_size, W // self.patch_size
        N = h_p * w_p

        # Stem: Hyperkernel (marker-aware) or SharedPatchEmbed (marker-agnostic)
        if isinstance(self.hyperkernel, Hyperkernel):
            x_hk = x.reshape(B * C, 1, H, W)
            x_enc = self.hyperkernel(x_hk, channel_ids)  # (B, embed_dim, h_p, w_p)
        else:
            x_enc = self.hyperkernel(x, channel_ids)  # (B, embed_dim, h_p, w_p)

        if self._is_spatial_backbone:
            return self._forward_spatial(x_enc, h_p, w_p, mask)
        else:
            return self._forward_vit(x_enc, h_p, w_p, N, B, mask)

    def _forward_vit(self, x_enc, h_p, w_p, N, B, mask):
        """ViT backbone forward — token sequence with CLS."""
        x_enc = x_enc.flatten(2).transpose(1, 2)  # (B, N, D)

        # Apply iBOT mask
        if mask is not None:
            if self.mask_strategy == "learnable_full":
                # Per-position learnable mask tokens (1, N, D), one per patch slot.
                mask_tokens = self.mask_token[:, :x_enc.shape[1], :].to(x_enc.dtype)
                x_enc = torch.where(
                    mask.unsqueeze(-1),
                    mask_tokens.expand(x_enc.shape[0], -1, -1), x_enc)
            else:
                mask_val = self.mask_token.squeeze(0).squeeze(0).to(x_enc.dtype)
                x_enc = torch.where(mask.unsqueeze(-1), mask_val.unsqueeze(0).unsqueeze(0).expand_as(x_enc), x_enc)

        # CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x_enc = torch.cat([cls, x_enc], dim=1)

        # Positional embedding (DINOv2) or RoPE applied inside blocks (DINOv3)
        if self.dino_version == "v2":
            x_enc = x_enc + self.pos_embed[:, :N + 1, :]

        # Register tokens
        if self.register_tokens is not None:
            reg = self.register_tokens.expand(B, -1, -1)
            x_enc = torch.cat([x_enc[:, :1], reg, x_enc[:, 1:]], dim=1)

        num_prefix = self.num_prefix_tokens

        if self.dino_version == "v3":
            for blk in self.blocks:
                x_enc = blk(x_enc, rope=self.rope, h=h_p, w=w_p, num_prefix=num_prefix)
        else:
            for blk in self.blocks:
                x_enc = blk(x_enc)

        x_enc = self.norm(x_enc)
        return {
            "cls_token": x_enc[:, 0],
            "patch_tokens": x_enc[:, num_prefix:],
        }

    def forward_features_context(self, x, channel_ids, context_mask):
        """I-JEPA context encoder: only process unmasked (context) patches.

        Unlike forward_features which processes ALL patches (replacing targets
        with mask tokens), this method drops target patches entirely before the
        transformer, reducing attention cost from O(N²) to O(N_ctx²).

        Args:
            x: (B, C_markers, H, W)
            channel_ids: (B, C_markers) marker token IDs
            context_mask: (B, N_patches) boolean — True at context positions to KEEP.
        Returns:
            dict with "context_tokens" (B, N_ctx, D) and "context_indices" (B, N_ctx)
        """
        B, C, H, W = x.shape
        h_p, w_p = H // self.patch_size, W // self.patch_size
        N = h_p * w_p

        # Stem: Hyperkernel or SharedPatchEmbed
        if isinstance(self.hyperkernel, Hyperkernel):
            x_hk = x.reshape(B * C, 1, H, W)
            x_enc = self.hyperkernel(x_hk, channel_ids)
        else:
            x_enc = self.hyperkernel(x, channel_ids)

        if self._is_spatial_backbone:
            # Convolution / window attention needs the full grid, so target
            # patches cannot be *dropped* — they are replaced by the mask token
            # at the input resolution (MAE-style). To keep this a genuine I-JEPA
            # objective, we then hand the predictor ONLY the true context tokens
            # (mapped to the possibly-downsampled output grid) so that target
            # positions are reconstructed from the predictor's own mask token
            # rather than from encoder features computed at those positions.
            feats = self._forward_spatial(x_enc, h_p, w_p, ~context_mask)
            patch_tokens = feats["patch_tokens"]  # (B, N_out, D)
            N_out = patch_tokens.shape[1]
            h_out = w_out = int(round(math.sqrt(N_out)))
            ctx_out = self._downsample_context_mask(context_mask, h_p, w_p, h_out, w_out)

            n_ctx = ctx_out.sum(dim=1)  # (B,)
            max_ctx = int(n_ctx.max().item())
            context_tokens = torch.zeros(B, max_ctx, patch_tokens.shape[-1],
                                         device=patch_tokens.device, dtype=patch_tokens.dtype)
            context_indices = torch.zeros(B, max_ctx, device=patch_tokens.device, dtype=torch.long)
            for i in range(B):
                idx = ctx_out[i].nonzero(as_tuple=True)[0]
                context_tokens[i, :len(idx)] = patch_tokens[i, idx]
                context_indices[i, :len(idx)] = idx
            return {
                "context_tokens": context_tokens,      # (B, N_ctx, D)
                "context_indices": context_indices,    # (B, N_ctx)
                "n_ctx": n_ctx,                         # (B,) actual counts
            }

        # ViT path — drop target tokens for efficiency
        x_enc = x_enc.flatten(2).transpose(1, 2)  # (B, N, D)

        # Keep only context positions (variable-length per sample → pad to max)
        n_ctx = context_mask.sum(dim=1)  # (B,)
        max_ctx = n_ctx.max().item()

        # Gather context tokens and their position indices
        context_tokens = torch.zeros(B, max_ctx, x_enc.shape[-1],
                                     device=x_enc.device, dtype=x_enc.dtype)
        context_indices = torch.zeros(B, max_ctx, device=x_enc.device, dtype=torch.long)

        for i in range(B):
            idx = context_mask[i].nonzero(as_tuple=True)[0]
            context_tokens[i, :len(idx)] = x_enc[i, idx]
            context_indices[i, :len(idx)] = idx

        # CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x_seq = torch.cat([cls, context_tokens], dim=1)  # (B, 1+N_ctx, D)

        # Positional embedding — select only context positions
        if self.dino_version == "v2":
            cls_pe = self.pos_embed[:, :1, :]  # (1, 1, D)
            patch_pe = self.pos_embed[:, 1:N + 1, :]  # (1, N, D)
            # Gather pos embeddings for context positions
            ctx_pe = torch.zeros(B, max_ctx, self.pos_embed.shape[-1],
                                 device=x_enc.device, dtype=x_enc.dtype)
            for i in range(B):
                idx = context_mask[i].nonzero(as_tuple=True)[0]
                ctx_pe[i, :len(idx)] = patch_pe[0, idx]
            pe = torch.cat([cls_pe.expand(B, -1, -1), ctx_pe], dim=1)
            x_seq = x_seq + pe

        # Register tokens
        if self.register_tokens is not None:
            reg = self.register_tokens.expand(B, -1, -1)
            x_seq = torch.cat([x_seq[:, :1], reg, x_seq[:, 1:]], dim=1)

        num_prefix = self.num_prefix_tokens

        if self.dino_version == "v3":
            for blk in self.blocks:
                x_seq = blk(x_seq, rope=self.rope, h=h_p, w=w_p, num_prefix=num_prefix)
        else:
            for blk in self.blocks:
                x_seq = blk(x_seq)

        x_seq = self.norm(x_seq)
        return {
            "context_tokens": x_seq[:, num_prefix:],  # (B, N_ctx, D)
            "context_indices": context_indices,        # (B, N_ctx)
            "n_ctx": n_ctx,                            # (B,) actual counts
        }

    def _downsample_context_mask(self, context_mask, h_in, w_in, h_out, w_out):
        """Map an input-resolution context mask onto a downsampled output grid.

        An output cell counts as *context* only if EVERY input patch that maps
        into it is context (no target overlap). Equivalent to the complement of
        max-pooling the target mask, which keeps it consistent with the
        `downsample_mask` (max-pool) used to build the target mask in the loop.
        """
        if h_in == h_out and w_in == w_out:
            return context_mask
        B = context_mask.shape[0]
        target_2d = (~context_mask).float().reshape(B, 1, h_in, w_in)
        pool_h = h_in // h_out
        pool_w = w_in // w_out
        target_out = F.max_pool2d(target_2d, kernel_size=(pool_h, pool_w),
                                  stride=(pool_h, pool_w))
        return ~(target_out.reshape(B, -1).bool())

    def _forward_spatial(self, x_enc, h_p, w_p, mask):
        """ConvNeXt / Swin backbone forward — spatial feature maps."""
        # x_enc: (B, D, h_p, w_p)

        # Apply iBOT mask in spatial domain
        if mask is not None:
            B, D, H, W = x_enc.shape
            mask_2d = mask.reshape(B, 1, H, W).float()
            if self.mask_strategy == "learnable_full":
                # Per-position learnable mask tokens reshaped to the patch grid.
                mt = self.mask_token[:, :H * W, :].to(x_enc.dtype)  # (1, H*W, D)
                mask_val = mt.transpose(1, 2).reshape(1, D, H, W)   # (1, D, H, W)
            else:
                mask_val = self.mask_token.squeeze(0).transpose(0, 1).unsqueeze(-1)  # (1, D, 1)
                mask_val = mask_val.unsqueeze(-1).expand_as(x_enc)  # (1, D, H, W)
            x_enc = x_enc * (1 - mask_2d) + mask_val * mask_2d

        # Pass through spatial backbone (returns dict with "output" key)
        x_enc = self.spatial_encoder(x_enc)["output"]

        # Flatten to token-like format for DINO heads
        B, D_out, H_out, W_out = x_enc.shape
        patch_tokens = x_enc.flatten(2).transpose(1, 2)  # (B, H*W, D_out)
        patch_tokens = self.norm(patch_tokens)
        patch_tokens = self.backbone_proj(patch_tokens)

        # CLS = global average pooling
        cls_token = patch_tokens.mean(dim=1)

        return {
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
        }

    def forward(self, x, channel_ids, mask=None, return_ibot=False):
        features = self.forward_features(x, channel_ids, mask=mask)
        dino_out = self.dino_head(features["cls_token"])

        if return_ibot and mask is not None and self.ibot_head is not None:
            masked_patches = features["patch_tokens"][mask]
            ibot_out = self.ibot_head(masked_patches)
            return dino_out, ibot_out

        return dino_out


# ============================================================
# LOSSES
# ============================================================

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, nglobal_crops=2):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.nglobal_crops = nglobal_crops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(max(nepochs - warmup_teacher_temp_epochs, 0)) * teacher_temp,
        ))

    def forward(self, student_output, teacher_output, epoch, update_center=True):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)
        temp = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.nglobal_crops)

        total_loss, n = 0, 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                total_loss += torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1).mean()
                n += 1
        total_loss /= n
        if update_center:
            self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        bc = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + bc * (1 - self.center_momentum)


class iBOTLoss(nn.Module):
    def __init__(self, out_dim, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(max(nepochs - warmup_teacher_temp_epochs, 0)) * teacher_temp,
        ))

    def forward(self, student_logits, teacher_logits, epoch, update_center=True):
        if student_logits.shape[0] == 0:
            return student_logits.sum() * 0
        temp = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)]
        s = student_logits / self.student_temp
        t = F.softmax((teacher_logits - self.center) / temp, dim=-1).detach()
        loss = torch.sum(-t * F.log_softmax(s, dim=-1), dim=-1)
        if update_center:
            self._uc(teacher_logits)
        return loss.mean()

    @torch.no_grad()
    def _uc(self, t):
        self.center = self.center * self.center_momentum + t.mean(0, keepdim=True) * (1 - self.center_momentum)


class KoLeoLoss(nn.Module):
    def forward(self, features):
        features = F.normalize(features, dim=-1, p=2)
        dists = torch.cdist(features, features)
        dists = dists.masked_fill(torch.eye(dists.shape[0], dtype=torch.bool, device=dists.device), float("inf"))
        return -torch.log(dists.min(dim=-1).values + 1e-8).mean()


# ============================================================
# MULTI-CROP AUGMENTATION & DATA
# ============================================================

class MultiCropTransform:
    def __init__(self, global_size, local_size, global_scale, local_scale,
                 local_crops_number, global_crops_number=2):
        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number
        self.global_transform = Compose([
            RandomResizedCrop(global_size, scale=tuple(global_scale), interpolation=InterpolationMode.BILINEAR),
            RandomRotation(180, interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
        ])
        self.local_transform = Compose([
            RandomResizedCrop(local_size, scale=tuple(local_scale), interpolation=InterpolationMode.BILINEAR),
            RandomRotation(180, interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
        ])

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
        crops = []
        for _ in range(self.global_crops_number):
            crops.append(self.global_transform(img))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(img))
        return crops


class DINODatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, channel_ids, panel_idx, img_path = self.base_dataset[idx]
        return self.transform(img), channel_ids, panel_idx, img_path


def dino_collate_fn(batch, channel_fraction=None):
    num_crops = len(batch[0][0])
    crops = [torch.stack([item[0][i] for item in batch]) for i in range(num_crops)]
    channel_ids = torch.stack([item[1] for item in batch])
    if channel_fraction is not None:
        C = crops[0].shape[1]
        frac = random.uniform(*channel_fraction)
        n_keep = max(1, int(C * frac))
        if n_keep < C:
            perm = torch.randperm(C)[:n_keep].sort().values
            crops = [crop[:, perm] for crop in crops]
            channel_ids = channel_ids[:, perm]
    return crops, channel_ids


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0):
    warmup_iters = warmup_epochs * niter_per_ep
    warmup_schedule = np.array([]) if warmup_epochs == 0 else np.linspace(0, base_value, warmup_iters)
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    return np.concatenate((warmup_schedule, schedule))


def generate_ibot_mask(batch_size, num_patches, mask_ratio=0.3, device=None):
    num_masked = int(num_patches * mask_ratio)
    mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
    for i in range(batch_size):
        mask[i, torch.randperm(num_patches, device=device)[:num_masked]] = True
    return mask


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Unified ImmuKRONOS training (ViT/ConvNeXt/Swin × DINOv2/v3)")
    parser.add_argument("config", help="Path to config YAML")
    parser.add_argument("--from-checkpoint", default=None, help="Resume from checkpoint")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    yaml = YAML(typ="safe")
    with open(args.config, "r") as f:
        config = yaml.load(f)

    device = torch.device(args.device or config.get("device", "cuda"))
    dino_version = config.get("dino_version", "v2")
    backbone = config.get("backbone", "kronos_vit")
    assert dino_version in ("v2", "v3"), f"dino_version must be 'v2' or 'v3', got '{dino_version}'"

    print(f"=== ImmuKRONOS | backbone={backbone} | DINO {dino_version} | device={device} ===")

    # ---- Preprocessing mode ----
    preprocessing = config.get("preprocessing", "immuvis")
    assert preprocessing in ("immuvis", "virtues"), f"preprocessing must be 'immuvis' or 'virtues', got '{preprocessing}'"
    use_butterworth = preprocessing == "immuvis"
    use_clip_norm = preprocessing == "immuvis"
    print(f"Preprocessing: {preprocessing} (butterworth={use_butterworth}, clip_norm={use_clip_norm})")

    # ---- Data ----
    PANEL_CONFIG = YAML().load(open(config["panel_config_path"]))
    TOKENIZER = YAML().load(open(config["tokenizer_config_path"]))
    num_markers = len(TOKENIZER)

    global_crops_number = config.get("global_crops_number", 2)
    local_crops_number = config.get("local_crops_number", 6)
    ncrops = global_crops_number + local_crops_number

    channel_fraction = config.get("channel_fraction", None)
    if channel_fraction is not None:
        channel_fraction = tuple(channel_fraction)
        collate_fn = functools.partial(dino_collate_fn, channel_fraction=channel_fraction)
    else:
        collate_fn = dino_collate_fn

    transform = MultiCropTransform(
        global_size=config["global_crops_size"],
        local_size=config["local_crops_size"],
        global_scale=config["global_crops_scale"],
        local_scale=config["local_crops_scale"],
        local_crops_number=local_crops_number,
        global_crops_number=global_crops_number,
    )

    train_base = DatasetFromTIFF(
        panels_config=PANEL_CONFIG, split="train", marker_tokenizer=TOKENIZER,
        transform=None, use_preprocessing=False, use_butterworth_filter=use_butterworth,
        use_clip_normalization=use_clip_norm, file_extension=config.get("file_extension", "npy"),
    )
    train_dataset = DINODatasetWrapper(train_base, transform)
    train_sampler = PanelBatchSampler(train_base, config["batch_size"])
    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler,
        num_workers=config.get("num_workers", 4), collate_fn=collate_fn,
        pin_memory=True, persistent_workers=config.get("num_workers", 4) > 0,
        prefetch_factor=4 if config.get("num_workers", 4) > 0 else None,
    )

    val_base = DatasetFromTIFF(
        panels_config=PANEL_CONFIG, split="test", marker_tokenizer=TOKENIZER,
        transform=None, use_preprocessing=False, use_butterworth_filter=use_butterworth,
        use_clip_normalization=use_clip_norm, file_extension=config.get("file_extension", "npy"),
    )
    val_dataset = DINODatasetWrapper(val_base, transform)
    val_sampler = PanelBatchSampler(val_base, config["batch_size"], shuffle=False)
    val_loader = DataLoader(
        val_dataset, batch_sampler=val_sampler,
        num_workers=config.get("num_workers", 4), collate_fn=dino_collate_fn,
        pin_memory=True,
    )

    # ---- Model ----
    embed_dim = config.get("embed_dim", 768)
    depth = config.get("depth", 12)
    num_heads = config.get("num_heads", 12)
    ibot_out_dim = config.get("ibot_out_dim", None) if dino_version == "v3" else None

    model_kwargs = dict(
        num_markers=num_markers,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        patch_size=config.get("patch_size", 8),
        out_dim=config["out_dim"],
        drop_path_rate=config.get("drop_path_rate", 0.1),
        num_register_tokens=config.get("num_register_tokens", 0),
        ibot_out_dim=ibot_out_dim,
        init_values=config.get("init_values", None),
        mask_strategy=config.get("mask_strategy", "zero"),
        backbone=backbone,
        dino_version=dino_version,
        backbone_config=config.get("backbone_config", None),
        use_hyperkernel=config.get("use_hyperkernel", True),
    )

    student = ImmuKRONOS(**model_kwargs).to(device)
    teacher_kwargs = {**model_kwargs, "drop_path_rate": 0.0}
    teacher = ImmuKRONOS(**teacher_kwargs).to(device)

    teacher.load_state_dict(student.state_dict(), strict=False)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    param_count = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Student parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

    # ---- Losses ----
    dino_loss_fn = DINOLoss(
        out_dim=config["out_dim"], ncrops=ncrops,
        warmup_teacher_temp=config["warmup_teacher_temp"],
        teacher_temp=config["teacher_temp"],
        warmup_teacher_temp_epochs=config["warmup_teacher_temp_epochs"],
        nepochs=config["epochs"], nglobal_crops=global_crops_number,
    ).to(device)

    ibot_loss_fn = koleo_loss_fn = None
    if dino_version == "v3":
        ibot_loss_fn = iBOTLoss(
            out_dim=ibot_out_dim,
            warmup_teacher_temp=config.get("ibot_warmup_teacher_temp", config["warmup_teacher_temp"]),
            teacher_temp=config.get("ibot_teacher_temp", config["teacher_temp"]),
            warmup_teacher_temp_epochs=config["warmup_teacher_temp_epochs"],
            nepochs=config["epochs"],
        ).to(device)
        koleo_loss_fn = KoLeoLoss()

    # ---- Optimizer ----
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=config["lr"], weight_decay=config["weight_decay"],
    )

    niter_per_ep = len(train_loader)
    lr_schedule = cosine_scheduler(config["lr"], config["final_lr"], config["epochs"], niter_per_ep, config["warmup_epochs"])
    wd_schedule = cosine_scheduler(config["weight_decay"], config["weight_decay_final"], config["epochs"], niter_per_ep)
    momentum_schedule = cosine_scheduler(config["teacher_momentum"], 1.0, config["epochs"], niter_per_ep)

    ibot_weight = config.get("ibot_loss_weight", 1.0)
    koleo_weight = config.get("koleo_loss_weight", 0.1)
    mask_ratio = config.get("ibot_mask_ratio", 0.3)
    grad_accum_steps = config.get("gradient_accumulation_steps", 1)
    clip_grad = config.get("clip_grad", 1.0)

    # ---- Resume ----
    start_epoch = 0
    ckpt_path = args.from_checkpoint or config.get("from_checkpoint")
    use_fp16 = config.get("use_fp16", False)
    autocast_dtype = torch.float16 if use_fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        student.load_state_dict(ckpt["student_state_dict"])
        teacher.load_state_dict(ckpt["teacher_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "dino_loss_state_dict" in ckpt:
            dino_loss_fn.load_state_dict(ckpt["dino_loss_state_dict"])
        if ibot_loss_fn and "ibot_loss_state_dict" in ckpt:
            ibot_loss_fn.load_state_dict(ckpt["ibot_loss_state_dict"])
        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"  Resumed at epoch {start_epoch}")

    init_experiment(config)
    experiment = comet_ml.get_global_experiment()
    run_name = get_run_name()

    ckpt_dir = config.get("checkpoints_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"Backbone: {backbone} | DINO: {dino_version} | embed={embed_dim} depth={depth} "
          f"heads={num_heads} patch={config.get('patch_size', 8)}")
    print(f"Crops: {global_crops_number}G + {local_crops_number}L | "
          f"Batch: {config['batch_size']} × {grad_accum_steps} accum")
    print(f"Training: epochs {start_epoch}..{config['epochs'] - 1} | {niter_per_ep} iters/ep")

    # ============================================================
    # TRAINING LOOP
    # ============================================================
    for epoch in range(start_epoch, config["epochs"]):
        student.train()
        teacher.eval()
        r_loss = r_dino = r_ibot = r_koleo = 0.0

        for batch_idx, (crops, channel_ids) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            gs = niter_per_ep * epoch + batch_idx
            si = min(gs, len(lr_schedule) - 1)

            for pg in optimizer.param_groups:
                pg["lr"] = lr_schedule[si]
                pg["weight_decay"] = wd_schedule[si]

            crops = [c.to(device, dtype=torch.float32, non_blocking=True) for c in crops]
            channel_ids = channel_ids.to(device, dtype=torch.long, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=autocast_dtype):

                if dino_version == "v2":
                    # === DINOv2 mode ===
                    with torch.no_grad():
                        t_out = teacher(torch.cat(crops[:global_crops_number]),
                                        channel_ids.repeat(global_crops_number, 1))
                    s_global = student(torch.cat(crops[:global_crops_number]),
                                       channel_ids.repeat(global_crops_number, 1))
                    if local_crops_number > 0:
                        s_local = student(torch.cat(crops[global_crops_number:]),
                                          channel_ids.repeat(local_crops_number, 1))
                        s_out = torch.cat([s_global, s_local])
                    else:
                        s_out = s_global
                    loss_dino = dino_loss_fn(s_out, t_out, epoch)
                    total_loss = loss_dino

                else:
                    # === DINOv3 mode ===
                    B = crops[0].shape[0]
                    h_p = crops[0].shape[2] // config.get("patch_size", 8)
                    w_p = crops[0].shape[3] // config.get("patch_size", 8)
                    n_patches = h_p * w_p
                    ibot_mask = generate_ibot_mask(B * global_crops_number, n_patches, mask_ratio, device)

                    # Teacher: global, no mask
                    with torch.no_grad():
                        t_in = torch.cat(crops[:global_crops_number])
                        t_cids = channel_ids.repeat(global_crops_number, 1)
                        t_feats = teacher.forward_features(t_in, t_cids, mask=None)
                        t_dino = teacher.dino_head(t_feats["cls_token"])
                        t_ibot = teacher.ibot_head(t_feats["patch_tokens"][ibot_mask])

                    # Student: global WITH mask
                    s_in = torch.cat(crops[:global_crops_number])
                    s_cids = channel_ids.repeat(global_crops_number, 1)
                    s_feats = student.forward_features(s_in, s_cids, mask=ibot_mask)
                    s_dino_global = student.dino_head(s_feats["cls_token"])
                    s_ibot = student.ibot_head(s_feats["patch_tokens"][ibot_mask])

                    # Student: local, no mask
                    if local_crops_number > 0:
                        s_local = student(torch.cat(crops[global_crops_number:]),
                                          channel_ids.repeat(local_crops_number, 1))
                        s_dino_all = torch.cat([s_dino_global, s_local])
                    else:
                        s_dino_all = s_dino_global

                    loss_dino = dino_loss_fn(s_dino_all, t_dino, epoch)
                    loss_ibot = ibot_loss_fn(s_ibot, t_ibot, epoch)
                    loss_koleo = koleo_loss_fn(s_feats["cls_token"])
                    total_loss = loss_dino + ibot_weight * loss_ibot + koleo_weight * loss_koleo

                    r_ibot += loss_ibot.item()
                    r_koleo += loss_koleo.item()

            acc_loss = total_loss / grad_accum_steps
            scaler.scale(acc_loss).backward()

            if ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(train_loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    m = momentum_schedule[si]
                    for ps, pt in zip(student.parameters(), teacher.parameters()):
                        pt.data.mul_(m).add_((1 - m) * ps.detach().data)

            r_loss += total_loss.item()
            r_dino += loss_dino.item()

            if (batch_idx + 1) % 10 == 0 and experiment is not None:
                log = {"train/total_loss": total_loss.item(), "train/dino_loss": loss_dino.item(),
                       "train/lr": lr_schedule[si], "train/momentum": momentum_schedule[si]}
                if dino_version == "v3":
                    log["train/ibot_loss"] = loss_ibot.item()
                    log["train/koleo_loss"] = loss_koleo.item()
                experiment.log_metrics(log, step=gs)

        # Validation
        student.eval()
        v_loss = 0.0
        with torch.no_grad():
            for crops, channel_ids in tqdm(val_loader, desc=f"Val {epoch}"):
                crops = [c.to(device, dtype=torch.float32, non_blocking=True) for c in crops]
                channel_ids = channel_ids.to(device, dtype=torch.long, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    t_out = teacher(torch.cat(crops[:global_crops_number]),
                                    channel_ids.repeat(global_crops_number, 1))
                    s_g = student(torch.cat(crops[:global_crops_number]),
                                  channel_ids.repeat(global_crops_number, 1))
                    if local_crops_number > 0:
                        s_l = student(torch.cat(crops[global_crops_number:]),
                                      channel_ids.repeat(local_crops_number, 1))
                        s_out = torch.cat([s_g, s_l])
                    else:
                        s_out = s_g
                    v_loss += dino_loss_fn(s_out, t_out, epoch, update_center=False).item()

        n_train = max(len(train_loader), 1)
        n_val = max(len(val_loader), 1)
        print(f"Epoch {epoch} | Train: {r_loss / n_train:.4f} (DINO: {r_dino / n_train:.4f}" +
              (f", iBOT: {r_ibot / n_train:.4f}, KoLeo: {r_koleo / n_train:.4f}" if dino_version == "v3" else "") +
              f") | Val: {v_loss / n_val:.4f}")

        if experiment is not None:
            log_ep = {"epoch/train_loss": r_loss / n_train, "epoch/val_loss": v_loss / n_val,
                      "epoch/dino_loss": r_dino / n_train}
            if dino_version == "v3":
                log_ep["epoch/ibot_loss"] = r_ibot / n_train
                log_ep["epoch/koleo_loss"] = r_koleo / n_train
            experiment.log_metrics(log_ep, epoch=epoch)

        if (epoch + 1) % config.get("save_checkpoint_freq", 10) == 0:
            save_dict = {
                "student_state_dict": student.state_dict(),
                "teacher_state_dict": teacher.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "dino_loss_state_dict": dino_loss_fn.state_dict(),
                "epoch": epoch, "config": config,
            }
            if ibot_loss_fn is not None:
                save_dict["ibot_loss_state_dict"] = ibot_loss_fn.state_dict()
            torch.save(save_dict, f"{ckpt_dir}/immukronos_{backbone}_{dino_version}-{run_name}-epoch_{epoch}.pth")

    # Final save
    torch.save({
        "student_state_dict": student.state_dict(),
        "teacher_state_dict": teacher.state_dict(),
        "epoch": config["epochs"] - 1, "config": config,
    }, f"{ckpt_dir}/immukronos_{backbone}_{dino_version}-{run_name}-final.pth")

    print("Training finished!")
    finish_experiment()


if __name__ == "__main__":
    main()
