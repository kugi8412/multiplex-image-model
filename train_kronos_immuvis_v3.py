#!/usr/bin/env python
# -*- coding: utf-8 -*-
# train_kronos_immuvis_v3.py
#
# ImmuKRONOS DINOv3 training with Hyperkernel stem + RoPE/RMSNorm/SwiGLU.
# Supports channel_fraction dropout (like standard KRONOS) when set in config.
#
# Usage:
#   python train_kronos_immuvis_v3.py configs/exp6b_immukronos_v3.yaml

import functools
import os
import random
import sys

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

# Imports from ImmuVis ecosystem
from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler
from multiplex_model.utils import init_experiment, finish_experiment, get_run_name
from multiplex_model.modules.immuvis import Hyperkernel
from multiplex_model.kronos.dino_head import DINOHead

# ------------------------------------------
# 1. DINOv3 BUILDING BLOCKS
# ------------------------------------------

class RMSNorm(nn.Module):
    """RMSNorm — DINOv3 replaces LayerNorm with RMSNorm."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


class RotaryPositionEmbedding(nn.Module):
    """2D Rotary Position Embedding for vision transformers (DINOv3-style).
    Applies RoPE to queries and keys. No learnable parameters."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
        self.head_dim = head_dim
        self.half_dim = head_dim // 4  # each spatial axis gets head_dim/4 sin/cos pairs
        inv_freq = 1.0 / (base ** (torch.arange(0, self.half_dim, dtype=torch.float32) / self.half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _build_freqs(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Build [h*w, head_dim] frequency table for 2D grid."""
        y = torch.arange(h, device=device, dtype=torch.float32)
        x = torch.arange(w, device=device, dtype=torch.float32)
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        gy, gx = gy.reshape(-1), gx.reshape(-1)

        inv = self.inv_freq.to(device)
        freqs_y = torch.outer(gy, inv)  # [N, half_dim]
        freqs_x = torch.outer(gx, inv)  # [N, half_dim]

        # [sin_y, cos_y, sin_x, cos_x] each of size half_dim
        freqs = torch.cat([freqs_y, freqs_y, freqs_x, freqs_x], dim=-1)  # [N, head_dim]
        return freqs.to(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, h: int, w: int, num_prefix: int = 0):
        """Apply 2D RoPE to q,k. Shape: [B, heads, N, head_dim].
        num_prefix = number of prefix tokens (CLS + registers) to skip."""
        freqs = self._build_freqs(h, w, q.device, q.dtype)
        cos_f = torch.cos(freqs)
        sin_f = torch.sin(freqs)

        def rotate(t: torch.Tensor) -> torch.Tensor:
            t1, t2 = t[..., ::2], t[..., 1::2]
            return torch.stack((-t2, t1), dim=-1).flatten(-2)

        if num_prefix > 0:
            q_prefix, q_patch = q[:, :, :num_prefix], q[:, :, num_prefix:]
            k_prefix, k_patch = k[:, :, :num_prefix], k[:, :, num_prefix:]
        else:
            q_prefix = k_prefix = None
            q_patch, k_patch = q, k

        q_patch = q_patch * cos_f + rotate(q_patch) * sin_f
        k_patch = k_patch * cos_f + rotate(k_patch) * sin_f

        if num_prefix > 0:
            q = torch.cat([q_prefix, q_patch], dim=2)
            k = torch.cat([k_prefix, k_patch], dim=2)
        else:
            q, k = q_patch, k_patch

        return q, k


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN — DINOv3 default feedforward layer."""
    def __init__(self, in_features: int, hidden_features: int = None,
                 out_features: int = None, act_layer=None, drop: float = 0.0, bias: bool = True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class DropPath(nn.Module):
    """Stochastic depth (drop path)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        random_tensor.div_(keep_prob)
        return x * random_tensor


class DINOv3Attention(nn.Module):
    """Attention with RoPE. No QKV bias by default (DINOv3)."""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 proj_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, rope: RotaryPositionEmbedding = None,
                h: int = None, w: int = None, num_prefix: int = 0) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if rope is not None and h is not None and w is not None:
            q, k = rope(q, k, h, w, num_prefix=num_prefix)

        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DINOv3Block(nn.Module):
    """Transformer block: RMSNorm + RoPE Attention + SwiGLU + DropPath."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = False, proj_bias: bool = True,
                 drop_path: float = 0.0, init_values: float = None):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = DINOv3Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLUFFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ls1 = nn.Parameter(init_values * torch.ones(dim)) if init_values else None
        self.ls2 = nn.Parameter(init_values * torch.ones(dim)) if init_values else None

    def forward(self, x: torch.Tensor, rope: RotaryPositionEmbedding = None,
                h: int = None, w: int = None, num_prefix: int = 0) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x), rope=rope, h=h, w=w, num_prefix=num_prefix)
        if self.ls1 is not None:
            attn_out = attn_out * self.ls1
        x = x + self.drop_path(attn_out)

        mlp_out = self.mlp(self.norm2(x))
        if self.ls2 is not None:
            mlp_out = mlp_out * self.ls2
        x = x + self.drop_path(mlp_out)
        return x


# ------------------------------------------
# 2. KRONOS-IMMUVIS DINOv3 MODEL
# ------------------------------------------

class ImmuvisDINOv3(nn.Module):
    """
    KRONOS-ImmuVis model with DINOv3 architecture.

    Upgrades over DINOv2:
    - RMSNorm instead of LayerNorm
    - SwiGLU FFN instead of standard MLP
    - 2D Rotary Position Embedding (RoPE) — no learnable pos_embed
    - No QKV bias
    - Native register (storage) tokens
    - Separate DINO (CLS) and iBOT (patch) projection heads
    """

    def __init__(self, num_markers: int, embed_dim: int = 768, depth: int = 12,
                 num_heads: int = 12, patch_size: int = 16, out_dim: int = 65536,
                 drop_path_rate: float = 0.0, num_register_tokens: int = 4,
                 ibot_out_dim: int = 8192, init_values: float = None,
                 mask_strategy: str = "learnable"):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_register_tokens = num_register_tokens
        self.mask_strategy = mask_strategy

        # 1. STEM: Hyperkernel (marker-aware patch embedding)
        self.hyperkernel = Hyperkernel(
            num_channels=num_markers,
            input_dim=1,
            embedding_dim=embed_dim,
            module_type="encoder",
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        # 2. Special tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens > 0 else None
        )

        # 3. RoPE (replaces learnable pos_embed)
        head_dim = embed_dim // num_heads
        self.rope = RotaryPositionEmbedding(head_dim=head_dim)

        # 4. Transformer blocks with stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            DINOv3Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0,
                qkv_bias=False, drop_path=dpr[i], init_values=init_values,
            )
            for i in range(depth)
        ])
        self.norm = RMSNorm(embed_dim)

        # 5. DINO head (CLS token projection)
        self.dino_head = DINOHead(in_dim=embed_dim, out_dim=out_dim)

        # 6. iBOT head (patch token projection for masked prediction)
        self.ibot_head = DINOHead(in_dim=embed_dim, out_dim=ibot_out_dim)

        # 7. Mask token for iBOT — strategy-dependent
        if mask_strategy == "learnable":
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        elif mask_strategy == "zero":
            self.register_buffer("mask_token", torch.zeros(1, 1, embed_dim))
        elif mask_strategy == "negative":
            self.register_buffer("mask_token", torch.full((1, 1, embed_dim), -1.0))
        else:
            raise ValueError(f"Unknown mask_strategy '{mask_strategy}'. Use 'learnable', 'zero', or 'negative'.")

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.mask_strategy == "learnable":
            nn.init.normal_(self.mask_token, std=0.02)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)

    @property
    def num_prefix_tokens(self) -> int:
        return 1 + (self.num_register_tokens if self.register_tokens is not None else 0)

    def forward_features(self, x: torch.Tensor, channel_ids: torch.Tensor,
                         mask: torch.Tensor = None):
        """
        Args:
            x: [B, C, H, W] multiplex image
            channel_ids: [B, C] marker token IDs
            mask: [B, N_patches] boolean mask for iBOT (True = masked)
        Returns:
            dict with cls_token [B, D] and patch_tokens [B, N, D]
        """
        B, C, H, W = x.shape
        h_p, w_p = H // self.patch_size, W // self.patch_size

        # Hyperkernel: per-marker patch embedding -> spatial fusion
        x_hk = x.reshape(B * C, 1, H, W)
        x_enc = self.hyperkernel(x_hk, channel_ids)  # [B, embed_dim, h_p, w_p]
        x_enc = x_enc.flatten(2).transpose(1, 2)  # [B, N, embed_dim]

        # Apply iBOT masking
        if mask is not None:
            mask_value = self.mask_token.squeeze(0).squeeze(0).to(x_enc.dtype)
            x_enc = torch.where(mask.unsqueeze(-1), mask_value.unsqueeze(0).unsqueeze(0).expand_as(x_enc), x_enc)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_enc = torch.cat([cls_tokens, x_enc], dim=1)

        # Insert register tokens after CLS
        if self.register_tokens is not None:
            reg = self.register_tokens.expand(B, -1, -1)
            x_enc = torch.cat([x_enc[:, :1], reg, x_enc[:, 1:]], dim=1)

        num_prefix = self.num_prefix_tokens

        for blk in self.blocks:
            x_enc = blk(x_enc, rope=self.rope, h=h_p, w=w_p, num_prefix=num_prefix)

        x_enc = self.norm(x_enc)

        return {
            "cls_token": x_enc[:, 0],
            "patch_tokens": x_enc[:, num_prefix:],
        }

    def forward(self, x: torch.Tensor, channel_ids: torch.Tensor,
                mask: torch.Tensor = None, return_ibot: bool = False):
        features = self.forward_features(x, channel_ids, mask=mask)
        dino_out = self.dino_head(features["cls_token"])

        if return_ibot and mask is not None:
            masked_patches = features["patch_tokens"][mask]
            ibot_out = self.ibot_head(masked_patches)
            return dino_out, ibot_out

        return dino_out

# ------------------------------------------
# 3. DINOv3 LOSSES
# ------------------------------------------

class DINOLoss(nn.Module):
    def __init__(self, out_dim: int, ncrops: int, warmup_teacher_temp: float,
                 teacher_temp: float, warmup_teacher_temp_epochs: int, nepochs: int,
                 student_temp: float = 0.1, center_momentum: float = 0.9,
                 nglobal_crops: int = 2):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.nglobal_crops = nglobal_crops
        self.register_buffer("center", torch.zeros(1, out_dim))

        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch, update_center=True):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        temp = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.nglobal_crops)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms

        if update_center:
            self.update_center(teacher_output)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True) / teacher_output.shape[0]
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class iBOTLoss(nn.Module):
    """iBOT patch-level self-distillation loss (DINOv3 component)."""
    def __init__(self, out_dim: int, warmup_teacher_temp: float, teacher_temp: float,
                 warmup_teacher_temp_epochs: int, nepochs: int,
                 student_temp: float = 0.1, center_momentum: float = 0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_patch_logits, teacher_patch_logits, epoch, update_center=True):
        """Both inputs: [M, out_dim] where M = number of masked patches across batch."""
        if student_patch_logits.shape[0] == 0:
            return student_patch_logits.sum() * 0

        temp = self.teacher_temp_schedule[min(epoch, len(self.teacher_temp_schedule) - 1)]
        student_out = student_patch_logits / self.student_temp
        teacher_out = F.softmax((teacher_patch_logits - self.center) / temp, dim=-1).detach()
        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)

        if update_center:
            self._update_center(teacher_patch_logits)

        return loss.mean()

    @torch.no_grad()
    def _update_center(self, teacher_output):
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class KoLeoLoss(nn.Module):
    """KoLeo diversity regularizer — encourages uniform CLS feature spacing."""
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=-1, p=2)
        dists = torch.cdist(features, features)
        diag_mask = torch.eye(dists.shape[0], dtype=torch.bool, device=dists.device)
        dists = dists.masked_fill(diag_mask, float("inf"))
        min_dists = dists.min(dim=-1).values
        return -torch.log(min_dists + 1e-8).mean()


class GramLoss(nn.Module):
    """Gram anchoring loss (DINOv3) — anchors student patch feature correlations
    to a frozen Gram teacher to prevent dense feature degradation.
    Only unmasked patches are used to avoid correlating mask-token artifacts."""
    def forward(self, student_patches: torch.Tensor, gram_teacher_patches: torch.Tensor,
                unmasked_mask: torch.Tensor = None) -> torch.Tensor:
        if unmasked_mask is not None:
            student_patches = student_patches[unmasked_mask].unsqueeze(0) if student_patches.dim() == 2 else torch.stack([sp[um] for sp, um in zip(student_patches, unmasked_mask)])
            gram_teacher_patches = gram_teacher_patches[unmasked_mask].unsqueeze(0) if gram_teacher_patches.dim() == 2 else torch.stack([tp[um] for tp, um in zip(gram_teacher_patches, unmasked_mask)])
        s = F.normalize(student_patches, dim=-1)
        t = F.normalize(gram_teacher_patches, dim=-1)
        gram_s = torch.bmm(s, s.transpose(1, 2))
        gram_t = torch.bmm(t, t.transpose(1, 2))
        return F.mse_loss(gram_s, gram_t)

# ------------------------------------------
# 4. MULTI-CROP AUGMENTATION & DATA
# ------------------------------------------

class MultiCropTransform:
    """Multi-crop views with rotation for spatial proteomics data."""
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
        crops = self.transform(img)
        return crops, channel_ids, panel_idx, img_path

def dino_collate_fn(batch, channel_fraction=None):
    """Collate multi-crop batch with optional channel dropout.

    If channel_fraction is set (e.g. (0.75, 1.0)), randomly drops a fraction
    of channels uniformly across the entire batch. All samples in a
    PanelBatchSampler batch share the same marker set.
    """
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


# ------------------------------------------
# 5. iBOT MASKING UTILITIES
# ------------------------------------------

def generate_ibot_mask(batch_size: int, num_patches: int, mask_ratio: float = 0.3,
                       device: torch.device = None) -> torch.Tensor:
    """Generate random boolean mask for iBOT. True = masked."""
    num_masked = int(num_patches * mask_ratio)
    mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
    for i in range(batch_size):
        indices = torch.randperm(num_patches, device=device)[:num_masked]
        mask[i, indices] = True
    return mask


# ------------------------------------------
# 6. MAIN TRAINING LOOP
# ------------------------------------------
def main():
    config_path = sys.argv[1]
    yaml = YAML(typ="safe")
    with open(config_path, "r") as f:
        config = yaml.load(f)

    device = torch.device(config.get("device", "cuda"))
    print(f"Using device: {device}")

    PANEL_CONFIG = YAML().load(open(config['panel_config_path']))
    TOKENIZER = YAML().load(open(config['tokenizer_config_path']))
    num_markers = len(TOKENIZER)

    global_crops_number = config.get('global_crops_number', 2)
    local_crops_number = config['local_crops_number']
    ncrops = global_crops_number + local_crops_number

    # Channel dropout
    channel_fraction = config.get("channel_fraction", None)
    if channel_fraction is not None:
        channel_fraction = tuple(channel_fraction)
        print(f"Channel dropout enabled: keep fraction {channel_fraction}")
        collate_fn = functools.partial(dino_collate_fn, channel_fraction=channel_fraction)
    else:
        collate_fn = dino_collate_fn

    dino_transform = MultiCropTransform(
        global_size=config['global_crops_size'],
        local_size=config['local_crops_size'],
        global_scale=config['global_crops_scale'],
        local_scale=config['local_crops_scale'],
        local_crops_number=local_crops_number,
        global_crops_number=global_crops_number,
    )

    train_dataset_base = DatasetFromTIFF(
        panels_config=PANEL_CONFIG, split="train", marker_tokenizer=TOKENIZER,
        transform=None, use_preprocessing=False, use_butterworth_filter=True,
        use_clip_normalization=True, file_extension="npy",
    )
    train_dataset = DINODatasetWrapper(train_dataset_base, dino_transform)
    train_sampler = PanelBatchSampler(train_dataset_base, config['batch_size'])
    train_dataloader = DataLoader(
        train_dataset, batch_sampler=train_sampler,
        num_workers=config.get('num_workers', 4), collate_fn=collate_fn,
        pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )

    val_dataset_base = DatasetFromTIFF(
        panels_config=PANEL_CONFIG, split="test", marker_tokenizer=TOKENIZER,
        transform=None, use_preprocessing=False, use_butterworth_filter=True,
        use_clip_normalization=True, file_extension="npy",
    )
    val_dataset = DINODatasetWrapper(val_dataset_base, dino_transform)
    val_sampler = PanelBatchSampler(val_dataset_base, config['batch_size'], shuffle=False)
    val_dataloader = DataLoader(
        val_dataset, batch_sampler=val_sampler,
        num_workers=config.get('num_workers', 4), collate_fn=dino_collate_fn,
        pin_memory=True,
    )

    # Model Config
    embed_dim = config.get('embed_dim', 768)
    depth = config.get('depth', 12)
    num_heads = config.get('num_heads', 12)
    drop_path_rate = config.get('drop_path_rate', 0.1)
    num_register_tokens = config.get('num_register_tokens', 4)
    ibot_out_dim = config.get('ibot_out_dim', 8192)
    init_values = config.get('init_values', None)
    mask_strategy = config.get('mask_strategy', 'learnable')

    # Student (with DropPath)
    student = ImmuvisDINOv3(
        num_markers=num_markers, embed_dim=embed_dim, depth=depth,
        num_heads=num_heads, patch_size=config['patch_size'],
        out_dim=config['out_dim'], drop_path_rate=drop_path_rate,
        num_register_tokens=num_register_tokens,
        ibot_out_dim=ibot_out_dim, init_values=init_values,
        mask_strategy=mask_strategy,
    ).to(device)

    # Teacher (no DropPath — EMA target must be deterministic)
    teacher = ImmuvisDINOv3(
        num_markers=num_markers, embed_dim=embed_dim, depth=depth,
        num_heads=num_heads, patch_size=config['patch_size'],
        out_dim=config['out_dim'], drop_path_rate=0.0,
        num_register_tokens=num_register_tokens,
        ibot_out_dim=ibot_out_dim, init_values=init_values,
        mask_strategy=mask_strategy,
    ).to(device)

    teacher.load_state_dict(student.state_dict(), strict=False)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    # Losses
    dino_loss_fn = DINOLoss(
        out_dim=config['out_dim'], ncrops=ncrops,
        warmup_teacher_temp=config['warmup_teacher_temp'],
        teacher_temp=config['teacher_temp'],
        warmup_teacher_temp_epochs=config['warmup_teacher_temp_epochs'],
        nepochs=config['epochs'], nglobal_crops=global_crops_number,
    ).to(device)

    ibot_loss_fn = iBOTLoss(
        out_dim=ibot_out_dim,
        warmup_teacher_temp=config.get('ibot_warmup_teacher_temp', config['warmup_teacher_temp']),
        teacher_temp=config.get('ibot_teacher_temp', config['teacher_temp']),
        warmup_teacher_temp_epochs=config['warmup_teacher_temp_epochs'],
        nepochs=config['epochs'],
    ).to(device)

    koleo_loss_fn = KoLeoLoss()

    # Optional Gram teacher
    use_gram = config.get('use_gram_loss', False)
    gram_loss_fn = GramLoss() if use_gram else None
    gram_teacher = None
    if use_gram:
        gram_checkpoint = config.get('gram_teacher_checkpoint', None)
        if gram_checkpoint and os.path.exists(gram_checkpoint):
            gram_teacher = ImmuvisDINOv3(
                num_markers=num_markers, embed_dim=embed_dim, depth=depth,
                num_heads=num_heads, patch_size=config['patch_size'],
                out_dim=config['out_dim'], drop_path_rate=0.0,
                num_register_tokens=num_register_tokens,
                ibot_out_dim=ibot_out_dim,
                mask_strategy=mask_strategy,
            ).to(device)
            ckpt = torch.load(gram_checkpoint, map_location=device, weights_only=True)
            gram_teacher.load_state_dict(ckpt.get("student_state_dict", ckpt), strict=False)
            for p in gram_teacher.parameters():
                p.requires_grad = False
            gram_teacher.eval()
            print(f"Loaded Gram teacher from {gram_checkpoint}")
        else:
            print("Gram loss enabled but no valid checkpoint — disabling Gram loss.")
            use_gram = False

    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=config['lr'], weight_decay=config['weight_decay'],
    )

    niter_per_ep = len(train_dataloader)
    lr_schedule = cosine_scheduler(config['lr'], config['final_lr'], config['epochs'], niter_per_ep, config['warmup_epochs'])
    wd_schedule = cosine_scheduler(config['weight_decay'], config['weight_decay_final'], config['epochs'], niter_per_ep)
    momentum_schedule = cosine_scheduler(config['teacher_momentum'], 1.0, config['epochs'], niter_per_ep)

    # Loss weights
    ibot_weight = config.get('ibot_loss_weight', 1.0)
    koleo_weight = config.get('koleo_loss_weight', 0.1)
    gram_weight = config.get('gram_loss_weight', 0.1)
    mask_ratio = config.get('ibot_mask_ratio', 0.3)

    # Checkpoints resume
    start_epoch = 0
    checkpoint_path = config.get("from_checkpoint", None)

    use_fp16 = config.get('use_fp16', False)
    autocast_dtype = torch.float16 if use_fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        student.load_state_dict(checkpoint["student_state_dict"])
        teacher.load_state_dict(checkpoint["teacher_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if "dino_loss_state_dict" in checkpoint:
            dino_loss_fn.load_state_dict(checkpoint["dino_loss_state_dict"])
        if "ibot_loss_state_dict" in checkpoint:
            ibot_loss_fn.load_state_dict(checkpoint["ibot_loss_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")
    elif checkpoint_path is not None:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Starting from scratch.")

    init_experiment(config)
    experiment = comet_ml.get_global_experiment()
    run_name = get_run_name()

    checkpoints_path = config.get("checkpoints_dir", "checkpoints")
    os.makedirs(checkpoints_path, exist_ok=True)

    print("Starting DINOv3 KRONOS-IMMUVIS Training...")
    grad_accum_steps = config.get('gradient_accumulation_steps', 1)

    # Training loop
    for epoch in range(start_epoch, config['epochs']):
        student.train()
        train_loss_acc = 0.0
        train_dino_acc = 0.0
        train_ibot_acc = 0.0

        for batch_idx, (crops, channel_ids) in enumerate(tqdm(train_dataloader, desc=f"Train Epoch {epoch}")):
            global_step = niter_per_ep * epoch + batch_idx

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule[global_step]
                param_group["weight_decay"] = wd_schedule[global_step]

            crops = [c.to(device, dtype=torch.float32, non_blocking=True) for c in crops]
            channel_ids = channel_ids.to(device, dtype=torch.long, non_blocking=True)

            # Generate iBOT mask for global crops only
            B = crops[0].shape[0]
            h_p = crops[0].shape[2] // config['patch_size']
            w_p = crops[0].shape[3] // config['patch_size']
            n_patches_global = h_p * w_p
            ibot_mask = generate_ibot_mask(
                B * global_crops_number, n_patches_global,
                mask_ratio=mask_ratio, device=device,
            )

            with torch.amp.autocast('cuda', dtype=autocast_dtype):
                # TEACHER (global crops, no masking, no grad)
                with torch.no_grad():
                    teacher_global_in = torch.cat(crops[:global_crops_number])
                    teacher_cids = channel_ids.repeat(global_crops_number, 1)
                    teacher_feats = teacher.forward_features(teacher_global_in, teacher_cids, mask=None)
                    teacher_dino_out = teacher.dino_head(teacher_feats["cls_token"])
                    teacher_ibot_out = teacher.ibot_head(teacher_feats["patch_tokens"][ibot_mask])

                # STUDENT: global crops WITH iBOT masking
                student_global_in = torch.cat(crops[:global_crops_number])
                student_cids_global = channel_ids.repeat(global_crops_number, 1)
                student_global_feats = student.forward_features(
                    student_global_in, student_cids_global, mask=ibot_mask
                )
                student_dino_global = student.dino_head(student_global_feats["cls_token"])
                student_ibot_out = student.ibot_head(student_global_feats["patch_tokens"][ibot_mask])

                # STUDENT: local crops (no masking, DINO only)
                if local_crops_number > 0:
                    student_local_in = torch.cat(crops[global_crops_number:])
                    student_cids_local = channel_ids.repeat(local_crops_number, 1)
                    student_local_out = student(student_local_in, student_cids_local, mask=None)
                    student_dino_all = torch.cat([student_dino_global, student_local_out])
                else:
                    student_dino_all = student_dino_global

                # === Losses ===
                loss_dino = dino_loss_fn(student_dino_all, teacher_dino_out, epoch)
                loss_ibot = ibot_loss_fn(student_ibot_out, teacher_ibot_out, epoch)
                loss_koleo = koleo_loss_fn(student_global_feats["cls_token"])

                total_loss = loss_dino + ibot_weight * loss_ibot + koleo_weight * loss_koleo

                # Optional Gram anchoring
                if use_gram and gram_teacher is not None:
                    with torch.no_grad():
                        gram_feats = gram_teacher.forward_features(
                            student_global_in, student_cids_global, mask=None
                        )
                    unmasked = ~ibot_mask
                    loss_gram = gram_loss_fn(
                        student_global_feats["patch_tokens"],
                        gram_feats["patch_tokens"],
                        unmasked_mask=unmasked,
                    )
                    total_loss = total_loss + gram_weight * loss_gram

            accumulated_loss = total_loss / grad_accum_steps
            scaler.scale(accumulated_loss).backward()

            if ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(train_dataloader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), config.get('clip_grad', 1.0))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # EMA update of teacher
                with torch.no_grad():
                    m = momentum_schedule[global_step]
                    for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            train_loss_acc += total_loss.item()
            train_dino_acc += loss_dino.item()
            train_ibot_acc += loss_ibot.item()

            if (batch_idx + 1) % 10 == 0 and experiment is not None:
                log_dict = {
                    "train/total_loss": total_loss.item(),
                    "train/dino_loss": loss_dino.item(),
                    "train/ibot_loss": loss_ibot.item(),
                    "train/koleo_loss": loss_koleo.item(),
                    "train/lr": lr_schedule[global_step],
                    "train/teacher_momentum": momentum_schedule[global_step],
                }
                if use_gram and gram_teacher is not None:
                    log_dict["train/gram_loss"] = loss_gram.item()
                experiment.log_metrics(log_dict, step=global_step)

        # Validation
        student.eval()
        val_loss_acc = 0.0
        with torch.no_grad():
            for crops, channel_ids in tqdm(val_dataloader, desc=f"Val Epoch {epoch}"):
                crops = [c.to(device, dtype=torch.float32, non_blocking=True) for c in crops]
                channel_ids = channel_ids.to(device, dtype=torch.long, non_blocking=True)

                with torch.amp.autocast('cuda', dtype=autocast_dtype):
                    teacher_global_in = torch.cat(crops[:global_crops_number])
                    teacher_cids = channel_ids.repeat(global_crops_number, 1)
                    teacher_out = teacher(teacher_global_in, teacher_cids)

                    student_global = student(
                        torch.cat(crops[:global_crops_number]),
                        channel_ids.repeat(global_crops_number, 1),
                    )
                    if local_crops_number > 0:
                        student_local = student(
                            torch.cat(crops[global_crops_number:]),
                            channel_ids.repeat(local_crops_number, 1),
                        )
                        student_out = torch.cat([student_global, student_local])
                    else:
                        student_out = student_global

                    val_loss = dino_loss_fn(student_out, teacher_out, epoch, update_center=False)

                val_loss_acc += val_loss.item()

        avg_train = train_loss_acc / len(train_dataloader)
        avg_dino = train_dino_acc / len(train_dataloader)
        avg_ibot = train_ibot_acc / len(train_dataloader)
        avg_val = val_loss_acc / len(val_dataloader)

        print(f"Epoch {epoch} | Train: {avg_train:.4f} (DINO: {avg_dino:.4f}, iBOT: {avg_ibot:.4f}) | Val: {avg_val:.4f}")

        if experiment is not None:
            experiment.log_metrics({
                "epoch/train_loss": avg_train,
                "epoch/dino_loss": avg_dino,
                "epoch/ibot_loss": avg_ibot,
                "epoch/val_dino_loss": avg_val,
            }, epoch=epoch)

        if (epoch + 1) % config.get("save_checkpoint_freq", 5) == 0:
            torch.save({
                "student_state_dict": student.state_dict(),
                "teacher_state_dict": teacher.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "dino_loss_state_dict": dino_loss_fn.state_dict(),
                "ibot_loss_state_dict": ibot_loss_fn.state_dict(),
                "epoch": epoch,
            }, f"{checkpoints_path}/kronos_immuvis_dinov3-{run_name}-epoch_{epoch}.pth")

    print("Training finished!")
    torch.save(student.state_dict(), f"{checkpoints_path}/kronos_immuvis_dinov3-{run_name}-final.pth")
    finish_experiment()


if __name__ == "__main__":
    main()
