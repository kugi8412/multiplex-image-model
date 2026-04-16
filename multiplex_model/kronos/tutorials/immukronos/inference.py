"""
Inference utilities for loading trained ImmuKronos (DINOv3) models
and extracting features for downstream tasks.

The trained ImmuvisDINOv3 model produces:
    - CLS token:     (B, embed_dim)       — global patch representation
    - Patch tokens:  (B, N_patches, embed_dim) — spatial token features

For compatibility with the original KRONOS downstream pipeline, we provide
a wrapper that returns the same 3-tuple:
    (patch_features, patch_marker_features, patch_token_features)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_immukronos_model(
    checkpoint_path: str,
    num_markers: int = 512,
    embed_dim: int = 384,
    depth: int = 12,
    num_heads: int = 6,
    patch_size: int = 8,
    out_dim: int = 65536,
    num_register_tokens: int = 4,
    ibot_out_dim: int = 8192,
    device: str = "cuda",
    key: str = "student_state_dict",
):
    """
    Load a trained ImmuvisDINOv3 model from a checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        num_markers: Number of markers in the tokenizer.
        embed_dim: Transformer embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        patch_size: Patch size used during training.
        out_dim: DINO head output dimension.
        num_register_tokens: Number of register tokens.
        ibot_out_dim: iBOT head output dimension.
        device: Device to load the model on.
        key: Key in checkpoint dict containing the state_dict.

    Returns:
        model: Loaded ImmuvisDINOv3 in eval mode.
        embed_dim: Embedding dimension.
    """
    # Import here to avoid circular imports
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    from train_kronos_immuvis_v3 import ImmuvisDINOv3

    model = ImmuvisDINOv3(
        num_markers=num_markers,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        patch_size=patch_size,
        out_dim=out_dim,
        drop_path_rate=0.0,
        num_register_tokens=num_register_tokens,
        ibot_out_dim=ibot_out_dim,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get(key, checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"\033[92mLoaded ImmuKronos-DINOv3 from {checkpoint_path}\033[0m")
    return model, embed_dim


class ImmuKronosWrapper(nn.Module):
    """
    Wrapper that makes ImmuvisDINOv3 return the same 3-tuple as original KRONOS:
        (patch_features, patch_marker_features, patch_token_features)

    This enables drop-in compatibility with all existing downstream pipelines.

    Uses marker-attributed projection: the Hyperkernel's learned per-marker
    embedding weights serve as query probes to extract marker-specific features
    from the fused patch token representation via cross-attention.

    Given input x of shape (B, num_markers, H, W):
        - patch_features:        (B, embed_dim)                          — CLS token
        - patch_marker_features: (B, num_markers, embed_dim)             — marker-attributed features
        - patch_token_features:  (B, num_markers, h_p, w_p, embed_dim)   — spatial per-marker features
    """

    def __init__(self, model, patch_size: int):
        super().__init__()
        self.model = model
        self.patch_size = patch_size

    def _marker_probes(self, marker_ids: torch.Tensor) -> torch.Tensor:
        """Build normalized marker probe vectors from Hyperkernel weights.

        Args:
            marker_ids: (B, M) marker token IDs

        Returns:
            (B, M, D) normalized marker probe vectors
        """
        # Hyperkernel weights: (B, M, input_dim * out_dim) where out_dim = D * K * K
        hk_weights = self.model.hyperkernel.hyperkernel_weights(marker_ids)
        B, M, _ = hk_weights.shape
        D = self.model.embed_dim
        # Reshape to (B, M, input_dim * K * K, D) and average over the kernel/input dims
        marker_probes = hk_weights.reshape(B, M, -1, D).mean(dim=2)  # (B, M, D)
        return F.normalize(marker_probes, dim=-1)

    @torch.no_grad()
    def forward(self, x, marker_ids=None):
        """
        Args:
            x: (B, num_markers, H, W) multiplex image
            marker_ids: (B, num_markers) marker token IDs

        Returns:
            Tuple of (patch_features, patch_marker_features, patch_token_features)
        """
        B, num_markers, H, W = x.shape
        h_p = H // self.patch_size
        w_p = W // self.patch_size

        features = self.model.forward_features(x, marker_ids, mask=None)

        # CLS token — global patch embedding
        patch_features = features["cls_token"]  # (B, D)

        # Patch tokens — (B, h_p*w_p, D)
        patch_tokens = features["patch_tokens"]  # (B, N, D)
        D = patch_tokens.shape[-1]

        # --- Marker-attributed projection ---
        # Use Hyperkernel weights as per-marker query probes to extract
        # marker-specific features from the fused spatial tokens.
        marker_probes = self._marker_probes(marker_ids)  # (B, M, D)

        # Cross-attention: each marker probe attends over spatial patch tokens
        # attn_weights: (B, M, N)
        attn_logits = torch.bmm(marker_probes, patch_tokens.transpose(1, 2))
        attn_weights = F.softmax(attn_logits / (D ** 0.5), dim=-1)

        # patch_marker_features: weighted average of patch tokens per marker (B, M, D)
        patch_marker_features = torch.bmm(attn_weights, patch_tokens)  # (B, M, D)

        # patch_token_features: per-marker spatially resolved features (B, M, h_p, w_p, D)
        # Weight each spatial token by its marker-specific attention score
        spatial = patch_tokens.reshape(B, h_p, w_p, D)
        # attn_weights reshaped: (B, M, h_p, w_p) — per-marker spatial importance
        attn_spatial = attn_weights.reshape(B, num_markers, h_p, w_p)
        # Scale spatial features by marker attention: (B, M, h_p, w_p, D)
        patch_token_features = spatial.unsqueeze(1) * attn_spatial.unsqueeze(-1)

        return patch_features, patch_marker_features, patch_token_features
