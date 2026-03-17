import torch
import torch.nn as nn

from .base_modules import Encoder
from .registry import ENCODER_REGISTRY

try:
    import timm
except ImportError:
    timm = None


@ENCODER_REGISTRY.register("dino")
class DinoEncoder(Encoder):
    """
    DINO Encoder wrapper using the `timm` library, supporting both v2 and v3.
    Adapts isotropic ViT outputs to spatial feature maps [B, C, H, W]
    and optionally projects the embedding dimension to a target size.
    """

    def __init__(
        self,
        input_channels: int = 1,
        version: str = "v3",        
        model_size: str = "base",  
        freeze_backbone: bool = False,
        target_dim: int | None = None,
        img_size: int = 128,
        **kwargs,
    ):
        super().__init__()
        
        if timm is None:
            raise ImportError("Library 'timm' is required for DinoEncoder!")

        self.freeze_backbone = freeze_backbone

        if version == "v2":
            size_map = {
                "small": "vit_small_patch14_dinov2.lvd142m",
                "base": "vit_base_patch14_dinov2.lvd142m",
                "large": "vit_large_patch14_dinov2.lvd142m"
            }
        elif version == "v3":
            size_map = {
                "small": "vit_small_patch16_dinov3.lvd1689m", # DINOv3 small
                "base": "vit_base_patch16_dinov3.lvd1689m",   # DINOv3 base (Latent Dim = 768)
                "large": "vit_large_patch16_dinov3.lvd1689m"  # DINOv3 large
            }
        else:
            raise ValueError(f"Wrong DINO version: {version}. Choose 'v2' or 'v3'.")
            
        model_name = size_map.get(model_size)

        if model_name is None:
            raise ValueError(f"Wrong DINO size: {model_size}. Choose 'small', 'base' lub 'large'.")
       
        self.backbone = timm.create_model(
            model_name, 
            pretrained=True, 
            num_classes=0, 
            in_chans=input_channels, 
            img_size=img_size
        )
        
        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.native_embed_dim = self.backbone.embed_dim
        
        # Freez
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

            self.backbone.eval()
                
        self.embed_dim = target_dim if target_dim is not None else self.native_embed_dim
        if target_dim is not None and target_dim != self.native_embed_dim:
            self.proj = nn.Conv2d(self.native_embed_dim, self.embed_dim, kernel_size=1)
        else:
            self.proj = nn.Identity()

    def train(self, mode: bool = True):
        """Overwrite train() for reeze_backbone=True."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()

        return self

    def forward(self, x: torch.Tensor, return_features: bool = False) -> dict:
        outputs = {}
        B, C, H, W = x.shape
        
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Image shape ({H}x{W}) must be divided by patch size for"
                f"DINO model: {self.patch_size} ({self.patch_size}). Change 'input_image_size'!"
            )
            
        h_feat, w_feat = H // self.patch_size, W // self.patch_size
        
        features = self.backbone.forward_features(x)
        
        if features.dim() == 3:
            num_spatial_tokens = h_feat * w_feat
            patch_tokens = features[:, -num_spatial_tokens:, :] 
            spatial_features = patch_tokens.transpose(1, 2).reshape(B, self.native_embed_dim, h_feat, w_feat)
        elif features.dim() == 4:
            spatial_features = features

        spatial_features = self.proj(spatial_features)

        outputs["output"] = spatial_features
        if return_features:
            outputs["features"] = [spatial_features]
            
        return outputs
