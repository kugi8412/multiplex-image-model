import torch
import torch.nn as nn

from .base_modules import Encoder
from .registry import ENCODER_REGISTRY

try:
    import timm
    from timm.layers import PatchEmbed, trunc_normal_
except ImportError:
    timm = None


@ENCODER_REGISTRY.register("dino")
class DinoEncoder(Encoder):
    """
    DINO Encoder wrapper using the `timm` library, supporting both v2 and v3.
    Adapts isotropic ViT outputs to spatial feature maps [B, C, H, W]
    and optionally projects the embedding dimension to a target size.
    Dynamically supports overriding the native patch size.
    """

    def __init__(
        self,
        input_channels: int = 1,
        version: str = "v3",        
        model_size: str = "base",  
        freeze_backbone: bool = False,
        target_dim: int | None = None,
        img_size: int = 128,
        patch_size: int | None = None,  # <--- DODANO ODBIÓR Z YAMLA
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
                "small": "vit_small_patch16_dinov3.lvd1689m", 
                "base": "vit_base_patch16_dinov3.lvd1689m",   
                "large": "vit_large_patch16_dinov3.lvd1689m"  
            }
        else:
            raise ValueError(f"Wrong DINO version: {version}. Choose 'v2' or 'v3'.")
            
        model_name = size_map.get(model_size)

        if model_name is None:
            raise ValueError(f"Wrong DINO size: {model_size}. Choose 'small', 'base' lub 'large'.")
       
        # KROK 1: Ładujemy bazowy model (np. z łatką 16)
        self.backbone = timm.create_model(
            model_name, 
            pretrained=True, 
            num_classes=0, 
            in_chans=input_channels, 
            img_size=img_size
        )
        
        native_patch_size = self.backbone.patch_embed.patch_size[0]
        self.patch_size = patch_size if patch_size is not None else native_patch_size
        self.native_embed_dim = self.backbone.embed_dim
        
        # KROK 2: Chirurgia modelu (Jeśli YAML wymusza łatkę 8, a model ma 16)
        # KROK 2: Chirurgia modelu (Jeśli YAML wymusza łatkę 8, a model ma 16)
        if self.patch_size != native_patch_size:
            print(f"🔧 Adapting DINO from native patch {native_patch_size}x{native_patch_size} to requested {self.patch_size}x{self.patch_size}...")
            
            old_pe = self.backbone.patch_embed
            
            # Zgrywamy oryginalne ustawienia formatowania z timm (żeby nie zepsuć wymiarów!)
            pe_kwargs = {}
            if hasattr(old_pe, 'flatten'):
                pe_kwargs['flatten'] = old_pe.flatten
            if hasattr(old_pe, 'output_fmt'):
                pe_kwargs['output_fmt'] = old_pe.output_fmt
                
            # Podmieniamy warstwę konwolucyjną na nową, zachowując stary format
            self.backbone.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=self.patch_size,
                in_chans=input_channels,
                embed_dim=self.native_embed_dim,
                **pe_kwargs
            )
            
            # Resetujemy tradycyjną mapę pozycyjną (TYLKO jeśli model jej używa)
            # (Modele oparte na EVA często używają RoPE i w ogóle nie mają pos_embed!)
            if hasattr(self.backbone, 'pos_embed') and self.backbone.pos_embed is not None:
                num_patches = self.backbone.patch_embed.num_patches
                num_prefix_tokens = self.backbone.num_prefix_tokens
                
                self.backbone.pos_embed = nn.Parameter(
                    torch.zeros(1, num_patches + num_prefix_tokens, self.native_embed_dim)
                )
                trunc_normal_(self.backbone.pos_embed, std=0.02)
        
        # KROK 3: Inteligentne Freezowanie
        if self.freeze_backbone:
            for name, param in self.backbone.named_parameters():
                # Nie mrozimy nowej warstwy wejściowej ani pozycji
                if "patch_embed" not in name and "pos_embed" not in name:
                    param.requires_grad = False

            self.backbone.eval()
                
        self.embed_dim = target_dim if target_dim is not None else self.native_embed_dim
        if target_dim is not None and target_dim != self.native_embed_dim:
            self.proj = nn.Conv2d(self.native_embed_dim, self.embed_dim, kernel_size=1)
        else:
            self.proj = nn.Identity()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            # Utrzymujemy wyłączone dropouty/batchnormy w starym mózgu
            self.backbone.eval()
            # Upewniamy się, że nowa warstwa pozostaje w trybie treningu (jeśli ma np. dropout)
            self.backbone.patch_embed.train(mode)

        return self

    def forward(self, x: torch.Tensor, return_features: bool = False) -> dict:
        outputs = {}
        B, _, H, W = x.shape
        
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Image shape ({H}x{W}) must be divisible by patch size "
                f"({self.patch_size}). Change 'input_image_size'!"
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
