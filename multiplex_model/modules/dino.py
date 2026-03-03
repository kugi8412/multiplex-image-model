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
        version: str = "v3",        # Wybór: "v2" lub "v3"
        model_size: str = "small",  # Wybór: "small", "base", "large"
        freeze_backbone: bool = False,
        in_channels: int = 1,
        target_dim: int | None = None,
    ):
        super().__init__()
        
        if timm is None:
            raise ImportError("Biblioteka 'timm' jest wymagana do użycia DinoEncoder. Zainstaluj ją: pip install timm")

        # Rejestr nazw modeli w timm dla obu generacji
        if version == "v2":
            size_map = {
                "small": "vit_small_patch14_dinov2.lvd142m",
                "base": "vit_base_patch14_dinov2.lvd142m",
                "large": "vit_large_patch14_dinov2.lvd142m"
            }
        elif version == "v3":
            # Konwencja timm dla DINOv3 (patch size 16 to standard dla v3)
            size_map = {
                "small": "vit_small_patch16_dinov3.lvd142m",
                "base": "vit_base_patch16_dinov3.lvd142m",
                "large": "vit_large_patch16_dinov3.lvd142m"
            }
        else:
            raise ValueError(f"Nieobsługiwana wersja DINO: {version}. Wybierz 'v2' lub 'v3'.")
            
        model_name = size_map.get(model_size)
        if model_name is None:
            raise ValueError(f"Nieobsługiwany rozmiar modelu: {model_size}. Wybierz 'small', 'base' lub 'large'.")

        print(f"Ładowanie modelu DINO{version} ({model_size}) przez timm: {model_name}...")
        
        # Pobieramy model bez klasyfikatora (num_classes=0)
        self.backbone = timm.create_model(
            model_name, 
            pretrained=True, 
            num_classes=0, 
            in_chans=in_channels,
            dynamic_img_size=True
        )
        
        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.native_embed_dim = self.backbone.embed_dim
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Warstwa projekcyjna rzutująca wymiary (jeśli target_dim jest wymuszony przez config dekodera)
        self.embed_dim = target_dim if target_dim is not None else self.native_embed_dim
        if target_dim is not None and target_dim != self.native_embed_dim:
            print(f"Dodawanie warstwy projekcyjnej dla DINO: {self.native_embed_dim} -> {self.embed_dim}")
            self.proj = nn.Conv2d(self.native_embed_dim, self.embed_dim, kernel_size=1)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor, return_features: bool = False) -> dict:
        outputs = {}
        B, C, H, W = x.shape
        
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Wymiary obrazu ({H}x{W}) muszą być podzielne przez wielkość patcha "
                f"modelu DINO{self.patch_size} ({self.patch_size}). Zmień 'input_image_size' w configu."
            )
            
        h_feat, w_feat = H // self.patch_size, W // self.patch_size
        
        features = self.backbone.forward_features(x)
        
        # Ekstrakcja tylko tokenów przestrzennych i reshape do 2D
        if features.dim() == 3:
            num_spatial_tokens = h_feat * w_feat
            patch_tokens = features[:, -num_spatial_tokens:, :] 
            spatial_features = patch_tokens.transpose(1, 2).reshape(B, self.native_embed_dim, h_feat, w_feat)
        elif features.dim() == 4:
            spatial_features = features
        else:
            raise RuntimeError(f"Niespodziewany kształt wyjścia z modelu timm: {features.shape}")

        spatial_features = self.proj(spatial_features)

        outputs["output"] = spatial_features
        if return_features:
            outputs["features"] = [spatial_features]
            
        return outputs
