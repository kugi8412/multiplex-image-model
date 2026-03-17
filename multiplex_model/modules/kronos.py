import torch

from .base_modules import Encoder
from .registry import ENCODER_REGISTRY, resolve_encoder_class
from ..kronos.vision_transformer import vit_small, vit_large


@ENCODER_REGISTRY.register("kronos_hybrid")
class KronosHybridEncoder(Encoder):
    """
    Koder hybrydowy (Kameleon): 
    - Faza Marker-Agnostic (stem=True): Używa ConvNeXta.
    - Faza Pan-Marker (stem=False): Używa bloków KRONOSa (ViT + Tokeny Rejestru).
    """
    def __init__(
        self, 
        input_channels: int, 
        layers_blocks: list[int] = None, 
        embedding_dims: list[int] = None, 
        stem: bool = False,
        model_type: str = "vits16",
        drop_path_rate: float = 0.1,
        **kwargs # KRYTYCZNE: Pochłania nadmiarowe argumenty z YAML (np. block_parameters)
    ):
        super().__init__()
        self.stem = stem
        
        if self.stem:
            # 1. TWORZENIE FAZY MARKER-AGNOSTIC (Lokalne filtry)
            # Dynamicznie ładujemy Twojego ConvNeXta z rejestru
            convnext_cls = resolve_encoder_class("convnext")
            self.ma_encoder = convnext_cls(
                input_channels=input_channels,
                layers_blocks=layers_blocks,
                embedding_dims=embedding_dims,
                stem=True,
                **kwargs
            )
        else:
            # 2. TWORZENIE FAZY PAN-MARKER (KRONOS)
            if model_type == "vits16":
                # Wymiar 384
                self.backbone = vit_small(patch_size=16, num_markers=1, drop_path_rate=drop_path_rate)
                self.embed_dim = 384
            else:
                # Wymiar 1024
                self.backbone = vit_large(patch_size=16, num_markers=1, drop_path_rate=drop_path_rate)
                self.embed_dim = 1024

            assert input_channels == self.embed_dim, f"Hyperkernel musi wypuszczać wymiar {self.embed_dim} dla tego modelu KRONOS!"

    def forward(self, x: torch.Tensor, return_features: bool = False) -> dict:
        # Jeśli jesteśmy w fazie wczesnej kompresji, oddajemy tensor do ConvNeXta
        if self.stem:
            return self.ma_encoder(x, return_features=return_features)
            
        # ====================================================
        # PRZEPUST KRONOSA (Właściwa analiza przestrzenna)
        # ====================================================
        B, E, H, W = x.shape
        
        # Spłaszczanie dla Transformera
        x_flat = x.flatten(2).transpose(1, 2)  # Kształt: [B, H*W, E]
        num_patches = H * W
        
        # Dodanie Tokena CLS, Positional Embeddings i Tokenów Rejestru
        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)
        x_flat = torch.cat((cls_tokens, x_flat), dim=1)
        pos_embed = self.backbone.interpolate_pos_encoding(x_flat, W, H, num_patches)
        x_flat = x_flat + pos_embed
        
        if self.backbone.register_tokens is not None:
            reg_tokens = self.backbone.register_tokens.expand(B, -1, -1)
            x_flat = torch.cat((x_flat[:, :1], reg_tokens, x_flat[:, 1:]), dim=1)
            
        # Self-Attention KRONOSa
        features = []
        for blk in self.backbone.blocks:
            x_flat = blk(x_flat)
            if return_features:
                features.append(x_flat)
                
        x_flat = self.backbone.norm(x_flat)
        num_reg = self.backbone.num_register_tokens
        
        # Wycinamy patche (odrzucamy CLS i Register Tokens)
        patch_tokens = x_flat[:, 1 + num_reg:] 
        out_spatial = patch_tokens.transpose(1, 2).reshape(B, E, H, W)
        outputs = {"output": out_spatial}

        if return_features:
            outputs["features"] = features
            
        return outputs
