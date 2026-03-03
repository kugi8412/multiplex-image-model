import torch
import torch.nn as nn

from .base_modules import Block, Encoder, LayerNorm
from .registry import BLOCK_REGISTRY, ENCODER_REGISTRY

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

@BLOCK_REGISTRY.register("csmamba")
class CrossScanMambaBlock(Block):
    """4-Way Cross-Scan Mamba Block (Horizontal & Vertical) for perfect 2D spatial awareness."""

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 1,
    ):
        """Initialize Cross-Scan Mamba block.
        
        Args:
            dim (int): Embedding dimension.
            d_state (int): SSM state dimension.
            d_conv (int): Local convolution width.
            expand (int): Block expansion factor.
        """
        super().__init__()
        if Mamba is None:
            raise ImportError("Zainstaluj mamba-ssm i causal-conv1d: pip install mamba-ssm causal-conv1d")

        self.norm = LayerNorm(dim, data_format="channels_first")
        
        # 4 niezależne bloki Mamba dla 4 kierunków (Cross-Scan)
        # H = Horizontal (Wiersze), V = Vertical (Kolumny)
        self.mamba_h_fwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_h_bwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_v_fwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_v_bwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        
        # Liniowa projekcja do połączenia cech ze wszystkich 4 kierunków
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        
        # --- 1. SKANOWANIE POZIOME (Wierszami) ---
        # Spłaszczanie: [B, C, H, W] -> [B, H*W, C]
        x_h = x_norm.flatten(2).transpose(1, 2)
        
        # W przód (Od lewej do prawej)
        out_h_fwd = self.mamba_h_fwd(x_h)
        
        # W tył (Od prawej do lewej)
        out_h_bwd = torch.flip(self.mamba_h_bwd(torch.flip(x_h, dims=[1])), dims=[1])

        # Powrót do siatki 2D
        out_h_fwd = out_h_fwd.transpose(1, 2).reshape(B, C, H, W)
        out_h_bwd = out_h_bwd.transpose(1, 2).reshape(B, C, H, W)

        # --- 2. SKANOWANIE PIONOWE (Kolumnami) ---
        # Transpozycja przestrzenna i spłaszczanie: [B, C, H, W] -> [B, C, W, H] -> [B, W*H, C]
        x_v = x_norm.transpose(2, 3).flatten(2).transpose(1, 2)
        
        # W dół (Z góry na dół po kolumnach)
        out_v_fwd = self.mamba_v_fwd(x_v)
        
        # W górę (Z dołu do góry po kolumnach)
        out_v_bwd = torch.flip(self.mamba_v_bwd(torch.flip(x_v, dims=[1])), dims=[1])

        # Powrót do siatki 2D (wymaga ponownej transpozycji osi H i W)
        out_v_fwd = out_v_fwd.transpose(1, 2).reshape(B, C, W, H).transpose(2, 3)
        out_v_bwd = out_v_bwd.transpose(1, 2).reshape(B, C, W, H).transpose(2, 3)

        # --- 3. FUZJA ---
        # Sumujemy wiedzę z 4 kierunków
        out_fuzja = out_h_fwd + out_h_bwd + out_v_fwd + out_v_bwd
        
        # Opcjonalnie: warstwa projekcyjna integrująca sumę i Residual Connection
        out = self.proj(out_fuzja)
        return x + out


@ENCODER_REGISTRY.register("vim")
class VisionMambaEncoder(Encoder):
    """Hierarchical Vision Mamba Encoder backbone with 4-way Cross-Scan."""

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
        block_parameters = block_parameters or {}
        self.patch_embeds = nn.ModuleList()

        # Stem (Patch Embedding)
        if stem:
            self.patch_embeds.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, embedding_dims[0], kernel_size=patch_size, stride=patch_size),
                    LayerNorm(embedding_dims[0], data_format="channels_first"),
                )
            )
        else:
            self.patch_embeds.append(nn.Identity())

        # Patch Merging (Downsampling)
        for i, out_dim in enumerate(embedding_dims[1:]):
            input_dim = embedding_dims[i]
            self.patch_embeds.append(
                nn.Sequential(
                    LayerNorm(input_dim, data_format="channels_first"),
                    nn.Conv2d(input_dim, out_dim, kernel_size=2, stride=2),
                )
            )

        # Tworzenie bloków Cross-Scan Mamba
        self.blocks = nn.ModuleList()
        for blocks, dim in zip(layers_blocks, embedding_dims):
            self.blocks.append(
                nn.Sequential(*[CrossScanMambaBlock(dim=dim, **block_parameters) for _ in range(blocks)])
            )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> dict:
        outputs = {}
        features = []

        for patch_embed, blocks in zip(self.patch_embeds, self.blocks):
            x = patch_embed(x)
            x = blocks(x)
            if return_features:
                features.append(x)

        outputs["output"] = x
        if return_features:
            outputs["features"] = features
            
        return outputs
