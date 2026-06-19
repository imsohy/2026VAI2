"""Model definitions for the ViT tuning assignment.

UVA Tutorial 15 adaptation notes
--------------------------------
The VisionTransformer, img_to_patch, and Pre-LN AttentionBlock are adapted from
UVA DL Tutorial 15. Changes made for this assignment:
- kept the UVA-style PyTorch Lightning training path in `src/lit_module.py`
- used batch_first=True in nn.MultiheadAttention for clearer [B, T, D] tensors
- added pooling_mode='cls' or 'mean' for the student twist
- added optional positional embedding removal
- added a small CNN baseline for fair comparison on CIFAR-10
"""

from __future__ import annotations

from typing import Dict, Any

import torch
from torch import nn


def img_to_patch(x: torch.Tensor, patch_size: int, flatten_channels: bool = True) -> torch.Tensor:
    """Split image tensor into non-overlapping patches.

    This follows the UVA Tutorial 15 idea:
    [B, C, H, W] -> [B, num_patches, C * patch_size * patch_size]
    when flatten_channels=True.
    """
    b, c, h, w = x.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(f"Image size {(h, w)} must be divisible by patch_size={patch_size}.")

    x = x.reshape(b, c, h // patch_size, patch_size, w // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)               # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)           # [B, H'*W', C*p_H*p_W]
    return x


class AttentionBlock(nn.Module):
    """Pre-LN Transformer encoder block.

    Professor's lecture emphasized that ViT commonly uses Pre-LN for stable
    training and better gradient flow. This block keeps the input/output tensor
    shape unchanged: [B, T, D] -> [B, T, D].
    """

    def __init__(self, embed_dim: int, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.layer_norm_1(x)
        x = x + self.attn(attn_input, attn_input, attn_input, need_weights=False)[0]
        x = x + self.mlp(self.layer_norm_2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer for CIFAR-10 classification.

    Main components:
    - patch embedding via a linear projection
    - optional CLS token
    - learnable positional embedding
    - Pre-LN Transformer blocks
    - MLP classification head
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        num_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
        pooling_mode: str = "cls",
        use_positional_embedding: bool = True,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if pooling_mode not in {"cls", "mean"}:
            raise ValueError("pooling_mode must be either 'cls' or 'mean'")

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.pooling_mode = pooling_mode
        self.use_positional_embedding = use_positional_embedding

        patch_dim = num_channels * patch_size * patch_size
        self.input_layer = nn.Linear(patch_dim, embed_dim)
        self.transformer = nn.Sequential(
            *[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))

        if pooling_mode == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            pos_tokens = self.num_patches + 1
        else:
            self.cls_token = None
            pos_tokens = self.num_patches

        if use_positional_embedding:
            self.pos_embedding = nn.Parameter(torch.randn(1, pos_tokens, embed_dim))
        else:
            self.register_parameter("pos_embedding", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = img_to_patch(x, self.patch_size, flatten_channels=True)
        x = self.input_layer(patches)

        if self.pooling_mode == "cls":
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_token, x], dim=1)

        if self.pos_embedding is not None:
            x = x + self.pos_embedding[:, : x.shape[1], :]

        x = self.dropout(x)
        x = self.transformer(x)

        if self.pooling_mode == "cls":
            pooled = x[:, 0]
        else:
            pooled = x.mean(dim=1)
        return self.mlp_head(pooled)


class SimpleCNN(nn.Module):
    """Small CNN baseline for CIFAR-10.

    This is intentionally simple, so the report can compare ViT against a
    convolutional model with an explicit locality bias.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(config: Dict[str, Any]) -> nn.Module:
    model_cfg = config["model"]
    model_type = model_cfg["type"].lower()

    if model_type == "vit":
        return VisionTransformer(
            image_size=config["data"].get("image_size", 32),
            patch_size=model_cfg.get("patch_size", 4),
            num_channels=model_cfg.get("num_channels", 3),
            num_classes=config["data"].get("num_classes", 10),
            embed_dim=model_cfg.get("embed_dim", 128),
            hidden_dim=model_cfg.get("hidden_dim", 256),
            num_heads=model_cfg.get("num_heads", 4),
            num_layers=model_cfg.get("num_layers", 4),
            dropout=model_cfg.get("dropout", 0.1),
            pooling_mode=model_cfg.get("pooling_mode", "cls"),
            use_positional_embedding=model_cfg.get("use_positional_embedding", True),
        )
    if model_type == "cnn":
        return SimpleCNN(num_classes=config["data"].get("num_classes", 10), dropout=model_cfg.get("dropout", 0.1))
    raise ValueError(f"Unknown model type: {model_type}")
