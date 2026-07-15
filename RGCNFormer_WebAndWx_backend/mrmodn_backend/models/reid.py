"""CPU CNN-Transformer ReID model and visualization-aware forward pass."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = ConvBNAct(
            in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.activation = nn.GELU()
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.pointwise(self.depthwise(inputs))
        if self.use_residual:
            output = output + inputs
        return self.activation(output)


class CNNBackbone(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.stem = ConvBNAct(3, 32, kernel_size=3, stride=2)
        self.stages = nn.Sequential(
            SeparableConvBlock(32, 64, stride=2),
            SeparableConvBlock(64, 64),
            SeparableConvBlock(64, 128, stride=2),
            SeparableConvBlock(128, 128),
            SeparableConvBlock(128, 192, stride=2),
            SeparableConvBlock(192, 192),
            SeparableConvBlock(192, output_dim, stride=2),
            SeparableConvBlock(output_dim, output_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.stages(self.stem(images))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("Transformer dimension must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.attention_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(dim, dim)
        self.projection_dropout = nn.Dropout(dropout)

    def _forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, token_count, channels = tokens.shape
        qkv = self.qkv(tokens).reshape(
            batch_size, token_count, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        attention = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attention = self.attention_dropout(torch.softmax(attention, dim=-1))
        output = torch.matmul(attention, value)
        output = output.transpose(1, 2).reshape(batch_size, token_count, channels)
        output = self.projection_dropout(self.projection(output))
        return output, attention

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self._forward(tokens)[0]

    def forward_with_attention(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._forward(tokens)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attention = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attention(self.norm1(tokens))
        return tokens + self.mlp(self.norm2(tokens))

    def forward_with_attention(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention_output, attention = self.attention.forward_with_attention(self.norm1(tokens))
        tokens = tokens + attention_output
        return tokens + self.mlp(self.norm2(tokens)), attention


class CNNTransformerReID(nn.Module):
    def __init__(
        self,
        num_classes: int,
        image_height: int = 288,
        image_width: int = 144,
        transformer_dim: int = 256,
        transformer_depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        embedding_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.image_height = image_height
        self.image_width = image_width
        self.transformer_dim = transformer_dim
        self.transformer_depth = transformer_depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout
        self.grid_height = (image_height + 31) // 32
        self.grid_width = (image_width + 31) // 32

        self.cnn = CNNBackbone(transformer_dim)
        token_count = self.grid_height * self.grid_width
        self.class_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, token_count + 1, transformer_dim))
        self.position_dropout = nn.Dropout(dropout)
        self.transformer = nn.Sequential(
            *[
                TransformerBlock(transformer_dim, num_heads, mlp_ratio, dropout)
                for _ in range(transformer_depth)
            ]
        )
        self.final_norm = nn.LayerNorm(transformer_dim)
        self.embedding_head = nn.Linear(transformer_dim, embedding_dim, bias=False)
        self.embedding_bn = nn.BatchNorm1d(embedding_dim)
        self.embedding_bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)

        self.apply(self._initialize_weights)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _tokens_from_feature_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        patch_tokens = feature_map.flatten(2).transpose(1, 2)
        class_token = self.class_token.expand(patch_tokens.shape[0], -1, -1)
        return torch.cat((class_token, patch_tokens), dim=1)

    def forward_features(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.cnn(images)
        tokens = self._tokens_from_feature_map(feature_map)
        tokens = self.position_dropout(tokens + self.position_embedding)
        tokens = self.final_norm(self.transformer(tokens))
        feature = self.embedding_bn(self.embedding_head(tokens[:, 0]))
        embedding = F.normalize(feature, p=2.0, dim=1, eps=1e-12)
        return embedding, feature

    def forward_visualization(self, images: torch.Tensor) -> dict[str, Any]:
        """Return normal outputs plus spatial tensors used by the Web visualization."""

        feature_map = self.cnn(images)
        tokens = self._tokens_from_feature_map(feature_map)
        position_tokens = tokens[:, 1:] + self.position_embedding[:, 1:]
        tokens = self.position_dropout(tokens + self.position_embedding)
        attentions: list[torch.Tensor] = []
        for block in self.transformer:
            tokens, attention = block.forward_with_attention(tokens)
            attentions.append(attention)
        tokens = self.final_norm(tokens)
        feature = self.embedding_bn(self.embedding_head(tokens[:, 0]))
        embedding = F.normalize(feature, p=2.0, dim=1, eps=1e-12)
        logits = self.classifier(feature)
        return {
            "feature_map": feature_map,
            "position_tokens": position_tokens,
            "attentions": attentions,
            "embedding": embedding,
            "feature": feature,
            "logits": logits,
        }

    def forward_embedding(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_features(images)[0]

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding, feature = self.forward_features(images)
        return embedding, self.classifier(feature)

    def get_config(self) -> dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "image_height": self.image_height,
            "image_width": self.image_width,
            "transformer_dim": self.transformer_dim,
            "transformer_depth": self.transformer_depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "embedding_dim": self.embedding_dim,
            "dropout": self.dropout_rate,
        }


def load_pretrained_reid(
    checkpoint_path: str | Path, device: str = "cpu"
) -> tuple[CNNTransformerReID, dict[str, Any]]:
    if device.lower() != "cpu":
        raise ValueError("ReID visualization supports CPU only")
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ReID checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model_config", "model", "label_to_pid", "data_splits"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"ReID checkpoint is missing keys: {', '.join(missing)}")
    model = CNNTransformerReID(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(torch.device("cpu"))
    model.eval()
    return model, checkpoint
