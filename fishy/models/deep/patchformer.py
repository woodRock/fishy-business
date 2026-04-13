# -*- coding: utf-8 -*-
"""
PatchFormer: A Vision-Transformer-style model for REIMS mass spectrometry.

The standard 1D Transformer treats the entire spectrum as a single token
(seq_len=1), making self-attention degenerate. PatchFormer instead splits
the spectrum into non-overlapping patches and embeds each patch as a token,
giving the transformer a real sequence to attend over.

With input_dim=2080 and patch_size=32:
    2080 / 32 = 65 tokens  →  65×65 attention matrix per head

This allows the model to learn long-range dependencies between spectral
regions (e.g. correlating peaks at different m/z ranges), which the
standard 1D Transformer cannot do.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union


class PatchEmbed(nn.Module):
    """Split spectrum into patches and linearly embed each one."""

    def __init__(self, input_dim: int, patch_size: int, embed_dim: int):
        super().__init__()
        assert (
            input_dim % patch_size == 0
        ), f"input_dim ({input_dim}) must be divisible by patch_size ({patch_size})"
        self.patch_size = patch_size
        self.num_patches = input_dim // patch_size
        self.proj = nn.Linear(patch_size, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F] → [B, num_patches, patch_size] → [B, num_patches, embed_dim]
        B, F = x.shape
        x = x.view(B, self.num_patches, self.patch_size)
        return self.proj(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with SwiGLU FFN."""

    def __init__(
        self, embed_dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.3
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = embed_dim * mlp_ratio
        # SwiGLU FFN
        self.w1 = nn.Linear(embed_dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, embed_dim, bias=False)
        self.w3 = nn.Linear(embed_dim, hidden, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        residual = x
        h = self.norm2(x)
        x = residual + self.drop(self.w2(F.silu(self.w1(h)) * self.w3(h)))
        return x


class PatchFormer(nn.Module):
    """
    Patch-based Transformer for 1D REIMS spectra.

    Splits the spectrum into fixed-size patches (default 32 bins each),
    embeds them linearly, adds learnable positional encodings, then applies
    standard transformer blocks. Attention now operates over 65 real tokens
    instead of a single degenerate one.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.3,
        patch_size: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()

        # Pad input_dim to the next multiple of patch_size if needed
        self.patch_size = patch_size
        self.pad = (patch_size - input_dim % patch_size) % patch_size
        padded_dim = input_dim + self.pad

        self.patch_embed = PatchEmbed(padded_dim, patch_size, hidden_dim)
        num_patches = padded_dim // patch_size

        # Learnable positional embeddings (one per patch)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, num_heads, mlp_ratio=4, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False, *args, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        if x.dim() == 3:
            x = x.squeeze(1)

        # Pad if input_dim isn't divisible by patch_size
        if self.pad > 0:
            x = F.pad(x, (0, self.pad))

        x = self.patch_embed(x)  # [B, num_patches, hidden_dim]
        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x.mean(dim=1)  # global average pool over patches
        logits = self.fc_out(x)

        if return_attention:
            return logits, []
        return logits


__all__ = ["PatchFormer"]
