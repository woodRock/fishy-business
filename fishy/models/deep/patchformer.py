# -*- coding: utf-8 -*-
"""
PatchFormer: Advanced Patch-based Transformer for spectral data.
Uses a Convolutional Stem and Spectral Gating for robust local-to-global modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, List
from .ttt_mixin import TTTMixin


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, num_heads: int) -> None:
        super().__init__()
        assert input_dim % num_heads == 0
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.qkv = nn.Linear(input_dim, 3 * input_dim)
        self.fc_out = nn.Linear(input_dim, input_dim)
        self.scale = self.head_dim**-0.5

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.input_dim)
        out = self.fc_out(out)
        if return_attention:
            return out, attn
        return out


class PatchFormer(nn.Module, TTTMixin):
    """
    Advanced Transformer with Convolutional Stem and Spectral Gating.
    Designed for robust patch-based processing of REIMS data.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_heads: int = 8,
        patch_size: int = 64,
        **kwargs,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        
        self.stem = nn.Conv1d(
            in_channels=1, 
            out_channels=hidden_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.num_patches = input_dim // patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_dim))
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.attention_layers = nn.ModuleList(
            [MultiHeadAttention(hidden_dim, num_heads) for _ in range(num_layers)]
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.norm_stem = nn.LayerNorm(hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False, *args, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        if x.dim() == 2:
            x = x.unsqueeze(1) # [B, 1, F]
            
        B, C, F = x.shape
        pad_len = (self.patch_size - (F % self.patch_size)) % self.patch_size
        if pad_len > 0:
            x = torch.nn.functional.pad(x, (0, pad_len))
        
        x = self.stem(x) 
        x = x.transpose(1, 2) 
        
        gate_weights = self.gate(x)
        x = x * gate_weights
        x = self.norm_stem(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) 
        
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = torch.nn.functional.interpolate(
                self.pos_embed.transpose(1, 2), 
                size=x.size(1), 
                mode='linear'
            ).transpose(1, 2)
            x = x + pos_embed
        else:
            x = x + self.pos_embed
            
        x = self.dropout(x)

        attentions = []
        for attention in self.attention_layers:
            residual = x
            x = self.layer_norm1(x)
            if return_attention:
                attn_out, attn_weights = attention(x, return_attention=True)
                x = residual + self.dropout(attn_out)
                attentions.append(attn_weights)
            else:
                x = residual + self.dropout(attention(x))

        residual = x
        x = self.layer_norm2(x)
        x = residual + self.dropout(self.feed_forward(x))

        x = x[:, 0]
        x = self.fc_out(x)

        if return_attention:
            return x, attentions
        return x


__all__ = ["PatchFormer"]
