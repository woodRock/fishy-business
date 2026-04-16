# -*- coding: utf-8 -*-
"""
AugFormerV2: Enhanced version of AugFormer with advancements from Parameter Golf.

Features implemented:
- QK-Gain: Learnable per-head query scaling.
- Parallel Residuals: Attention and MLP branches in parallel (GPT-J style).
- Depth Recurrence: Selective looping of transformer blocks.
- LeakyReLU(0.5)^2 Activation: Non-monotonic activation for efficiency.
- TTT Support: forward_ttt method for entropy minimization at test-time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union, Optional
from .augformer import AugFormer, RMSNorm, TransformerBlock, MultiHeadAttention


class LeakyReluSq(nn.Module):
    """Non-monotonic activation used in high-performing Golf models."""

    def __init__(self, neg_slope: float = 0.5):
        super().__init__()
        self.neg_slope = neg_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, negative_slope=self.neg_slope).square()


class MultiHeadAttentionV2(MultiHeadAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        use_xsa: bool = False,
        use_qk_gain: bool = False,
    ):
        super().__init__(embed_dim, num_heads, use_xsa)
        self.use_qk_gain = use_qk_gain
        if self.use_qk_gain:
            # Learnable per-head gain for queries
            self.q_gain = nn.Parameter(torch.ones(num_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # qkv: [B, N, 3*C] -> [B, N, 3, H, D]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        # q, k, v: [B, H, N, D]
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        # Apply RMSNorm to Q and K as in modded-nanogpt
        # Q and K are [B, H, N, D], we norm over the last dimension D
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        if self.use_qk_gain:
            # self.q_gain is [H], we need to broadcast it to [B, H, N, D]
            # Broadcasting: [H] -> [1, H, 1, 1] works for [B, H, N, D]
            q = q * self.q_gain[None, :, None, None]

        # attn: [B, H, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # y: [B, H, N, D]
        y = attn @ v

        if self.use_xsa:
            Vn = torch.nn.functional.normalize(v, dim=-1)
            y = y - (y * Vn).sum(dim=-1, keepdim=True) * Vn

        # Reshape y back to [B, N, C]
        x = y.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TransformerBlockV2(TransformerBlock):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
        use_xsa: bool = False,
        use_qk_gain: bool = False,
        use_parallel_residuals: bool = False,
        use_leaky_sq: bool = False,
    ):
        super().__init__(embed_dim, num_heads, mlp_ratio, dropout, use_xsa)
        self.use_parallel_residuals = use_parallel_residuals
        self.use_leaky_sq = use_leaky_sq

        # Replace attention with V2
        self.attn = MultiHeadAttentionV2(embed_dim, num_heads, use_xsa, use_qk_gain)

        if self.use_leaky_sq:
            # Use LeakyReLU(0.5)^2 instead of Silu for the MLP
            # In AugFormer's SwiGLU: w2(F.silu(w1(h)) * w3(h))
            # For simplicity in V2 with leaky_sq, we use a simpler bottleneck: w2(sq(w1(h)))
            # This reduces parameters while keeping nonlinearity complex.
            self.act = LeakyReluSq(neg_slope=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_parallel_residuals:
            # Parallel Residuals (GPT-J style):
            # Attention and MLP branches read from the same norm output.
            x_norm = self.norm1(x)
            attn_out = self.attn(x_norm)

            if self.use_leaky_sq:
                # Optimized bottleneck path
                mlp_out = self.w2(self.act(self.w1(x_norm)))
            else:
                # Standard SwiGLU path
                mlp_out = self.w2(F.silu(self.w1(x_norm)) * self.w3(x_norm))

            return x + self.drop(attn_out) + self.drop(mlp_out)
        else:
            # Sequential Residuals (Original style)
            x = x + self.drop(self.attn(self.norm1(x)))
            h = self.norm2(x)
            if self.use_leaky_sq:
                mlp_out = self.w2(self.act(self.w1(h)))
            else:
                mlp_out = self.w2(F.silu(self.w1(h)) * self.w3(h))
            return x + self.drop(mlp_out)


class AugFormerV2(AugFormer):
    """
    Enhanced Augmentation-as-Sequence Transformer.
    Allows toggling Parameter Golf advancements for benchmarking.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.3,
        num_views: int = 6,
        use_xsa: bool = False,
        # V2 specific flags
        use_qk_gain: bool = False,
        use_parallel_residuals: bool = False,
        recurrence_layers: Optional[List[int]] = None,
        use_leaky_sq: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            num_views=num_views,
            use_xsa=use_xsa,
            **kwargs,
        )
        self.use_qk_gain = use_qk_gain
        self.use_parallel_residuals = use_parallel_residuals
        self.recurrence_layers = recurrence_layers or []
        self.use_leaky_sq = use_leaky_sq

        # Rebuild blocks with V2 implementation
        self.blocks = nn.ModuleList(
            [
                TransformerBlockV2(
                    hidden_dim,
                    num_heads,
                    mlp_ratio=2,
                    dropout=dropout,
                    use_xsa=use_xsa,
                    use_qk_gain=use_qk_gain,
                    use_parallel_residuals=use_parallel_residuals,
                    use_leaky_sq=use_leaky_sq,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False, *args, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        if x.dim() == 3 and x.shape[1] != (self.num_views + 1):
            # Probably [B, 1, F] from some wrapper, squeeze it
            x = x.squeeze(1)

        if x.dim() == 2:
            # Standard input: [B, F] -> Build views
            views = self._build_views(x)  # [B, V+1, F]
        else:
            # Input already has views (e.g. from TTT loop): [B, V+1, F]
            views = x

        B = views.shape[0]
        V_total = views.shape[1]

        # Flatten B and V for shared embedding and gating
        tokens = views.reshape(B * V_total, -1)
        tokens = self.view_embed(tokens)
        tokens = self.pre_gate(tokens)

        # Reshape back to [B, V_total, hidden_dim]
        tokens = tokens.reshape(B, V_total, -1)

        tokens[:, 0:1] = tokens[:, 0:1] + self.original_embed
        tokens[:, 1:] = tokens[:, 1:] + self.aug_embed
        anchor_residual = tokens[:, 0:1].clone()

        for i, block in enumerate(self.blocks):
            tokens = block(tokens)
            # Depth Recurrence: loop the block if its index is in recurrence_layers
            if i in self.recurrence_layers:
                tokens = block(tokens)

        tokens[:, 0:1] = tokens[:, 0:1] + anchor_residual
        tokens = self.norm(tokens)
        logits = self.fc_out(tokens[:, 0])

        if return_attention:
            return logits, []
        return logits


__all__ = ["AugFormerV2"]
