# -*- coding: utf-8 -*-
"""
NextFormer model for spectral analysis.

The NextFormer Architecture features:
1. RMSNorm: Replaces standard LayerNorm. It operates strictly on the variance (omitting the mean centering)
   for faster computation while maintaining/exceeding training stability.
2. Pre-Norm Architecture: Normalization is applied before the attention and feed-forward blocks,
   creating an unimpeded residual pathway (crucial for deep networks).
3. Grouped Query Attention (GQA): Replaces Multi-Head Attention (MHA). By sharing Keys and Values
   across groups of Query heads, it reduces KV cache memory usage and accelerates generation.
4. SwiGLU Feed-Forward Networks: Introduces a gating mechanism (SiLU(xW1) * xW3)W2 that consistently
   outperforms traditional MLPs.
5. Bias-free Linear Layers: All internal linear layers omit biases for training stability.
"""

import math
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fishy.models.utils import ensure_conv_input


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """
        Initializes the RMSNorm layer.

        Args:
            dim (int): Input dimension.
            eps (float, optional): Epsilon for numerical stability. Defaults to 1e-6.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the RMS normalization."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU activation function / Feed-Forward Network.

    Attributes:
        w1 (nn.Linear): First linear layer for gating.
        w2 (nn.Linear): Output projection layer.
        w3 (nn.Linear): Gating value layer.
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        """
        Initializes the SwiGLU layer.

        Args:
            dim (int): Input/output dimension.
            hidden_dim (int): Intermediate dimension.
        """
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying the SwiGLU gating mechanism."""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats the Key and Value heads for Grouped Query Attention.

    Args:
        x (torch.Tensor): Input tensor of shape (batch, n_kv_heads, seq_len, head_dim).
        n_rep (int): Number of times to repeat each head.

    Returns:
        torch.Tensor: Repeated tensor of shape (batch, n_heads, seq_len, head_dim).
    """
    batch, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(batch, n_kv_heads, n_rep, slen, head_dim)
        .reshape(batch, n_kv_heads * n_rep, slen, head_dim)
    )


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) mechanism.

    Attributes:
        n_heads (int): Total number of query heads.
        n_kv_heads (int): Total number of key/value heads.
        n_rep (int): Number of times KV heads are repeated.
        head_dim (int): Dimension of each head.
        wq (nn.Linear): Query projection.
        wk (nn.Linear): Key projection.
        wv (nn.Linear): Value projection.
        wo (nn.Linear): Output projection.
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int) -> None:
        """
        Initializes GQA.

        Args:
            dim (int): Input dimension.
            n_heads (int): Number of query heads.
            n_kv_heads (int): Number of KV heads.
        """
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for GQA."""
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)

        if return_attention:
            return output, scores
        return output


class NextFormerBlock(nn.Module):
    """
    A single NextFormer block.

    Attributes:
        attention (GroupedQueryAttention): GQA layer.
        feed_forward (SwiGLU): SwiGLU FFN layer.
        attention_norm (RMSNorm): Norm before attention.
        ffn_norm (RMSNorm): Norm before FFN.
        dropout (nn.Dropout): Dropout for regularization.
    """

    def __init__(
        self, dim: int, n_heads: int, n_kv_heads: int, hidden_dim: int, dropout: float
    ) -> None:
        """Initializes the block."""
        super().__init__()
        self.attention = GroupedQueryAttention(dim, n_heads, n_kv_heads)
        self.feed_forward = SwiGLU(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with pre-norm architecture and residual connections."""
        h = x
        x = self.attention_norm(x)
        if return_attention:
            attn_out, attn_weights = self.attention(x, return_attention=True)
            h = h + self.dropout(attn_out)
            x = self.ffn_norm(h)
            out = h + self.dropout(self.feed_forward(x))
            return out, attn_weights
        else:
            h = h + self.dropout(self.attention(x))
            x = self.ffn_norm(h)
            out = h + self.dropout(self.feed_forward(x))
            return out


class NextFormer(nn.Module):
    """
    NextFormer Architecture for 1D spectral data.

    Attributes:
        blocks (nn.ModuleList): Stack of NextFormer blocks.
        norm (RMSNorm): Final output normalization.
        fc_out (nn.Linear): Output classification/regression head.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        **kwargs,
    ) -> None:
        """
        Initializes the NextFormer model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output classes/dimensions.
            hidden_dim (int, optional): Intermediate dimension of the feed-forward layer. Defaults to 128.
            num_layers (int, optional): Number of transformer blocks. Defaults to 4.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            num_heads (int, optional): Number of attention heads. Defaults to 8.
            num_kv_heads (int, optional): Number of KV heads for GQA. Defaults to 2.
        """
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                NextFormerBlock(input_dim, num_heads, num_kv_heads, hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        self.norm = RMSNorm(input_dim)
        self.fc_out = nn.Linear(input_dim, output_dim, bias=False)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False, *args, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward pass."""
        # Ensure input has 3 dimensions [batch_size, seq_length, features]
        x = ensure_conv_input(x)

        attentions = []
        for block in self.blocks:
            if return_attention:
                x, attn = block(x, return_attention=True)
                attentions.append(attn)
            else:
                x = block(x)

        # Apply final norm before output head
        x = self.norm(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Final linear layer
        x = self.fc_out(x)

        if return_attention:
            return x, attentions
        return x


__all__ = [
    "NextFormer",
    "NextFormerBlock",
    "GroupedQueryAttention",
    "RMSNorm",
    "SwiGLU",
]
