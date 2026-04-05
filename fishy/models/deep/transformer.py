# -*- coding: utf-8 -*-
"""
Transformer model for time series classification.

This model uses multi-head attention and feed-forward layers to process sequential data.
It is designed to handle variable-length sequences and can be used for tasks such as classification or regression.
The architecture includes layer normalization, dropout for regularization, and a final fully connected layer for output.
"""

import torch
from fishy.models.utils import ensure_conv_input
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, List


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.

    Attributes:
        input_dim (int): Number of input features.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        qkv (nn.Linear): Combined projection for Q, K, and V.
        fc_out (nn.Linear): Final output projection.
        scale (float): Scaling factor for dot-product attention.
    """

    def __init__(self, input_dim: int, num_heads: int) -> None:
        """
        Initializes the MultiHeadAttention layer.

        Args:
            input_dim (int): Number of input features.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        assert input_dim % num_heads == 0

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # Combined projection for Q, K, V
        self.qkv = nn.Linear(input_dim, 3 * input_dim)
        self.fc_out = nn.Linear(input_dim, input_dim)

        self.scale = self.head_dim**-0.5

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).
            return_attention (bool): Whether to return attention weights.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, input_dim).
            torch.Tensor (optional): Attention weights of shape (batch_size, num_heads, seq_length, seq_length).
        """
        batch_size, seq_len, _ = x.shape

        # Single matrix multiplication for all projections
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Combine heads
        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.input_dim)
        out = self.fc_out(out)

        if return_attention:
            return out, attn
        return out


class Transformer(nn.Module):
    """
    Transformer architecture for 1D spectral data.

    Attributes:
        attention_layers (nn.ModuleList): List of multi-head attention layers.
        feed_forward (nn.Sequential): Position-wise feed-forward network.
        layer_norm1 (nn.LayerNorm): Norm layer before attention.
        layer_norm2 (nn.LayerNorm): Norm layer before feed-forward.
        dropout (nn.Dropout): Dropout layer.
        fc_out (nn.Linear): Final classification/regression head.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        num_heads: int = 4,
        **kwargs,
    ) -> None:
        """
        Initializes the Transformer model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output classes/dimensions.
            hidden_dim (int, optional): Intermediate dimension of the feed-forward layer. Defaults to 128.
            num_layers (int, optional): Number of transformer blocks. Defaults to 1.
            dropout (float, optional): Dropout rate for regularization. Defaults to 0.1.
            num_heads (int, optional): Number of attention heads. Defaults to 4.
        """
        super().__init__()

        self.attention_layers = nn.ModuleList(
            [MultiHeadAttention(input_dim, num_heads) for _ in range(num_layers)]
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False, *args, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input spectrum of shape (batch_size, input_dim) or (batch_size, seq_len, input_dim).
            return_attention (bool): Whether to return attention weights from all layers.

        Returns:
            torch.Tensor: Logits/predictions of shape (batch_size, output_dim).
            List[torch.Tensor] (optional): List of attention weights from each layer.
        """
        # Ensure input has 3 dimensions [batch_size, seq_length, features]
        # For 1D spectral data, we treat the peaks as features in a sequence of length 1
        x = ensure_conv_input(x)

        attentions = []
        # Apply attention layers with residual connections
        for attention in self.attention_layers:
            residual = x
            x = self.layer_norm1(x)
            if return_attention:
                attn_out, attn_weights = attention(x, return_attention=True)
                x = residual + self.dropout(attn_out)
                attentions.append(attn_weights)
            else:
                x = residual + self.dropout(attention(x))

        # Feed-forward with residual connection
        residual = x
        x = self.layer_norm2(x)
        x = residual + self.dropout(self.feed_forward(x))

        # Global pooling and classification
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc_out(x)

        if return_attention:
            return x, attentions
        return x


__all__ = [
    "Transformer",
    "MultiHeadAttention",
]
