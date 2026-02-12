# -*- coding: utf-8 -*-
"""
Transformer model for spectral classification.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Implements positional encoding for the Transformer model.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies positional encoding to the input tensor.
        """
        x = x + self.pe[: x.size(0), :]
        return x


class Transformer(nn.Module):
    """
    Transformer-based model for spectral data classification.

    Attributes:
        input_dim (int): Number of input features.
        output_dim (int): Number of output classes.
        hidden_dim (int): Dimension of the model's hidden layers.
        num_layers (int): Number of transformer layers.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
        num_heads: int = 4,
        **kwargs
    ) -> None:
        """
        Initializes the Transformer model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output classes.
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 128.
            num_layers (int, optional): Number of transformer layers. Defaults to 4.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
            num_heads (int, optional): Number of attention heads. Defaults to 4.
        """
        super(Transformer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Linear(1, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=input_dim)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, output_dim).
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch_size, input_dim, 1)
        
        x = x.transpose(0, 1)  # (input_dim, batch_size, 1)
        x = self.embedding(x)  # (input_dim, batch_size, hidden_dim)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Global average pooling
        x = self.dropout(x)
        return self.fc_out(x)
