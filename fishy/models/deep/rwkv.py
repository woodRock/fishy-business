# -*- coding: utf-8 -*-
"""
Receptance Weighted Key-Value (RWKV) model for spectral classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RWKV(nn.Module):
    """
    Receptance-Weighted Key-Value (RWKV) model.

    Attributes:
        input_dim (int): Number of input features.
        output_dim (int): Number of output classes.
        hidden_dim (int): Recurrent state dimension.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,  # Used to scale depth
        dropout: float = 0.2,
        **kwargs
    ) -> None:
        """
        Initializes the RWKV model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output classes.
            hidden_dim (int, optional): Hidden dimension. Defaults to 128.
            num_layers (int, optional): Number of blocks. Defaults to 4.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(RWKV, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Linear layers for key, value, and output
        self.key_layer = nn.Linear(input_dim, hidden_dim)
        self.value_layer = nn.Linear(input_dim, hidden_dim)
        self.recurrent_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute keys and values
        keys = self.key_layer(x)
        values = self.value_layer(x)

        # Update hidden state
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        # Simplified recurrent step
        h = h + torch.tanh(keys + self.recurrent_layer(values))

        # Compute output
        output = self.output_layer(h)
        output = self.dropout(output)
        return output
