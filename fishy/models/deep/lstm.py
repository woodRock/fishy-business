# -*- coding: utf-8 -*-
"""
Long Short-Term Memory (LSTM) model for spectral classification.
"""

import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    LSTM-based model for spectral data classification.

    Attributes:
        input_dim (int): Number of input features.
        output_dim (int): Number of output classes.
        hidden_dim (int): Hidden layer dimension.
        num_layers (int): Number of LSTM layers.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
        **kwargs
    ) -> None:
        """
        Initializes the LSTM model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output classes.
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 128.
            num_layers (int, optional): Number of LSTM layers. Defaults to 4.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, output_dim).
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch_size, input_dim, 1)
        
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take the last time step
        out = self.dropout(out)
        return self.fc_out(out)
