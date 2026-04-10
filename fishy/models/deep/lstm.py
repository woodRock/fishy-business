# -*- coding: utf-8 -*-
"""
Bidirectional Gated Recurrent Unit (GRU) model for spectral classification.
"""

import torch
from fishy.models.utils import ensure_seq_input
import torch.nn as nn


class LSTM(nn.Module):
    """
    Bidirectional GRU-based model for spectral data classification.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        """
        Initializes the GRU model.
        """
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Using GRU as it's often more efficient than LSTM for 1D data
        # Bi-directional allows capturing context from both sides of the spectrum
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,
        )
        # Bi-directional means output size is 2 * hidden_dim
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GRU model.
        """
        x = ensure_seq_input(x)

        out, _ = self.gru(x)

        # Max pool over the sequence dimension to capture most prominent features
        # and ignore zeros/noise
        out, _ = torch.max(out, dim=1)

        out = self.dropout(out)
        return self.fc_out(out)
