# -*- coding: utf-8 -*-
"""
Kolmogorov-Arnold Network (KAN) model for spectral classification.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KAN(nn.Module):
    """
    KAN-based model for spectral data classification.
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
        super(KAN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        layers = []
        in_f = input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_f, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_f = hidden_dim
        
        layers.append(nn.Linear(in_f, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
