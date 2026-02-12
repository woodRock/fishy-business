# -*- coding: utf-8 -*-
"""
Neural Ordinary Differential Equation (ODE) model for spectral classification.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint
from typing import Dict, Any, Optional


class ODEFunc(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.2) -> None:
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ODE(nn.Module):
    """
    Neural ODE model for spectral classification.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,  # Used to scale complexity
        dropout: float = 0.2,
        **kwargs
    ) -> None:
        super(ODE, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.initial_layer = nn.Linear(input_dim, hidden_dim)
        self.ode_func = ODEFunc(hidden_dim, dropout=dropout)
        self.register_buffer("integration_times", torch.linspace(0, 1, 2))
        
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_layer(x)
        out = odeint(self.ode_func, x, self.integration_times, rtol=1e-3, atol=1e-3)
        x = out[1] # Final state
        return self.fc_out(x)
