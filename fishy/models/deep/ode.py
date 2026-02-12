# -*- coding: utf-8 -*-
"""
Neural Ordinary Differential Equation (ODE) model for spectral classification.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint
from typing import Dict, Any, Optional


class ODEFunc(nn.Module):
    """
    Implements the derivative function for the ODE system.
    """

    def __init__(self, channels: int, dropout: float = 0.5) -> None:
        super(ODEFunc, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the derivative dx/dt.
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out


class ODEBlock(nn.Module):
    """
    A single ODE block that solves an initial value problem.
    """

    def __init__(self, odefunc: ODEFunc) -> None:
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.register_buffer("integration_times", torch.linspace(0, 1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = odeint(self.odefunc, x, self.integration_times, rtol=1e-3, atol=1e-3)
        return out[1]


class ODE(nn.Module):
    """
    Neural ODE model for spectral classification.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,  # Used to scale depth if needed
        dropout: float = 0.2,
        **kwargs
    ) -> None:
        """
        Initializes the Neural ODE model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output classes.
            hidden_dim (int, optional): Hidden dimension. Defaults to 128.
            num_layers (int, optional): Number of blocks. Defaults to 4.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(ODE, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        ode_func = ODEFunc(32, dropout=dropout)
        self.ode_block = ODEBlock(ode_func)

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.initial_conv(x)
        x = self.ode_block(x)
        return self.fc_layers(x)
