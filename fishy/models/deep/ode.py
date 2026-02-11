"""Neural ODE model for spectral classification.

This model uses a neural ODE block to learn the dynamics of the input time series data.
It includes an initial convolutional layer, ODE blocks for learning the dynamics,
and a fully connected layer for classification.


References:

1. Chen, T., Rubanova, Y., Bettencourt, J., & Dumoulin, J. (2018).
   Neural ordinary differential equations.
   In Advances in neural information processing systems (pp. 6571-6583).
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint


class ODEFunc(nn.Module):
    """
    ODE function for the Neural ODE block.

    Defines the derivative dx/dt using a small convolutional network.
    """

    def __init__(self, channels: int, dropout: float = 0.5) -> None:
        """
        Initializes the ODE function.

        Args:
            channels (int): Number of input channels.
            dropout (float, optional): Dropout rate. Defaults to 0.5.
        """
        super(ODEFunc, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the derivative at time t.

        Args:
            t (torch.Tensor): Time tensor.
            x (torch.Tensor): Input state tensor of shape (batch_size, channels, seq_length).

        Returns:
            torch.Tensor: The derivative dx/dt.
        """
        dx = self.conv1(x)
        dx = self.bn1(dx)
        dx = self.relu(dx)
        dx = self.conv2(dx)
        dx = self.bn2(dx)
        dx = self.dropout(dx)
        return dx


class ODEBlock(nn.Module):
    """
    ODE block that wraps an ODE function and integrates it.
    """

    def __init__(self, odefunc: ODEFunc) -> None:
        """
        Initializes the ODE block.

        Args:
            odefunc (ODEFunc): The function defining the ODE system.
        """
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.register_buffer("integration_times", torch.linspace(0, 1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Integrates the ODE from t=0 to t=1.

        Args:
            x (torch.Tensor): Initial state tensor.

        Returns:
            torch.Tensor: Final state tensor after integration.
        """
        out = odeint(self.odefunc, x, self.integration_times, method="rk4")
        return out[-1]


class ODE(nn.Module):
    """
    Neural ODE model for spectral data classification.

    Attributes:
        initial_conv (nn.Sequential): Initial projection to channel space.
        ode_block1 (ODEBlock): First neural ODE integration block.
        downsample1 (nn.Sequential): Downsampling convolution.
        ode_block2 (ODEBlock): Second neural ODE integration block.
        adaptive_pool (nn.AdaptiveMaxPool1d): Pooling layer.
        fc_layers (nn.Sequential): Classification head.
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.3) -> None:
        """
        Initializes the Neural ODE model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output classes.
            dropout (float, optional): Dropout rate. Defaults to 0.3.
        """
        super(ODE, self).__init__()

        # Initial convolution to get to the desired channel size
        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.ReLU()
        )

        # ODE blocks
        self.ode_block1 = ODEBlock(ODEFunc(32, dropout=dropout))
        self.downsample1 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.ode_block2 = ODEBlock(ODEFunc(64, dropout=dropout))

        # Global pooling and final layers
        self.adaptive_pool = nn.AdaptiveMaxPool1d(4)
        self.flatten = nn.Flatten()
        self.flat_features = 64 * 4

        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_features, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input spectrum of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, output_dim).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        x = self.initial_conv(x)
        x = self.ode_block1(x)
        x = self.downsample1(x)
        x = self.ode_block2(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x
