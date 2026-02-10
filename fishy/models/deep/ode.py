"""Neural ODE model for time series classification.

This model uses a neural ODE block to learn the dynamics of the input time series data.
It includes an initial convolutional layer, ODE blocks for learning the dynamics,
and a fully connected layer for classification. The architecture is designed to handle
sequential data, such as time series or other ordered data.

References:
1. Chen, T., Rubanova, Y., Bettencourt, J., & Dumoulin, J. (2018).
   Neural ordinary differential equations.
   In Advances in neural information processing systems (pp. 6571-6583).
2. Srivastava, N., Hinton, G., Krizhevsky, A.,
   Sutskever, I., & Salakhutdinov, R. (2014).
   Dropout: a simple way to prevent neural networks from overfitting.
   The journal of machine learning research, 15(1), 1929-1958.
3. Hinton, G. E., Srivastava, N., Krizhevsky, A., Sutskever,
   I., & Salakhutdinov, R. R. (2012).
   Improving neural networks by preventing co-adaptation of feature detectors.
   arXiv preprint arXiv:1207.0580.
4. Loshchilov, I., & Hutter, F. (2017).
   Decoupled weight decay regularization.
   arXiv preprint arXiv:1711.05101.
5. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016).
   Rethinking the inception architecture for computer vision.
   In Proceedings of the IEEE conference on computer vision
   and pattern recognition (pp. 2818-2826).
6. Hendrycks, D., & Gimpel, K. (2016).
   Gaussian error linear units (gelus).
   arXiv preprint arXiv:1606.08415.
7. Loshchilov, I., & Hutter, F. (2017).
   Decoupled weight decay regularization.
   arXiv preprint arXiv:1711.05101.
8. Loshchilov, I., & Hutter, F. (2017).
   Decoupled weight decay regularization.
   arXiv preprint arXiv:1711.05101.
9. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016).
   Rethinking the inception architecture for computer vision.
   In Proceedings of the IEEE conference on computer vision
   and pattern recognition (pp. 2818-2826).
10. Hendrycks, D., & Gimpel, K. (2016).
    Gaussian error linear units (gelus).
    arXiv preprint arXiv:1606.08415.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import numpy as np
from tqdm import tqdm
from torchdiffeq import odeint


class ODEFunc(nn.Module):
    """ODE function for the Neural ODE block"""

    def __init__(self, channels, dropout=0.5) -> None:
        """Initialize the ODE function.

        Args:
            channels (int): Number of input channels.
            dropout (float): Dropout rate for regularization. Defaults to 0.5.
        """
        super(ODEFunc, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, t, x):
        """
        Forward pass through the ODE function.

        Args:
            t (torch.Tensor): Time tensor (not used in this case).
            x (torch.Tensor): Input tensor of shape (batch_size, channels, seq_length).

        Returns:
            torch.Tensor: Output tensor of the same shape as input x.
        """
        dx = self.conv1(x)
        dx = self.bn1(dx)
        dx = self.relu(dx)
        dx = self.conv2(dx)
        dx = self.bn2(dx)
        dx = self.dropout(dx)
        return dx


class ODEBlock(nn.Module):
    """ODE block for the Neural ODE model."""

    def __init__(self, odefunc) -> None:
        """Initialize the ODE block.

        Args:
            odefunc (ODEFunc): The ODE function to be used in the block.
        """
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.register_buffer("integration_times", torch.linspace(0, 1, 2))

    def forward(self, x):
        """Forward pass through the ODE block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, seq_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, seq_length).
            The output is the final state after integrating the ODE function over the specified time intervals.
        """
        # integration_times will automatically be on the same device as x
        out = odeint(self.odefunc, x, self.integration_times, method="rk4")
        return out[-1]  # Return only the final state


class ODE(nn.Module):
    """Neural ODE model for time series classification."""

    def __init__(self, input_dim, output_dim, dropout=0.3) -> None:
        """Initialize the Neural ODE model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output classes.
            dropout (float): Dropout rate for regularization. Defaults to 0.3.
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

    def forward(self, x):
        """Forward pass through the Neural ODE model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim),
            where output_dim is the number of classes.
        """
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.initial_conv(x)
        x = self.ode_block1(x)
        x = self.downsample1(x)
        x = self.ode_block2(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x
