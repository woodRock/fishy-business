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

    def __init__(self, channels, dropout=0.5):
        super(ODEFunc, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, t, x):
        """
        t: scalar time parameter
        x: feature tensor
        """
        dx = self.conv1(x)
        dx = self.bn1(dx)
        dx = self.relu(dx)
        dx = self.conv2(dx)
        dx = self.bn2(dx)
        dx = self.dropout(dx)
        return dx


class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.register_buffer("integration_times", torch.linspace(0, 1, 2))

    def forward(self, x):
        # integration_times will automatically be on the same device as x
        out = odeint(self.odefunc, x, self.integration_times, method="rk4")
        return out[-1]  # Return only the final state


class ODE(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3):
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
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.initial_conv(x)
        x = self.ode_block1(x)
        x = self.downsample1(x)
        x = self.ode_block2(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x
