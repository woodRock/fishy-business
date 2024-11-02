import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import numpy as np
from tqdm import tqdm
# from torch.optim.lr_scheduler import ReduceLROnPlateau

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5, downsample=False):
        super(ResidualBlock, self).__init__()
        stride = 2 if downsample else 1
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(p=dropout)

        # Shortcut path: if in_channels and out_channels differ, adjust with a 1x1 conv
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or downsample:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)  # Match dimensions
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        
        out += residual  # Add the shortcut (residual) connection
        out = self.relu(out)
        return out

class RCNN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3):
        super(RCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            ResidualBlock(1, 32, dropout=dropout),  # First block expects 1 channel
            ResidualBlock(32, 64, dropout=dropout, downsample=True),  # Downsample here
            # ResidualBlock(64, 128, dropout=dropout),
            # ResidualBlock(128, 256, dropout=dropout, downsample=True),  # Downsample here
            nn.AdaptiveMaxPool1d(4)  # Fixed output size to 4
        )
        
        self.flatten = nn.Flatten()
        self.flat_features = 64 * 4  # Adjusted based on the pooling layer
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_features, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1) # Add channel dimension
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x