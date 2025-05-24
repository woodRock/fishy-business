import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import numpy as np
from tqdm import tqdm

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, dropout=0.5):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm1d(in_channels + i * growth_rate),
                nn.ReLU(),
                nn.Conv1d(in_channels + i * growth_rate, growth_rate, 
                         kernel_size=3, padding=1),
                nn.Dropout(p=dropout)
            )
            self.layers.append(layer)
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super(TransitionLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.Dropout(p=dropout),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.layers(x)

class Dense(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3):
        super(Dense, self).__init__()
        
        # Initial convolution
        self.first_conv = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        
        # DenseNet configuration
        growth_rate = 16
        num_layers_per_block = 4
        
        # First dense block
        self.dense1 = DenseBlock(32, growth_rate, num_layers_per_block, dropout)
        num_channels = 32 + growth_rate * num_layers_per_block
        
        # Transition layer
        self.trans1 = TransitionLayer(num_channels, num_channels // 2, dropout)
        num_channels = num_channels // 2
        
        # Second dense block
        self.dense2 = DenseBlock(num_channels, growth_rate, num_layers_per_block, dropout)
        num_channels = num_channels + growth_rate * num_layers_per_block
        
        # Global pooling
        self.global_pool = nn.AdaptiveMaxPool1d(4)
        
        # Calculate the flattened features size
        self.flat_features = num_channels * 4
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_features, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.first_conv(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        return x

__all__ = ["Dense"] # Export the Dense model for use in other modules