import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import numpy as np
from tqdm import tqdm

class CNN(nn.Module):
    def __init__(self, input_dim, output_dim, input_channels, d_model, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, output_dim)  # Changed to output_dim
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)  # Changed to match conv3 output channels
        
    def forward(self, x):
        # Handle both [batch_size, seq_len, features] and [batch_size, features]
        if len(x.shape) == 3:
            # [batch_size, seq_len, features] -> [batch_size, channels, features]
            x = x.squeeze(1)
        
        # [batch_size, features] -> [batch_size, channels, features]
        x = x.unsqueeze(1)
        
        # Apply convolutions
        x = self.bn1(torch.relu(self.conv1(x)))
        x = self.bn2(torch.relu(self.conv2(x)))
        x = self.bn3(torch.relu(self.conv3(x)))
        
        # Global pooling and final FC layer
        x = self.pool(x).squeeze(-1)
        x = self.dropout(self.fc(x))
        
        return x