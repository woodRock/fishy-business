import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class DiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_steps, dropout_rate=0.1):
        super(DiffusionModel, self).__init__()
        self.num_steps = num_steps
        
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x, t):
        t = t.float() / self.num_steps
        t = t.view(-1, 1)
        x_input = torch.cat([x, t], dim=-1)
        
        noise = self.net(x_input)
        x_denoised = x - noise
        
        logits = self.classifier(x_denoised)
        return logits