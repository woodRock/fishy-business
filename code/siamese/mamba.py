import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dropout = nn.Dropout(dropout)
        
        # Learnable parameters
        self.A = nn.Parameter(torch.randn(d_state, d_state))
        self.B = nn.Parameter(torch.randn(d_state, d_model))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        
        self.D = nn.Parameter(torch.randn(d_model, d_model))
        
    def forward(self, x):
        # x shape: (batch_size, d_model) or (batch_size, seq_len, d_model)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.d_state, device=x.device)
        
        outputs = []
        for t in range(seq_len):
            u = x[:, t, :]
            h = torch.tanh(self.A @ h.unsqueeze(-1) + self.B @ u.unsqueeze(-1)).squeeze(-1)
            y = self.C @ h.unsqueeze(-1) + self.D @ u.unsqueeze(-1)
            y = self.dropout(y)
            outputs.append(y)
        
        return torch.stack(outputs, dim=1).squeeze(-1)

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, dropout)
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.proj(x)
        return x + residual

class Mamba(nn.Module):
    def __init__(self, input_dim, output_dim, d_state, num_layers, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(input_dim, output_dim)
        self.layers = nn.ModuleList([MambaBlock(output_dim, d_state, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        x = self.embed(x)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension if not present
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x.mean(dim=1)  # Global average pooling