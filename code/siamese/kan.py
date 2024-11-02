import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

class KANLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        return self.activation(self.linear(x))

class KAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_inner_functions, dropout=0.1):
        super().__init__()
        self.input_layer = KANLayer(input_dim, hidden_dim)
        self.inner_functions = nn.ModuleList([KANLayer(hidden_dim, hidden_dim) for _ in range(num_inner_functions)])
        self.output_layer = nn.Linear(hidden_dim * num_inner_functions, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        inner_outputs = [f(x) for f in self.inner_functions]
        x = torch.cat(inner_outputs, dim=1)
        x = self.output_layer(x)
        return x