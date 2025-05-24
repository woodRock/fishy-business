import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

class RWKV(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(RWKV, self).__init__()
        self.input_size = input_dim
        self.hidden_dim = hidden_dim
        self.output_size = output_dim

        # Linear layers for key, value, and output
        self.key_layer = nn.Linear(input_dim, hidden_dim)
        self.value_layer = nn.Linear(input_dim, hidden_dim)
        self.recurrent_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Initialization
        self.hidden = None

    def forward(self, x):
        # Compute keys and values
        keys = self.key_layer(x)
        values = self.value_layer(x)

        # If hidden state is not initialized, initialize it
        if self.hidden is None:
            self.hidden = torch.zeros(x.size(0), self.hidden_dim).to(x.device)

        # Update hidden state with current keys and values
        self.hidden = self.hidden + torch.tanh(keys + self.recurrent_layer(values))

        # Compute output
        output = self.output_layer(self.hidden)

        # Reset the hidden state.
        self.hidden = None
        return output