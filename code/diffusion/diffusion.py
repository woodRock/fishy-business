import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class DiffusionModel(nn.Module):
    def __init__(self, 
        input_dim: int, 
        hidden_dim: int, 
        num_classes: int, 
        num_steps: int = 100, 
        dropout_rate=0.1
    ) -> None:
        """ Diffusion model for classification.

        Args:
            input_dim (int): the input dimension.
            hidden_dim (int): the hidden dimension.
            num_classes (int): the number of classes.
            num_steps (int): the number of steps for the diffusion process.
            dropout_rate (float): the dropout rate. Defaults to 0.1.

        References: 
            1. Song, J., Meng, C., & Ermon, S. (2020). 
            Denoising diffusion implicit models. 
            arXiv preprint arXiv:2010.02502.
        
        """
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
        
    def forward(self, 
        x: torch.Tensor, 
        t: torch.Tensor
    ) -> torch.Tensor:
        """ Forward pass for the diffusion model.
        
        Args: 
            x (torch.Tensor): the input tensor.
            t (torch.Tensor): the time tensor.
        """
        t = t.float() / self.num_steps
        t = t.view(-1, 1)
        x_input = torch.cat([x, t], dim=-1)
        
        noise = self.net(x_input)
        x_denoised = x - noise
        
        logits = self.classifier(x_denoised)
        return logits