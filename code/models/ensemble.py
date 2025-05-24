import torch
import torch.nn as nn
from .transformer import Transformer 
from .lstm import LSTM 
from .mamba import Mamba

class Ensemble(nn.Module):
    """Simple stacked voting classifier combining LSTM, Transformer, and Mamba."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        self.device = device
        
        # Base models
        # self.lstm = LSTM(
        #     input_size=input_dim, 
        #     hidden_size=hidden_dim,
        #     num_layers=4,
        #     output_size=output_dim,
        #     dropout=dropout,
        # )

        self.t1 = Transformer(
            input_dim=input_dim,
            output_dim=output_dim,
            num_heads=2,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )
        
        self.t2 = Transformer(
            input_dim=input_dim,
            output_dim=output_dim,
            num_heads=4,
            hidden_dim=hidden_dim,
            num_layers=4,
            dropout=dropout,
        )

        self.t3 = Transformer(
            input_dim=input_dim,
            output_dim=output_dim,
            num_heads=8,
            hidden_dim=hidden_dim,
            num_layers=8,
            dropout=dropout,
        )
        
        # self.mamba = Mamba(
        #     d_model=input_dim,
        #     d_state=hidden_dim,
        #     d_conv=4,
        #     expand=2,
        #     depth=4,
        #     n_classes=output_dim,
        #     dropout=dropout,
        # )
        
        # Voting weights
        self.voting_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Move to device
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining predictions from all models."""
        x = x.to(self.device)
        
        # Get predictions from each model
        t1 = self.t1(x)
        t2 = self.t2(x)
        t3 = self.t3(x)
        
        # Weighted voting
        weighted_sum = (
            self.voting_weights[0] * t1 +
            self.voting_weights[1] * t2 +
            self.voting_weights[2] * t3
        )
        
        return weighted_sum

__all__ = ["Ensemble"]