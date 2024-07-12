import torch
import torch.nn as nn
import torch.nn.functional as F

class KAN(nn.Module):
    def __init__(self, 
                input_dim: int, 
                output_dim: int,
                hidden_dim: int = 64, 
                num_inner_functions: int = 10, 
                dropout_rate: float = 0.1
    ) -> None:
        super(KAN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_inner_functions = num_inner_functions

        # Inner functions (vectorized)
        self.inner_linear1 = nn.Linear(input_dim, hidden_dim)
        self.inner_bn1 = nn.BatchNorm1d(hidden_dim)
        self.inner_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.inner_bn2 = nn.BatchNorm1d(hidden_dim)
        self.inner_linear3 = nn.Linear(hidden_dim, num_inner_functions * (2 * input_dim + 1))

        # Outer functions (vectorized)
        self.outer_linear1 = nn.Linear(num_inner_functions, hidden_dim)
        self.outer_bn1 = nn.BatchNorm1d(hidden_dim)
        self.outer_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.outer_bn2 = nn.BatchNorm1d(hidden_dim)
        self.outer_linear3 = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        batch_size = x.size(0)

        # Inner functions (vectorized)
        inner = F.leaky_relu(self.inner_bn1(self.inner_linear1(x)))
        inner = self.dropout(inner)
        inner = F.leaky_relu(self.inner_bn2(self.inner_linear2(inner)))
        inner = self.dropout(inner)
        inner = self.inner_linear3(inner)
        inner = inner.view(batch_size, 2 * self.input_dim + 1, self.num_inner_functions)

        # Add the constant term
        constant_term = torch.ones(batch_size, 1, self.num_inner_functions, device=x.device)
        inner = torch.cat([inner, constant_term], dim=1)

        # Sum across the inner functions
        summed = torch.sum(inner, dim=1)

        # Outer functions (vectorized)
        outer = F.leaky_relu(self.outer_bn1(self.outer_linear1(summed)))
        outer = self.dropout(outer)
        outer = F.leaky_relu(self.outer_bn2(self.outer_linear2(outer)))
        outer = self.dropout(outer)
        output = self.outer_linear3(outer)

        return output