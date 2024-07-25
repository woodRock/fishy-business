import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock(nn.Module):
    def __init__(self, 
        d_model: int, 
        d_state: int, 
        d_conv: int, 
        expand: int,
        dropout: float = 0.2
    ) -> None:
        """ MambaBlock
        
        Args: 
            d_model (int): the input dimension.
            d_state (int): the state dimension.
            d_conv (int): the convolutional kernel size.
            expand (int): the expansion factor.
            dropout (float): the dropout rate. Defaults to 0.2.
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        
        self.in_proj = nn.Linear(d_model, expand * d_model)
        self.conv = nn.Conv1d(expand * d_model, expand * d_model, kernel_size=d_conv, padding=d_conv-1, groups=expand * d_model)
        self.activation = nn.SiLU()

        self.dropout = nn.Dropout(dropout)
        
        self.x_proj = nn.Linear(expand * d_model, d_state + d_model)
        self.dt_proj = nn.Linear(expand * d_model, d_state)
        
        self.out_proj = nn.Linear(expand * d_model, d_model)
        
        # Add a projection for us
        self.us_proj = nn.Linear(d_state, d_model)
        
    def forward(self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """ Forward pass

        Args: 
            x (torch.Tensor): the input tensor.

        Returns:
            x (torch.Tensor): the output tensor.
        """
        # This will give you the desired shape of [64,1,16] -> [64,100,16].
        B, L, D = x.shape
        
        x_in = self.in_proj(x)
        x_conv = x_in.transpose(1,2)
        x_conv = self.conv(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1,2)
        x_act = self.activation(x_conv)
        # Dropout
        x_act = self.dropout(x_act)
        
        x_and_ds = self.x_proj(x_act)
        dt = self.dt_proj(x_act)
        x, ds = x_and_ds.split([self.d_model, self.d_state], dim=-1)
        dt = F.softplus(dt)
        
        # Selective scan
        us = torch.zeros(B, L, self.d_state, device=x.device)
        for i in range(L):
            us[:, i] = us[:, i-1] if i > 0 else us[:, i]
            # us[:, i] = us[:, i] * torch.exp(-dt[:, i]) + ds[:, i] * torch.exp(-dt[:, i] / 2)
            us = us * torch.exp(-dt) + ds * torch.exp(-dt / 2)
        
        # Project us to match x's dimension
        us_projected = self.us_proj(us)
        
        x = x + us_projected

        # Dropout
        x = self.dropout(x)
        
        x = self.out_proj(torch.cat([x, x_act[:, :, self.d_model:]], dim=-1))
        # Dropout
        x = self.dropout(x)

        return x

class Mamba(nn.Module):
    def __init__(self, 
        d_model: int, 
        d_state: int, 
        d_conv: int, 
        expand: int, 
        depth: int, 
        n_classes=2,
        dropout: float = 0.2
    ):
        """ Mamba
        
        Args: 
            d_model (int): the input dimension.
            d_state (int): the state dimension.
            d_conv (int): the convolutional kernel size.
            expand (int): the expansion factor.
            depth (int): the depth of the model.
            n_classes (int): the number of classes.

        References: 
            1. Gu, A., & Dao, T. (2023). 
            Mamba: Linear-time sequence modeling with selective state spaces. 
            arXiv preprint arXiv:2312.00752.
        """
        super().__init__()
        self.layers = nn.ModuleList([MambaBlock(d_model, d_state, d_conv, expand) for _ in range(depth)])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, n_classes)
        
    def forward(self, x):
        """ Forward pass
        
        Args: 
            x (torch.Tensor): the input tensor.

        Returns: 
            x (torch.Tensor): the output tensor.
        """
        # Add sequence dimenion
        x = x.unsqueeze(1)
        # Make the sequence length 100
        x = x.repeat(1, 100, 1)
        for layer in self.layers:
            x = x + layer(x)
            # Apply dropout
            x = self.dropout(x)
        x = self.fc(x)
        # Remove the sequence dimension
        x = x[:, 0, :]
        return x