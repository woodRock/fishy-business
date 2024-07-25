import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock(nn.Module):
    def __init__(self, 
        d_model: int, 
        d_state: int, 
        d_conv: int, 
        expand: int,
        dropout: float = 0.2,
        layer_norm_eps: float = 1e-5,
        weight_decay: float = 0.01
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        
        self.in_proj = nn.Linear(d_model, expand * d_model)
        self.conv = nn.Conv1d(expand * d_model, expand * d_model, kernel_size=d_conv, padding=d_conv-1, groups=expand * d_model)
        self.activation = nn.SiLU()

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        self.x_proj = nn.Linear(expand * d_model, d_state + d_model)
        self.dt_proj = nn.Linear(expand * d_model, d_state)
        
        self.out_proj = nn.Linear(expand * d_model, d_model)
        self.us_proj = nn.Linear(d_state, d_model)
        
        # L2 regularization
        self.weight_decay = weight_decay
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        x = self.layer_norm(x)  # Layer normalization
        
        x_in = self.in_proj(x)
        x_conv = x_in.transpose(1,2)
        x_conv = self.conv(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1,2)
        x_act = self.activation(x_conv)
        x_act = self.dropout(x_act)
        
        x_and_ds = self.x_proj(x_act)
        dt = self.dt_proj(x_act)
        x, ds = x_and_ds.split([self.d_model, self.d_state], dim=-1)
        dt = F.softplus(dt)
        
        us = torch.zeros(B, L, self.d_state, device=x.device)
        for i in range(L):
            us[:, i] = us[:, i-1] if i > 0 else us[:, i]
            us = us * torch.exp(-dt) + ds * torch.exp(-dt / 2)
        
        us_projected = self.us_proj(us)
        
        x = x + us_projected
        x = self.dropout(x)
        
        x = self.out_proj(torch.cat([x, x_act[:, :, self.d_model:]], dim=-1))
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
        dropout: float = 0.2,
        layer_norm_eps: float = 1e-5,
        weight_decay: float = 0.01,
        spectral_norm: bool = True
    ):
        super().__init__()
        self.layers = nn.ModuleList([MambaBlock(d_model, d_state, d_conv, expand, dropout, layer_norm_eps, weight_decay) for _ in range(depth)])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, n_classes)
        
        if spectral_norm:
            self.fc = nn.utils.spectral_norm(self.fc)
        
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, 100, 1)
        
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual  # Residual connection
            x = self.dropout(x)
        
        x = self.layer_norm(x)  # Final layer normalization
        x = self.fc(x[:, 0, :])
        return x
    
    def get_l2_regularization(self):
        l2_reg = 0.0
        for layer in self.layers:
            for param in layer.parameters():
                l2_reg += torch.norm(param, p=2)
        return layer.weight_decay * l2_reg

# Usage example:
# model = Mamba(d_model=16, d_state=16, d_conv=4, expand=2, depth=4)
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
# 
# # During training:
# loss = criterion(model(x), y) + model.get_l2_regularization()
# loss.backward()
# optimizer.step()