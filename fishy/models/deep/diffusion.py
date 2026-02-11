"""Diffusion model for spectral data analysis.

This module implements a 1D diffusion model for spectral data, incorporating
noise schedules and denoising steps for classification and sampling.


References:

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models.
   In Advances in Neural Information Processing Systems (pp. 6840-6851).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for time encoding.
    """

    def __init__(self, dim: int) -> None:
        """
        Initializes the embeddings.

        Args:
            dim (int): Dimension of the embeddings.
        """
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Computes the sinusoidal position embeddings.

        Args:
            time (torch.Tensor): Input tensor of shape (batch_size,).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, dim).
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """
    A basic convolutional block for the diffusion model.

    Consists of two convolutional layers with group normalization and residual connection.
    """

    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        """
        Initializes the block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            time_dim (int): Dimension of the time embeddings.
        """
        super().__init__()
        self.time_mlp = nn.Linear(time_dim, out_channels)

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, seq_length).
            t (torch.Tensor): Time tensor of shape (batch_size, time_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, seq_length).
        """
        h = self.block1(x)
        time_emb = self.time_mlp(t)[:, :, None]
        h = h + time_emb
        h = self.block2(h)
        return h + self.shortcut(x)


class Diffusion(nn.Module):
    """
    1D Diffusion model for spectral classification and generation.

    Attributes:
        betas (torch.Tensor): Noise schedule.
        alphas_cumprod (torch.Tensor): Cumulative product of alphas.
        time_mlp (nn.Sequential): MLP for processing time embeddings.
        classifier (nn.Sequential): Final classification head.
    """

    def __init__(
        self,
        input_dim: int = 2080,
        hidden_dim: int = 128,
        time_dim: int = 64,
        output_dim: int = 2,
        num_timesteps: int = 4000,
    ) -> None:
        """
        Initializes the diffusion model.

        Args:
            input_dim (int, optional): Size of the input spectrum. Defaults to 2080.
            hidden_dim (int, optional): Base hidden dimension. Defaults to 128.
            time_dim (int, optional): Time embedding dimension. Defaults to 64.
            output_dim (int, optional): Number of output classes. Defaults to 2.
            num_timesteps (int, optional): Number of diffusion steps. Defaults to 4000.
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_timesteps = num_timesteps

        # Noise schedule
        beta_start = 1e-4
        beta_end = 0.02
        self.register_buffer(
            "betas", torch.linspace(beta_start, beta_end, num_timesteps)
        )
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))

        # Time embedding
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Initial projection
        self.init_conv = nn.Conv1d(1, hidden_dim, 1)

        # Down blocks
        self.down1 = Block(hidden_dim, hidden_dim, time_dim)
        self.down2 = Block(hidden_dim, hidden_dim * 2, time_dim)
        self.down3 = Block(hidden_dim * 2, hidden_dim * 2, time_dim)

        # Middle blocks
        self.mid1 = Block(hidden_dim * 2, hidden_dim * 2, time_dim)
        self.mid2 = Block(hidden_dim * 2, hidden_dim * 2, time_dim)

        # Up blocks
        self.up1 = Block(hidden_dim * 2, hidden_dim * 2, time_dim)
        self.up2 = Block(hidden_dim * 2, hidden_dim, time_dim)
        self.up3 = Block(hidden_dim, hidden_dim, time_dim)

        # Output layers
        self.final_conv = nn.Conv1d(hidden_dim, 1, 1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(1, output_dim)
        )

        self.target_noise = None

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adds noise to the original input based on the timestep.

        Args:
            x_0 (torch.Tensor): Original input tensor of shape (batch_size, input_dim).
            t (torch.Tensor): Timestep tensor of shape (batch_size,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (noisy_input, noise)
        """
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod[t])

        x_noisy = (
            sqrt_alphas_cumprod.view(-1, 1) * x_0
            + sqrt_one_minus_alphas_cumprod.view(-1, 1) * noise
        )
        return x_noisy, noise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input spectrum of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output logits.
        """
        batch_size = x.shape[0]
        device = x.device

        # Handle training vs inference mode
        if self.training:
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
            x_noisy, noise = self.add_noise(x, t)
            self.target_noise = noise
        else:
            t = torch.zeros(batch_size, device=device, dtype=torch.long)
            x_noisy = x

        # Time embeddings
        t = self.time_mlp(t.float())

        # Initial projection
        x = x_noisy.unsqueeze(1)
        x = self.init_conv(x)

        # Down blocks
        x = self.down1(x, t)
        x = self.down2(x, t)
        x = self.down3(x, t)

        # Middle blocks
        x = self.mid1(x, t)
        x = self.mid2(x, t)

        # Up blocks
        x = self.up1(x, t)
        x = self.up2(x, t)
        x = self.up3(x, t)

        # Output
        denoised = self.final_conv(x)
        logits = self.classifier(denoised)

        return logits

    @torch.no_grad()
    def sample(self, num_samples: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Samples from the diffusion model using the reverse process.

        Args:
            num_samples (int): Number of samples to generate.
            device (Optional[torch.device], optional): Computing device. Defaults to None.

        Returns:
            torch.Tensor: Generated samples of shape (num_samples, input_dim).
        """
        from fishy._core.utils import get_device
        if device is None:
            device = get_device()
        self.eval()
        x = torch.randn(num_samples, self.input_dim, device=device)

        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            t_emb = self.time_mlp(t_batch.float())

            # Forward pass
            x_input = x.unsqueeze(1)
            h = self.init_conv(x_input)

            # Process through network
            h = self.down1(h, t_emb)
            h = self.down2(h, t_emb)
            h = self.down3(h, t_emb)

            h = self.mid1(h, t_emb)
            h = self.mid2(h, t_emb)

            h = self.up1(h, t_emb)
            h = self.up2(h, t_emb)
            h = self.up3(h, t_emb)

            denoised = self.final_conv(h)

            # Update sample
            alpha_t = self.alphas[t]
            alpha_t_cumprod = self.alphas_cumprod[t]
            beta_t = self.betas[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0

            x = (
                1
                / torch.sqrt(alpha_t)
                * (
                    x
                    - (beta_t / (torch.sqrt(1 - alpha_t_cumprod))) * denoised.squeeze(1)
                )
                + torch.sqrt(beta_t) * noise
            )

        self.train()
        return x


__all__ = ["Diffusion"]
