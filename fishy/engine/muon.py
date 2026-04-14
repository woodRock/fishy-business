# -*- coding: utf-8 -*-
"""
Muon Optimizer: Orthogonalized gradient descent.
Adapted from the "Parameter Golf" challenge (modded-nanogpt).
"""

import torch
import torch.nn.functional as F
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 10, eps: float = 1e-7) -> torch.Tensor:
    """
    Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    Muon uses this to normalize matrix-shaped gradients before applying them.
    """
    # coefficients for 5th order Newton-Schulz
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # Work in a stable dtype
    orig_type = G.dtype
    X = G.to(torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32)
    
    X /= X.norm() + eps
    
    # Ensure X is "wide" for faster matmuls
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
        
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
        
    res = X.T if transposed else X
    return res.to(orig_type)


class Muon(Optimizer):
    """
    Muon: An optimizer that orthogonalizes updates for matrix parameters.
    Typically used for Linear layer weights, while AdamW/SGD is used for vectors.
    """
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, backend_steps: int = 5, nesterov: bool = True):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
            
        defaults = dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            steps = group["backend_steps"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                g = p.grad
                if g.ndim != 2:
                    # Muon is only designed for 2D matrices.
                    # This class assumes the caller has partitioned parameters correctly.
                    # If a vector accidentally ends up here, we fall back to standard SGD.
                    pass 

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                # Apply Newton-Schulz orthogonalization (The Muon step)
                if g.ndim == 2:
                    g = zeropower_via_newtonschulz5(g, steps=steps)
                    # Scale correction from Muon reference
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                
                p.add_(g, alpha=-lr)

        return loss
