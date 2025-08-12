""" MoCo model implementation for self-supervised learning.

This module defines the MoCo model architecture, which includes a query encoder, a key encoder, and a momentum mechanism for self-supervised learning tasks, particularly in computer vision.

References:
1.  He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020).
    Momentum contrast for unsupervised visual representation learning. 
    In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 9729-9738).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class MoCoModel(nn.Module):
    """MoCo model with a query encoder, key encoder, and a momentum mechanism."""

    def __init__(
        self,
        encoder: nn.Module,
        encoder_output_dim: int,
        dim: int = 256,
        K: int = 65536,
        m: float = 0.999,
        T: float = 0.07,
        mlp: bool = False,
    ):
        """
        dim: feature dimension (default: 256)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        mlp: whether to use mlp projection head (default: False)
        """
        super(MoCoModel, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        self.encoder_q = encoder
        self.encoder_k = copy.deepcopy(encoder)

        # Create projection heads for query and key encoders
        if mlp:
            self.projector_q = nn.Sequential(
                nn.Linear(encoder_output_dim, encoder_output_dim),
                nn.ReLU(),
                nn.Linear(encoder_output_dim, dim),
            )
            self.projector_k = nn.Sequential(
                nn.Linear(encoder_output_dim, encoder_output_dim),
                nn.ReLU(),
                nn.Linear(encoder_output_dim, dim),
            )
        else:
            self.projector_q = nn.Linear(encoder_output_dim, dim)
            self.projector_k = nn.Linear(encoder_output_dim, dim)

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # no gradient for key encoder

        for param_q, param_k in zip(
            self.projector_q.parameters(), self.projector_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # no gradient for key projector

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder"""
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

        for param_q, param_k in zip(
            self.projector_q.parameters(), self.projector_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q: torch.Tensor, im_k: torch.Tensor):
        """
        im_q: a batch of query images
        im_k: a batch of key images
        """
        # compute query features
        q = self.projector_q(self.encoder_q(im_q))  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to key encoder
            self._momentum_update_key_encoder()  # update the key encoder
            k = self.projector_k(self.encoder_k(im_k))  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return q, k, self.queue.clone().detach()


class MoCoLoss(nn.Module):
    """MoCo loss function."""

    def __init__(self, T: float = 0.07):
        super(MoCoLoss, self).__init__()
        self.T = T

    def forward(self, q: torch.Tensor, k: torch.Tensor, queue: torch.Tensor):
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, queue.detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)

        loss = F.cross_entropy(logits, labels)
        return loss


__all__ = ["MoCoModel", "MoCoLoss"]
