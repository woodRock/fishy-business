# -*- coding: utf-8 -*-
"""
Momentum Contrast (MoCo) model.
"""

import torch
import torch.nn as nn
import copy


class MoCoModel(nn.Module):
    """
    MoCo architecture for self-supervised learning.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int = 128,
        projection_dim: int = 128,
        dropout: float = 0.2,
        K: int = 4096,
        m: float = 0.999,
        T: float = 0.07,
        **kwargs,
    ) -> None:
        super(MoCoModel, self).__init__()
        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = nn.Sequential(
            backbone, nn.Linear(embedding_dim, projection_dim)
        )
        self.encoder_k = copy.deepcopy(self.encoder_q)

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(projection_dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=1)

        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)

        self._dequeue_and_enqueue(k)
        return logits, labels


class MoCoLoss(nn.Module):
    def __init__(self):
        super(MoCoLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        return self.criterion(logits, labels)
