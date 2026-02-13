# -*- coding: utf-8 -*-
"""
Dataset definitions and types for the deep learning pipeline.
"""

from enum import Enum, auto
from typing import Tuple, Iterator, List, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler


class BaseDataset(Dataset):
    def __init__(self, samples: np.ndarray, labels: np.ndarray) -> None:
        self.samples = torch.tensor(samples, dtype=torch.float32)
        self.labels = torch.tensor(np.array(labels), dtype=torch.float32)

        if self.samples.ndim > 1 and self.samples.shape[0] > 1:
            self.samples = F.normalize(self.samples, p=2, dim=0)

    def __len__(self) -> int:
        return self.samples.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx], self.labels[idx]


class CustomDataset(BaseDataset):
    """Standard PyTorch Dataset."""

    pass


class SiameseDataset(BaseDataset):
    """
    Dataset for contrastive learning, generating pairs of samples.
    Restored to return (x1, x2, pair_label, y1, y2) for balanced sampling.
    """

    def __init__(self, samples: np.ndarray, labels: np.ndarray) -> None:
        super().__init__(samples, labels)
        self.X1, self.X2, self.paired_labels, self.y1, self.y2 = (
            self._generate_pairs_vectorized(self.samples, self.labels)
        )

    def _generate_pairs_vectorized(
        self, original_samples: torch.Tensor, original_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n_samples = original_samples.shape[0]
        if n_samples < 2:
            return (
                torch.empty((0, original_samples.shape[1])),
                torch.empty((0, original_samples.shape[1])),
                torch.empty((0, 1)),
                torch.empty(
                    (0, original_labels.shape[1] if original_labels.ndim > 1 else 1)
                ),
                torch.empty(
                    (0, original_labels.shape[1] if original_labels.ndim > 1 else 1)
                ),
            )

        indices_i = torch.arange(n_samples).unsqueeze(1).expand(-1, n_samples).flatten()
        indices_j = torch.arange(n_samples).unsqueeze(0).expand(n_samples, -1).flatten()

        mask = indices_i < indices_j
        indices_i, indices_j = indices_i[mask], indices_j[mask]

        X1, X2 = original_samples[indices_i], original_samples[indices_j]
        y1, y2 = original_labels[indices_i], original_labels[indices_j]

        if y1.ndim > 1:
            same_label_mask = torch.all(y1 == y2, dim=1)
        else:
            same_label_mask = y1 == y2

        pair_labels_tensor = same_label_mask.to(torch.float32).unsqueeze(1)
        return X1, X2, pair_labels_tensor, y1, y2

    def __len__(self) -> int:
        return self.paired_labels.shape[0]

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.X1[idx],
            self.X2[idx],
            self.paired_labels[idx],
            self.y1[idx],
            self.y2[idx],
        )


class BalancedBatchSampler(Sampler):
    """
    Restored from original implementation to ensure 1:1 ratio of positive/negative pairs.
    """

    def __init__(
        self, pair_labels: Union[torch.Tensor, np.ndarray], batch_size: int
    ) -> None:
        if isinstance(pair_labels, torch.Tensor):
            self.labels = pair_labels.flatten().cpu().numpy()
        else:
            self.labels = np.array(pair_labels).flatten()

        self.neg_indices = np.where(self.labels == 0)[0]
        self.pos_indices = np.where(self.labels == 1)[0]
        self.batch_size = batch_size

        if len(self.neg_indices) == 0 or len(self.pos_indices) == 0:
            self.num_batches = 0
        else:
            self.num_batches = min(len(self.neg_indices), len(self.pos_indices)) // (
                batch_size // 2
            )

    def __iter__(self) -> Iterator[List[int]]:
        np.random.shuffle(self.neg_indices)
        np.random.shuffle(self.pos_indices)

        half_batch = self.batch_size // 2
        for i in range(self.num_batches):
            batch = []
            batch.extend(
                self.pos_indices[i * half_batch : (i + 1) * half_batch].tolist()
            )
            batch.extend(
                self.neg_indices[i * half_batch : (i + 1) * half_batch].tolist()
            )
            np.random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches
