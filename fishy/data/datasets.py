# -*- coding: utf-8 -*-
"""
Dataset definitions and types for the deep learning pipeline.
"""

from enum import Enum, auto
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from typing import Iterator, List


class DatasetType(Enum):
    """Enumeration for the different types of datasets."""
    SPECIES = auto()
    PART = auto()
    OIL = auto()
    OIL_SIMPLE = auto()
    OIL_REGRESSION = auto()
    CROSS_SPECIES = auto()
    CROSS_SPECIES_HARD = auto()
    INSTANCE_RECOGNITION = auto()
    INSTANCE_RECOGNITION_HARD = auto()

    @classmethod
    def from_string(cls, name: str) -> "DatasetType":
        """
        Converts a string to a DatasetType enum member.

        Examples:
            >>> DatasetType.from_string("species")
            <DatasetType.SPECIES: 1>
            >>> DatasetType.from_string("cross-species-hard")
            <DatasetType.CROSS_SPECIES_HARD: 7>
        """
        alias_map = {
            "species": cls.SPECIES,
            "part": cls.PART,
            "oil": cls.OIL,
            "oil_simple": cls.OIL_SIMPLE,
            "oil_regression": cls.OIL_REGRESSION,
            "cross-species": cls.CROSS_SPECIES,
            "cross-species-hard": cls.CROSS_SPECIES_HARD,
            "instance-recognition": cls.INSTANCE_RECOGNITION,
            "instance-recognition-hard": cls.INSTANCE_RECOGNITION_HARD,
        }
        target_name = name.lower()
        if target_name in alias_map:
            return alias_map[target_name]
        raise ValueError(f"Invalid dataset name: {name}. Must be one of {list(alias_map.keys())}")

class BaseDataset(Dataset):
    """
    Base class for spectral datasets.

    Examples:
        >>> import numpy as np
        >>> samples = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> labels = np.array([0, 1])
        >>> dataset = BaseDataset(samples, labels)
        >>> len(dataset)
        2
        >>> s, l = dataset[0]
        >>> torch.is_tensor(s)
        True
    """
    def __init__(self, samples: np.ndarray, labels: np.ndarray) -> None:
        self.samples = torch.tensor(samples, dtype=torch.float32)
        self.labels = torch.tensor(np.array(labels), dtype=torch.float32)

        if self.samples.ndim > 1 and self.samples.shape[0] > 1:
            self.samples = F.normalize(self.samples, p=2, dim=0)
        elif self.samples.ndim == 1 and self.samples.numel() > 0:
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

    Examples:
        >>> import numpy as np
        >>> samples = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        >>> labels = np.array([0, 1, 0])
        >>> dataset = SiameseDataset(samples, labels)
        >>> len(dataset) # C(3, 2) = 3
        3
        >>> x1, x2, label, y1, y2 = dataset[0]
        >>> label.shape
        torch.Size([1])
    """

    def __init__(self, samples: np.ndarray, labels: np.ndarray) -> None:
        super().__init__(samples, labels)
        (
            self.X1,
            self.X2,
            self.paired_labels,
            self.y1,
            self.y2,
        ) = self._generate_pairs_vectorized(self.samples, self.labels)

    def _generate_pairs_vectorized(
        self, original_samples: torch.Tensor, original_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n_samples = original_samples.shape[0]
        if n_samples < 2:
            return (
                torch.empty((0, original_samples.shape[1]), dtype=original_samples.dtype),
                torch.empty((0, original_samples.shape[1]), dtype=original_samples.dtype),
                torch.empty((0, 1), dtype=torch.float32),
                torch.empty((0, original_labels.shape[1]), dtype=original_labels.dtype),
                torch.empty((0, original_labels.shape[1]), dtype=original_labels.dtype),
            )

        indices_i = torch.arange(n_samples).unsqueeze(1).expand(-1, n_samples).flatten()
        indices_j = torch.arange(n_samples).unsqueeze(0).expand(n_samples, -1).flatten()

        mask = indices_i < indices_j
        indices_i, indices_j = indices_i[mask], indices_j[mask]

        X1, X2 = original_samples[indices_i], original_samples[indices_j]
        y1, y2 = original_labels[indices_i], original_labels[indices_j]

        if y1.ndim > 1 and y1.shape[1] > 1:
            same_label_mask = torch.all(y1 == y2, dim=1)
        else:
            same_label_mask = y1.squeeze() == y2.squeeze()

        pair_labels_tensor = same_label_mask.to(torch.float32).unsqueeze(1)
        return X1, X2, pair_labels_tensor, y1, y2

    def __len__(self) -> int:
        return self.paired_labels.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.X1[idx],
            self.X2[idx],
            self.paired_labels[idx],
            self.y1[idx],
            self.y2[idx],
        )


class BalancedBatchSampler(Sampler):
    """
    Generates balanced batches of positive and negative pairs for contrastive learning.

    Examples:
        >>> import numpy as np
        >>> pair_labels = np.array([1, 0, 1, 0, 1, 0])
        >>> sampler = BalancedBatchSampler(pair_labels, batch_size=4)
        >>> len(sampler)
        1
        >>> indices = next(iter(sampler))
        >>> len(indices)
        4
    """

    def __init__(self, pair_labels: np.ndarray, batch_size: int) -> None:
        """
        Initializes the sampler.

        Args:
            pair_labels (np.ndarray): Array of pair labels (0 for dissimilar, 1 for similar).
                                     Can be one-hot or flat.
            batch_size (int): Size of batches to generate.
        """
        if pair_labels.ndim > 1 and pair_labels.shape[1] > 1:
            class_labels = np.argmax(pair_labels, axis=1)
        else:
            class_labels = pair_labels.flatten()

        self.neg_indices = np.where(class_labels == 0)[0]
        self.pos_indices = np.where(class_labels == 1)[0]
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
            start_pos = i * half_batch
            start_neg = i * half_batch
            batch.extend(self.pos_indices[start_pos : start_pos + half_batch].tolist())
            batch.extend(self.neg_indices[start_neg : start_neg + half_batch].tolist())
            np.random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches
