# -*- coding: utf-8 -*-
"""
Dataset definitions and types for the deep learning pipeline.
"""

from enum import Enum, auto
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

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
        """Converts a string to a DatasetType enum member."""
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
    """Dataset for contrastive learning, generating pairs of samples."""

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
