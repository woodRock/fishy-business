# Create this new file at: siamese/util.py
"""
Final, robust version of the Siamese Network Utilities.
Includes corrected samplers, data loaders, and detailed debugging.
"""
import os
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Iterator, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass
class DataConfig:
    """Configuration for dataset preprocessing."""

    dataset_name: str = "instance-recognition"
    batch_size: int = 32
    test_size: float = 0.5
    data_path: Union[str, List[str]] = None


class DataPreprocessor:
    """Handles robust data loading and filtering."""

    @staticmethod
    def load_data(config: DataConfig) -> pd.DataFrame:
        path = Path(config.data_path).expanduser()
        if not path.is_absolute():
            raise ValueError("Data path must be absolute")

        if not path.exists():
            raise FileNotFoundError(
                f"Data file not found at the specified path: {path}"
            )
        return pd.read_excel(path)

    @staticmethod
    def filter_data(data: pd.DataFrame, dataset: str) -> pd.DataFrame:
        data = data[~data["m/z"].str.contains("QC", case=False, na=False)]
        if dataset == "instance-recognition":
            pattern = r"QC|HM|MO|fillet|frames|gonads|livers|skins|guts|frame|heads"
            data = data[
                ~data.iloc[:, 0].astype(str).str.contains(pattern, case=False, na=False)
            ]
        return data

    @staticmethod
    def encode_labels(data: pd.DataFrame, dataset: str) -> np.ndarray:
        if dataset == "instance-recognition":
            labels = data.iloc[:, 0].to_numpy()
            encoder = LabelEncoder()
            encoded = encoder.fit_transform(labels)
            return np.eye(len(encoder.classes_))[encoded]
        raise ValueError(f"Dataset type '{dataset}' not supported for encoding.")

    @staticmethod
    def extract_groups(data: pd.DataFrame) -> np.ndarray:
        """Extracts group labels from the sample names in the 'm/z' column."""
        # Assumes group is the part of the name before the first '_'
        sample_names = data["m/z"].astype(str)
        groups = sample_names.str.split("_").str[0]
        return groups.to_numpy()


class SiameseDataset(Dataset):
    """Generates positive and negative pairs from samples."""

    def __init__(self, samples: np.ndarray, labels: np.ndarray):
        self.samples = torch.from_numpy(samples).float()
        self.labels = labels
        self.pairs, self.pair_labels = self._generate_pairs()

    def _generate_pairs(self) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        pairs, labels = [], []
        n_samples = len(self.samples)

        if n_samples < 2:
            return [], np.empty((0, 2), dtype=np.float32)

        label_indices = [np.where(r == 1)[0][0] for r in self.labels]

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                pairs.append((i, j))
                is_similar = 1 if label_indices[i] == label_indices[j] else 0
                labels.append(is_similar)

        # one-hot: [dissimilar, similar] -> index 0=dissimilar, 1=similar
        one_hot_labels = np.eye(2)[np.array(labels, dtype=np.int64)]
        return pairs, one_hot_labels

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx1, idx2 = self.pairs[idx]
        return (
            self.samples[idx1],
            self.samples[idx2],
            torch.from_numpy(self.pair_labels[idx]).float(),
        )


class BalancedBatchSampler(Sampler):
    """Generates balanced batches of positive and negative pairs."""

    def __init__(self, pair_labels: np.ndarray, batch_size: int):
        class_labels = np.argmax(pair_labels, axis=1)
        self.neg_indices = np.where(class_labels == 0)[0]
        self.pos_indices = np.where(class_labels == 1)[0]
        self.batch_size = batch_size

        # Ensure we have pairs of both classes
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
            batch.extend(self.pos_indices[start_pos : start_pos + half_batch])
            batch.extend(self.neg_indices[start_neg : start_neg + half_batch])
            random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches
