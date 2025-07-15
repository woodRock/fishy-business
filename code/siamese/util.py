""" Siamese Network Utilities
This module provides utilities for setting up a Siamese dataset, data preprocessing,
and balanced sampling for contrastive learning tasks. It includes a custom dataset class,
a balanced sampler, and a data preprocessor to handle loading and filtering of data.
It also sets up logging for debugging and information purposes."""
import logging
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterator, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def setup_logger(name: str) -> logging.Logger:
    """Set up logger with both file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "siamese_dataset.log")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


# Create logger instance
logger = setup_logger(__name__)


@dataclass
class DataConfig:
    """Configuration for dataset preprocessing."""

    dataset_name: str = "instance-recognition"
    batch_size: int = 64
    pairs_per_sample: int = 50
    test_size: float = 0.4
    data_path: List[str] = None

    def __post_init__(self):
        if self.data_path is None:
            self.data_path = ["~/", "Desktop", "fishy-business", "data", "REIMS.xlsx"]
            # self.data_path = ["/vol", "ecrg-solar", "woodj4", "fishy-business", "data", "REIMS.xlsx"]


class SiameseDataset(Dataset):
    """Dataset for contrastive learning with all possible pairs."""

    def __init__(self, samples: np.ndarray, labels: np.ndarray):
        """Initialize Siamese dataset.
        Args:
            samples: Input features
            labels: Target labels
        """
        super().__init__()
        self.samples = samples
        self.labels = labels
        self.samples, self.labels = self._generate_pairs()

    def _generate_pairs(self) -> Tuple[List[torch.Tensor], np.ndarray]:
        """Generate all possible pairs for contrastive learning."""
        pairs = []
        labels = []
        n_samples = len(self.samples)

        # Generate all possible pairs
        for i in range(n_samples):
            for j in range(n_samples):
                if i != j:  # Exclude self-pairs
                    X1, y1 = self.samples[i], self.labels[i]
                    X2, y2 = self.samples[j], self.labels[j]

                    pair_feature = (X1, X2)
                    pair_label = (y1 == y2).all()

                    pairs.append(pair_feature)
                    labels.append(pair_label)

        labels = np.asarray(labels, dtype=int)
        print(f"unique labels: {np.unique(labels)}")
        n_classes = len(np.unique(labels, return_counts=True))
        one_hot_labels = np.eye(n_classes)[labels]
        print(
            f"unique onehot labels: {np.unique(one_hot_labels, axis=0, return_counts=True)}"
        )

        return pairs, one_hot_labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.samples[idx][0], self.samples[idx][1], self.labels[idx]


class ContrastiveBalancedSampler(Sampler):
    """
    Balanced sampler for contrastive learning that ensures equal class representation
    in each batch. Works with both numpy arrays and PyTorch tensors, one-hot encoded
    or class indices.

    Args:
        labels: One-hot encoded labels or class indices (numpy array or torch.Tensor)
        batch_size: Size of each batch
        num_samples_per_class: Number of samples to draw per class in each batch.
                             If None, will be calculated from batch_size
        drop_last: If True, drop the last incomplete batch
    """

    def __init__(
        self,
        labels: Union[np.ndarray, torch.Tensor],
        batch_size: int,
        num_samples_per_class: Optional[int] = None,
        drop_last: bool = False,
    ):
        # Convert to numpy array if tensor
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        # Convert one-hot labels to class indices if necessary
        if len(labels.shape) > 1:
            self.labels = np.argmax(labels, axis=1)
        else:
            self.labels = labels

        self.num_classes = len(np.unique(self.labels))
        self.class_indices = [
            np.where(self.labels == i)[0] for i in range(self.num_classes)
        ]

        # Calculate samples per class if not provided
        if num_samples_per_class is None:
            self.samples_per_class = batch_size // self.num_classes
        else:
            self.samples_per_class = num_samples_per_class

        self.batch_size = self.samples_per_class * self.num_classes
        self.drop_last = drop_last

        # Calculate number of batches
        min_class_size = min(len(indices) for indices in self.class_indices)
        self.num_batches = min_class_size // self.samples_per_class
        if not self.drop_last and min_class_size % self.samples_per_class != 0:
            self.num_batches += 1

        # Print class distribution info
        for i in range(self.num_classes):
            print(f"Class {i}: {len(self.class_indices[i])} samples")
        print(f"Samples per class per batch: {self.samples_per_class}")
        print(f"Total batch size: {self.batch_size}")
        print(f"Number of batches: {self.num_batches}")

    def __iter__(self) -> Iterator[List[int]]:
        # Create a copy of class indices to avoid modifying original
        class_indices = [indices.copy() for indices in self.class_indices]

        # Shuffle indices for each class
        for indices in class_indices:
            np.random.shuffle(indices)

        # Keep track of current position in each class
        class_positions = [0] * self.num_classes

        # Generate batches
        for _ in range(self.num_batches):
            batch_indices = []

            # Sample from each class
            for class_idx in range(self.num_classes):
                start_idx = class_positions[class_idx]
                end_idx = start_idx + self.samples_per_class

                # If we need more samples than available, reshuffle and reset
                if end_idx > len(class_indices[class_idx]):
                    if self.drop_last:
                        continue
                    # Reshuffle class indices
                    np.random.shuffle(class_indices[class_idx])
                    class_positions[class_idx] = 0
                    start_idx = 0
                    end_idx = self.samples_per_class

                # Add indices to batch
                batch_indices.extend(
                    class_indices[class_idx][start_idx:end_idx].tolist()
                )
                class_positions[class_idx] = end_idx

            if len(batch_indices) == self.batch_size or not self.drop_last:
                # Yield the entire batch as a list
                yield batch_indices

    def __len__(self) -> int:
        return self.num_batches


class DataPreprocessor:
    """Handle data loading and preprocessing for Siamese networks."""

    @staticmethod
    def load_data(config: DataConfig) -> pd.DataFrame:
        path = Path(*config.data_path).expanduser()
        return pd.read_excel(path)

    @staticmethod
    def filter_data(data: pd.DataFrame, dataset: str) -> pd.DataFrame:
        data = data[~data["m/z"].str.contains("QC", case=False, na=False)]

        if dataset in ["species", "part", "oil", "instance-recognition"]:
            data = data[~data["m/z"].str.contains("HM", case=False, na=False)]

        if dataset in ["species", "part", "cross-species"]:
            data = data[~data["m/z"].str.contains("MO", case=False, na=False)]

        if dataset == "instance-recognition":
            pattern = r"QC|HM|MO|fillet|frames|gonads|livers|skins|guts|frame|heads"
            data = data[
                ~data.iloc[:, 0].astype(str).str.contains(pattern, case=False, na=False)
            ]

        if len(data) == 0:
            raise ValueError(f"No data remaining after filtering for dataset {dataset}")

        return data

    @staticmethod
    def encode_labels(data: pd.DataFrame, dataset: str) -> np.ndarray:
        if dataset == "instance-recognition":
            labels = data.iloc[:, 0].to_numpy()
            encoder = LabelEncoder()
            encoded = encoder.fit_transform(labels)
            n_classes = len(np.unique(encoded))
            return np.eye(n_classes)[encoded]

        encoding_patterns = {
            "species": {"H": [0, 1], "default": [1, 0]},
            "part": {
                "Fillet": [1, 0, 0, 0, 0, 0],
                "Heads": [0, 1, 0, 0, 0, 0],
                "Livers": [0, 0, 1, 0, 0, 0],
                "Skins": [0, 0, 0, 1, 0, 0],
                "Guts": [0, 0, 0, 0, 1, 0],
                "Frames": [0, 0, 0, 0, 0, 1],
            },
            "cross-species": {"HM": [1, 0, 0], "H": [0, 1, 0], "M": [0, 0, 1]},
        }

        if dataset not in encoding_patterns:
            raise ValueError(f"Invalid dataset: {dataset}")

        pattern = encoding_patterns[dataset]
        labels = data["m/z"].apply(
            lambda x: [
                pattern.get(key)[i]
                for key in pattern.keys()
                for i in range(len(pattern[key]))
                if key in x
            ]
        )

        if len(labels) == 0:
            raise ValueError(f"No valid labels found for dataset {dataset}")

        return labels.values


def prepare_dataset(config: DataConfig) -> Tuple[DataLoader, DataLoader]:
    preprocessor = DataPreprocessor()

    data = preprocessor.load_data(config)
    data = preprocessor.filter_data(data, config.dataset_name)

    features = data.drop("m/z", axis=1).to_numpy()
    labels = preprocessor.encode_labels(data, config.dataset_name)

    if len(features) < 2:
        raise ValueError(
            f"Not enough samples ({len(features)}) to split into train/val sets"
        )

    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, stratify=labels, test_size=config.test_size, shuffle=True
    )

    train_dataset = SiameseDataset(X_train, y_train)
    val_dataset = SiameseDataset(X_val, y_val)

    # Create a balanced sampler for contrastive learning.
    train_sampler = ContrastiveBalancedSampler(
        labels=train_dataset.labels,  # Your dataset's labels
        batch_size=32,  # Desired batch size
        drop_last=False,  # Whether to drop incomplete batches
    )

    val_sampler = ContrastiveBalancedSampler(
        labels=val_dataset.labels,  # Your dataset's labels
        batch_size=32,  # Desired batch size
        drop_last=False,  # Whether to drop incomplete batches
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        # batch_size=64,
        # shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        # batch_size=64,
        # shuffle=True,
    )

    return train_loader, val_loader


def inspect_dataloaders(train_loader: DataLoader, val_loader: DataLoader) -> None:
    for name, loader in [("Training", train_loader), ("Validation", val_loader)]:
        class_counts = {0: 0, 1: 0}
        features = []

        for X1, X2, labels in loader:
            for label in [0, 1]:
                class_counts[label] += (labels == label).sum().item()
            features.append(torch.cat((X1, X2), dim=0))

        features = torch.cat(features, dim=0)
        class_dist = f"Class distribution: {class_counts}"
        feat_mean = f"Feature mean range: [{features.mean(0).min():.3f}, {features.mean(0).max():.3f}]"
        feat_std = f"Feature std range: [{features.std(0).min():.3f}, {features.std(0).max():.3f}]"
        logger.info(f"{name} Set - {class_dist} | {feat_mean} | {feat_std}")


if __name__ == "__main__":
    config = DataConfig()
    train_loader, val_loader = prepare_dataset(config)
    inspect_dataloaders(train_loader, val_loader)

    # Graph the mass spectrograph for the first instance in the train loader.
    # This is a 1D array with 2080 features.
    import matplotlib.pyplot as plt

    plt.plot(train_loader.dataset.samples[0][0])

    # Set the title and x-axis label
    plt.title("Mass Spectrograph")

    # Set the x-axis label
    plt.xlabel("Mass-to-charge ratio")

    # Make the x-ticks go from (77.04-999.32 m/z), with 2080 ticks.
    # Define the tick locations and labels
    tick_locations = np.arange(0, 2080, step=500)
    tick_labels = np.linspace(77.04, 999.32, num=len(tick_locations))

    # Set the tick locations and labels
    plt.xticks(tick_locations, labels=np.round(tick_labels, 2))

    # Set the y-axis label
    plt.ylabel("Intensity")

    # Save the figure
    plt.savefig("figures/mass_spectrograph.png")
