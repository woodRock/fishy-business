from dataclasses import dataclass
from enum import Enum, auto
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from transformer import Transformer

logger = logging.getLogger(__name__)


class DatasetType(Enum):
    """Enumeration of available dataset types."""

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
        """Convert string to DatasetType enum."""
        mapping = {
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
        if name.lower() not in mapping:
            raise ValueError(
                f"Invalid dataset name: {name}. Must be one of {list(mapping.keys())}"
            )
        return mapping[name.lower()]


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""

    enabled: bool = False
    num_augmentations: int = 5
    noise_enabled: bool = True
    shift_enabled: bool = False
    scale_enabled: bool = False
    noise_level: float = 0.1
    shift_range: float = 0.1
    scale_range: float = 0.1


class BaseDataset(Dataset):
    """Base dataset class with common functionality."""

    def __init__(self, samples: np.ndarray, labels: np.ndarray):
        """Initialize dataset with samples and labels.

        Args:
            samples: Input features
            labels: Target labels
        """
        self.samples = torch.tensor(samples, dtype=torch.float32)
        self.labels = torch.from_numpy(np.vstack(labels).astype(float))
        self.samples = F.normalize(self.samples, dim=0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx], self.labels[idx]


class CustomDataset(BaseDataset):
    """Dataset for standard classification tasks."""

    pass


class SiameseDataset(BaseDataset):
    """Dataset for contrastive learning with all possible pairs."""
    def __init__(self, samples: np.ndarray, labels: np.ndarray):
        """Initialize Siamese dataset.
        Args:
            samples: Input features
            labels: Target labels
        """
        super().__init__(samples, labels)
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
                    
                    difference = X1 - X2
                    pair_label = torch.FloatTensor([int(torch.all(y1 == y2))])
                    
                    pairs.append(difference)
                    labels.append(pair_label)
        
        labels = np.asarray(labels, dtype=int)
        n_classes = len(np.unique(labels))
        return pairs, np.eye(n_classes)[labels].squeeze()


class DataAugmenter:
    """Handles data augmentation operations on DataLoader inputs."""

    def __init__(self, config: AugmentationConfig):
        """Initialize augmenter with configuration."""
        self.config = config

    def augment(self, dataloader: DataLoader) -> DataLoader:
        """Perform data augmentation on batched data from a DataLoader.

        Args:
            dataloader: Input DataLoader containing (features, labels) pairs

        Returns:
            New DataLoader with augmented data
        """
        if not self.config.enabled:
            return dataloader

        logger.info(
            f"Starting data augmentation with {self.config.num_augmentations} augmentations per sample"
        )

        # Extract all data from dataloader
        all_samples = []
        all_labels = []
        for samples, labels in tqdm(dataloader, desc="Collecting data from loader"):
            all_samples.append(samples.numpy())
            all_labels.append(labels.numpy())
        
        X = np.vstack(all_samples)
        y = np.vstack(all_labels)

        # Initialize arrays for augmented data
        n_samples = len(X)
        n_features = X.shape[1]
        total_samples = n_samples * (self.config.num_augmentations + 1)

        X_augmented = np.zeros((total_samples, n_features))
        y_augmented = np.zeros((total_samples,) + y.shape[1:])

        # Copy original data
        X_augmented[:n_samples] = X
        y_augmented[:n_samples] = y

        # Create augmented samples
        for i in tqdm(range(n_samples), desc="Augmenting data"):
            x, y_sample = X[i], y[i]

            for j in range(self.config.num_augmentations):
                idx = n_samples * (j + 1) + i
                augmented = x.copy()

                # Apply noise augmentation
                if self.config.noise_enabled:
                    noise = np.random.normal(
                        loc=0,
                        scale=self.config.noise_level * np.std(augmented),
                        size=augmented.shape,
                    )
                    augmented += noise

                # Apply shift augmentation
                if self.config.shift_enabled:
                    shift_amount = int(
                        len(augmented) 
                        * np.random.uniform(-self.config.shift_range, self.config.shift_range)
                    )
                    augmented = np.roll(augmented, shift_amount)

                # Apply scale augmentation
                if self.config.scale_enabled:
                    scale_factor = np.random.uniform(
                        1 - self.config.scale_range, 1 + self.config.scale_range
                    )
                    augmented *= scale_factor

                X_augmented[idx] = augmented
                y_augmented[idx] = y_sample

        # Verify augmentation results
        actual_samples = len(X_augmented)
        expected_samples = n_samples * (self.config.num_augmentations + 1)

        if actual_samples != expected_samples:
            logger.warning(
                f"Mismatch in augmented samples. Expected {expected_samples}, got {actual_samples}"
            )

        logger.info(
            f"Augmentation complete. Dataset size increased from {n_samples} "
            f"to {actual_samples} samples"
        )

        # Shuffle the augmented dataset
        shuffle_idx = np.random.permutation(len(X_augmented))
        X_augmented = X_augmented[shuffle_idx]
        y_augmented = y_augmented[shuffle_idx]

        # Create new dataset and dataloader with augmented data
        augmented_dataset = CustomDataset(X_augmented, y_augmented)
        augmented_dataloader = DataLoader(
            augmented_dataset,
            batch_size=dataloader.batch_size,
            shuffle=True,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory
        )

        return augmented_dataloader


class DataProcessor:
    """Handles data loading and preprocessing."""

    def __init__(
        self,
        dataset_type: Union[str, DatasetType],
        batch_size: int = 64,
        train_split: float = 0.8,
    ):
        """Initialize data processor.

        Args:
            dataset_type: Type of dataset to process
            augmentation_config: Configuration for data augmentation
            batch_size: Batch size for DataLoader
            train_split: Proportion of data to use for training
        """
        self.dataset_type = (
            DatasetType.from_string(dataset_type)
            if isinstance(dataset_type, str)
            else dataset_type
        )
        self.batch_size = batch_size
        self.train_split = train_split

    def load_data(self, file_path: Union[str, Path, List[str]]) -> pd.DataFrame:
        """Load data from file.

        Args:
            file_path: Path to data file. Can be string, Path, or list of path components

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If the data file doesn't exist
            ValueError: If the file format is not supported
        """
        try:
            # Convert path components to Path object
            if isinstance(file_path, (str, Path)):
                path = Path(file_path)
            else:
                path = Path(os.path.join(*file_path))

            logger.info(f"Loading data from: {path}")

            # Check if file exists
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {path}")

            # Load based on file extension
            if path.suffix.lower() == ".xlsx":
                data = pd.read_excel(path)
            elif path.suffix.lower() == ".csv":
                data = pd.read_csv(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

            logger.info(f"Loaded data with shape: {data.shape}")
            return data

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def filter_data(
        self, data: pd.DataFrame, is_pre_train: bool = False
    ) -> pd.DataFrame:
        """Filter dataset based on type and training mode.

        Args:
            data: Input DataFrame
            is_pre_train: Whether filtering is for pre-training

        Returns:
            Filtered DataFrame
        """
        if is_pre_train:
            return data

        # Remove quality control samples
        filtered = data[~data["m/z"].str.contains("QC")]

        # Dataset-specific filtering
        if self.dataset_type in [
            DatasetType.SPECIES,
            DatasetType.PART,
            DatasetType.OIL,
        ]:
            filtered = filtered[~filtered["m/z"].str.contains("HM")]

        if self.dataset_type in [
            DatasetType.SPECIES,
            DatasetType.PART,
            DatasetType.CROSS_SPECIES,
        ]:
            filtered = filtered[~filtered["m/z"].str.contains("MO")]

        if self.dataset_type in [
            DatasetType.PART,
        ]:
            filtered = filtered[
                filtered['m/z']
                .str
                .contains(
                    "fillet|frames|gonads|livers|skins|guts|frame|heads", 
                    case=False, 
                    na=False
                )
            ]

        if self.dataset_type in [
            DatasetType.OIL
        ]:
            filtered = filtered[filtered['m/z'].str.contains('MO', na=False)]

        if self.dataset_type in [
            DatasetType.INSTANCE_RECOGNITION,
            DatasetType.INSTANCE_RECOGNITION_HARD,
        ]:
            filtered = filtered[
                ~filtered.iloc[:, 0]
                .astype(str)
                .str.contains(
                    "QC|HM|MO|fillet|frames|gonads|livers|skins|guts|frame|heads",
                    case=False,
                    na=False,
                )
            ]

        if self.dataset_type in [
            DatasetType.CROSS_SPECIES_HARD
        ]:
            filtered = filtered[
                ~filtered.iloc[:, 0]
                .astype(str)
                .str.contains(
                    "^H |^M |QC|MO|fillet|frames|gonads|livers|skins|guts|frame|heads",
                    case=False,
                    na=False,
                )
            ]

        logger.info(f"Filtered data shape: {filtered.shape}")
        return filtered

    def encode_labels(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Encode labels based on dataset type.

        Args:
            data: Input DataFrame

        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        if self.dataset_type in [
            DatasetType.INSTANCE_RECOGNITION,
            DatasetType.INSTANCE_RECOGNITION_HARD,
            DatasetType.CROSS_SPECIES_HARD,
        ]:
            X = data.iloc[:, 1:].to_numpy()
            y = data.iloc[:, 0].to_numpy()
            le = LabelEncoder()
            y = le.fit_transform(y)
            n_classes = len(np.unique(y))
            return X, np.eye(n_classes)[y]

        # Convert labels first
        y_series = data["m/z"].apply(self._get_label_encoder())

        # Filter out None values
        valid_mask = y_series.notna()
        filtered_data = data[valid_mask]

        # Get features and labels
        X = filtered_data.drop("m/z", axis=1).to_numpy()
        y = np.array(y_series[valid_mask].tolist())

        return X, y

    def _get_label_encoder(self):
        """Get appropriate label encoding function based on dataset type."""
        if self.dataset_type == DatasetType.SPECIES:
            return lambda x: [0, 1] if "H" in x else [1, 0]

        elif self.dataset_type == DatasetType.PART:

            def encode_part(x):
                if "Fillet" in x:
                    return [1, 0, 0, 0, 0, 0, 0]
                if "Heads" in x:
                    return [0, 1, 0, 0, 0, 0, 0]
                if "Livers" in x:
                    return [0, 0, 1, 0, 0, 0, 0]
                if "Skins" in x:
                    return [0, 0, 0, 1, 0, 0, 0]
                if "Guts" in x:
                    return [0, 0, 0, 0, 1, 0, 0]
                if "Gonads" in x:
                    return [0, 0, 0, 0, 0, 1, 0]
                if "Frames" in x:
                    return [0, 0, 0, 0, 0, 0, 1]
                return None

            return encode_part

        elif self.dataset_type == DatasetType.OIL:

            def encode_oil(x):
                if "MO 50" in x:
                    return [1, 0, 0, 0, 0, 0, 0]
                if "MO 25" in x:
                    return [0, 1, 0, 0, 0, 0, 0]
                if "MO 10" in x:
                    return [0, 0, 1, 0, 0, 0, 0]
                if "MO 05" in x:
                    return [0, 0, 0, 1, 0, 0, 0]
                if "MO 01" in x:
                    return [0, 0, 0, 0, 1, 0, 0]
                if "MO 0.1" in x:
                    return [0, 0, 0, 0, 0, 1, 0]
                if "MO 0" in x:
                    return [0, 0, 0, 0, 0, 0, 1]
                return None

            return encode_oil

        elif self.dataset_type == DatasetType.OIL_SIMPLE:
            return lambda x: [1, 0] if "MO" in x else [0, 1]

        elif self.dataset_type == DatasetType.CROSS_SPECIES:

            def encode_cross_species(x):
                if "HM" in x:
                    return [1, 0, 0]
                if "H" in x:
                    return [0, 1, 0]
                if "M" in x:
                    return [0, 0, 1]
                return None

            return encode_cross_species

        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")


def preprocess_dataset(
    dataset: str = "species",
    batch_size: int = 64,
    is_pre_train: bool = False,
) -> Tuple[DataLoader, pd.DataFrame]:
    """Preprocess dataset for training or pre-training."""
    # Use provided config or create default
    processor = DataProcessor(dataset, batch_size)

    try:
        # Load and process data
        logger.info(f"Loading dataset: {dataset}")
        # file_path = "/vol/ecrg-solar/woodj4/fishy-business/data/REIMS.xlsx"
        file_path = "/home/woodj/Desktop/fishy-business/data/REIMS.xlsx"
        # file_path = "/vol/ecrg-solar/woodj4/fishy-business/data/REIMS_data.xlsx"
        data = processor.load_data(file_path)

        # Filter data based on pre-training flag
        filtered_data = processor.filter_data(data, is_pre_train)
        logger.info(f"Dataset shape after filtering: {filtered_data.shape}")

        # Encode labels and prepare features
        X, y = processor.encode_labels(filtered_data)
        logger.info(f"Features shape: {X.shape}, Labels shape: {y.shape}")

        original_size = len(X)

        # Create dataset instance
        dataset_class = (
            SiameseDataset if dataset == "instance-recognition" else CustomDataset
        )
        train_dataset = dataset_class(X, y)

        if isinstance(train_dataset, SiameseDataset):
            logger.info(
                f"Dataset size increased from {original_size} to {len(train_dataset.samples)} samples"
            )

        # Create DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        return train_loader, data

    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise

class DataModule:
    """High-level interface for data management."""

    def __init__(
        self,
        dataset_name: str,
        batch_size: int = 64,
        is_pre_train: bool = False,
    ):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.is_pre_train = is_pre_train
        self.processor = DataProcessor(dataset_name, batch_size)

    def setup(self) -> Tuple[DataLoader, pd.DataFrame]:
        """Set up data processing pipeline."""
        return preprocess_dataset(
            dataset=self.dataset_name,
            batch_size=self.batch_size,
            is_pre_train=self.is_pre_train,
        )

    @staticmethod
    def get_num_classes(dataset_name: str) -> int:
        """Get number of classes for a dataset.

        Args:
            dataset_name: Name of dataset

        Returns:
            Number of classes in dataset
        """
        class_counts = {
            "species": 2,
            "part": 7,
            "oil": 7,
            "oil_simple": 2,
            "cross-species": 3,
            "instance-recognition": 2,
            "instance-recognition-hard": 24,
        }
        if dataset_name not in class_counts:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return class_counts[dataset_name]


def create_data_module(
    dataset_name: str,
    batch_size: int = 64,
    is_pre_train: bool = False,
    **augmentation_kwargs,
) -> DataModule:
    """Factory function to create DataModule with configuration.

    Args:
        dataset_name: Name of dataset to process
        batch_size: Batch size for DataLoader
        augmentation_enabled: Whether to enable data augmentation
        is_pre_train: Whether this is for pre-training
        **augmentation_kwargs: Additional augmentation parameters

    Returns:
        Configured DataModule instance
    """


    return DataModule(
        dataset_name=dataset_name,
        batch_size=batch_size,
        is_pre_train=is_pre_train,
    )

# Usage example:
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Example usage with DataModule
    data_module = create_data_module(
        dataset_name="part",
        batch_size=64,
        augmentation_enabled=True,
        num_augmentations=3,
        noise_level=0.05,
    )

    # Get data loader and original data
    train_loader, raw_data = data_module.setup()

    # Print dataset information
    logger.info(f"Dataset loaded with {len(train_loader.dataset)} samples")
    logger.info(f"Number of classes: {data_module.get_num_classes('species')}")
