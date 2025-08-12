"""This module provides utilities for deep learning datasets,
including dataset types, data augmentation, and dataset classes.
"""

from dataclasses import dataclass, fields as dataclass_fields  # For AugmentationConfig
from enum import Enum, auto
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# train_test_split is imported but not used in the provided snippet
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Assuming transformer.py exists and Transformer is defined
# from transformer import Transformer

logger = logging.getLogger(__name__)


class DatasetType(Enum):
    """Enumeration for the different types of datasets.

    Attributes:
        SPECIES: Dataset for species classification.
        PART: Dataset for part classification.
        OIL: Dataset for oil classification.
        OIL_SIMPLE: Dataset for simple oil classification.
        OIL_REGRESSION: Dataset for oil regression.
        CROSS_SPECIES: Dataset for cross-species classification.
        CROSS_SPECIES_HARD: Dataset for hard cross-species classification.
        INSTANCE_RECOGNITION: Dataset for instance recognition.
        INSTANCE_RECOGNITION_HARD: Dataset for hard instance recognition.
    """

    SPECIES = auto()
    PART = auto()
    OIL = auto()
    OIL_SIMPLE = auto()
    OIL_REGRESSION = auto()  # Not used in filtering/encoding logic provided
    CROSS_SPECIES = auto()
    CROSS_SPECIES_HARD = auto()
    INSTANCE_RECOGNITION = auto()
    INSTANCE_RECOGNITION_HARD = auto()

    @classmethod
    def from_string(cls, name: str) -> "DatasetType":
        """Converts a string to a DatasetType enum member.

        Args:
            name: The name of the dataset type.

        Returns:
            The corresponding DatasetType enum member.

        Raises:
            ValueError: If the dataset name is invalid.
        """
        # Create a mapping from lowercase string to enum member
        # Handles names with hyphens by replacing them with underscores for enum member lookup
        normalized_name_map = {
            key.lower().replace("_", "-"): member
            for key, member in cls.__members__.items()
        }
        # Add direct aliases for the provided strings if they don't match normalized enum names
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
        # Prefer alias map, then normalized enum names
        target_name = name.lower()
        if target_name in alias_map:
            return alias_map[target_name]
        if (
            target_name in normalized_name_map
        ):  # e.g. if user typed "species_type" and an enum SPECIES_TYPE existed
            return normalized_name_map[target_name]

        raise ValueError(
            f"Invalid dataset name: {name}. Must be one of {list(alias_map.keys())}"
        )


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation.

    Attributes:
        enabled: Whether augmentation is enabled.
        num_augmentations: The number of augmented versions to create per sample.
        noise_enabled: Whether to add noise to the samples.
        shift_enabled: Whether to shift the samples.
        scale_enabled: Whether to scale the samples.
        noise_level: The level of noise to add.
        shift_range: The range for shifting the samples.
        scale_range: The range for scaling the samples.
    """

    enabled: bool = False
    num_augmentations: int = 5  # Number of *additional* augmented versions per sample
    noise_enabled: bool = True
    shift_enabled: bool = False
    scale_enabled: bool = False
    noise_level: float = 0.1  # Can be absolute or relative (see DataAugmenter)
    shift_range: float = 0.1  # Proportion of total length
    scale_range: float = 0.1  # Range around 1.0 (e.g., 0.1 means 0.9 to 1.1)


class BaseDataset(Dataset):
    def __init__(self, samples: np.ndarray, labels: np.ndarray) -> None:
        """Initializes the dataset with samples and labels.

        Args:
            samples (np.ndarray): Array of shape (num_samples, num_features) containing the features.
            labels (np.ndarray): Array of shape (num_samples, num_classes) or (num_samples,) containing the labels.

        Raises:
            ValueError: If samples or labels are empty or have incompatible shapes.
        """
        self.samples = torch.tensor(samples, dtype=torch.float32)
        self.labels = torch.tensor(
            np.array(labels), dtype=torch.float32
        )  # Ensure labels are also float32

        # Normalize features (dim=0 normalizes each feature across samples)
        # Ensure there's more than one sample to avoid NaN with std=0 if only one sample.
        if self.samples.ndim > 1 and self.samples.shape[0] > 1:
            self.samples = F.normalize(self.samples, p=2, dim=0)
        elif self.samples.ndim == 1 and self.samples.numel() > 0:  # Single sample
            self.samples = F.normalize(self.samples, p=2, dim=0)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return self.samples.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieves a sample and its corresponding label by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the sample and its label.
        """
        return self.samples[idx], self.labels[idx]


class CustomDataset(BaseDataset):
    """A standard PyTorch Dataset that inherits from BaseDataset.

    This class serves as a simple wrapper around `BaseDataset` and does not introduce
    any additional functionality or modifications. It is intended for use cases
    where a basic dataset structure is sufficient without specialized behaviors
    like pair generation for contrastive learning.
    """

    pass


class SiameseDataset(BaseDataset):
    """Dataset for contrastive learning, generating pairs of samples.

    This dataset extends `BaseDataset` to create pairs of samples and their
    corresponding labels, indicating whether the samples in a pair belong to
    the same class or different classes. It is particularly useful for
    Siamese networks and other contrastive learning approaches.
    """

    def __init__(self, samples: np.ndarray, labels: np.ndarray) -> None:
        """Initializes the Siamese dataset with samples and labels.

        Args:
            samples (np.ndarray): Array of shape (num_samples, num_features) containing the features.
            labels (np.ndarray): Array of shape (num_samples, num_classes) or (num_samples,) containing the labels.

        Raises:
            ValueError: If samples or labels are empty or have incompatible shapes.
        """
        # Initialize with original data to allow BaseDataset to convert them to tensors
        super().__init__(samples, labels)
        # Now self.samples and self.labels are tensors. Generate pairs from these.
        self.paired_samples, self.paired_labels = self._generate_pairs_vectorized(
            self.samples, self.labels
        )

    def _generate_pairs_vectorized(
        self, original_samples: torch.Tensor, original_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates differing pairs for contrastive learning in a vectorized way.

        This method creates all possible unique pairs from the input samples and
        determines if each pair consists of samples from the same class or different
        classes. The output `paired_samples_tensor` contains the element-wise
        difference between the paired samples, and `pair_labels_tensor` indicates
        the similarity (1.0 for same class, 0.0 for different class).

        Args:
            original_samples (torch.Tensor): Original samples tensor of shape (num_samples, num_features).
            original_labels (torch.Tensor): Original labels tensor of shape (num_samples, num_classes) or (num_samples,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - paired_samples_tensor: Tensor of shape (num_pairs, num_features) with element-wise differences.
                - pair_labels_tensor: Tensor of shape (num_pairs, 1) with labels indicating if pairs are from the same class (1.0) or different (0.0).
        """
        n_samples = original_samples.shape[0]
        if n_samples < 2:
            return torch.empty(
                (0, original_samples.shape[1]), dtype=original_samples.dtype
            ), torch.empty((0, 1), dtype=torch.float32)

        # Create all possible (i, j) indices where i != j
        indices_i = torch.arange(n_samples).unsqueeze(1).expand(-1, n_samples).flatten()
        indices_j = torch.arange(n_samples).unsqueeze(0).expand(n_samples, -1).flatten()

        mask = indices_i != indices_j
        indices_i, indices_j = indices_i[mask], indices_j[mask]

        X1, X2 = original_samples[indices_i], original_samples[indices_j]
        y1, y2 = original_labels[indices_i], original_labels[indices_j]

        # Input for the model: element-wise difference (can be changed to concatenation or other)
        # paired_samples_tensor = torch.abs(X1 - X2) # Or just X1-X2, or torch.cat((X1,X2), dim=1)
        paired_samples_tensor = X1 - X2

        # Determine if labels are identical. Handles multi-class (one-hot) or single-class index labels.
        if y1.ndim > 1 and y1.shape[1] > 1:  # Assumed one-hot encoded
            same_label_mask = torch.all(y1 == y2, dim=1)
        else:  # Assumed class indices
            same_label_mask = y1.squeeze() == y2.squeeze()

        # Labels for pairs: 1.0 if same class, 0.0 if different
        # Shape: (num_pairs, 1) for compatibility with BCEWithLogitsLoss
        pair_labels_tensor = same_label_mask.to(torch.float32).unsqueeze(1)

        return paired_samples_tensor, pair_labels_tensor

    def __len__(self) -> int:
        """Returns the number of pairs in the dataset.

        Returns:
            int: Number of pairs in the dataset.
        """
        return self.paired_samples.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieves a pair of samples and their corresponding label by index.

        Args:
            idx (int): Index of the pair to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the paired sample and its label.
        """
        return self.paired_samples[idx], self.paired_labels[idx]


class DataAugmenter:
    """Applies various data augmentation techniques to a DataLoader.

    This class provides methods to augment data by adding noise, shifting, and scaling
    samples based on a provided `AugmentationConfig`. It operates by extracting
    data from an existing DataLoader, applying augmentations, and then creating
    a new DataLoader with the augmented data.

    NOTE: The current approach of extracting all data from a DataLoader,
    augmenting in NumPy, then creating a new DataLoader is highly inefficient
    and not recommended for large datasets or performance-critical applications.
    Augmentation is typically done on-the-fly within the Dataset.__getitem__ method.
    This refactoring makes the existing logic more concise but doesn't change the approach.
    """

    def __init__(self, config: AugmentationConfig) -> None:
        """Initializes the DataAugmenter with a configuration.

        Args:
            config: An `AugmentationConfig` object specifying the augmentation parameters.
        """
        self.config = config

    def _apply_augmentations_to_batch(self, X_batch: np.ndarray) -> np.ndarray:
        """Applies configured augmentations to a batch of samples.

        Args:
            X_batch: A NumPy array representing a batch of samples to be augmented.

        Returns:
            A NumPy array containing the augmented batch of samples.
        """
        X_augmented_batch = X_batch.copy()  # Augment from fresh copies
        n_samples, n_features = X_augmented_batch.shape

        if self.config.noise_enabled:
            # Assuming noise_level is a fraction of std dev of each feature if not absolute
            # For simplicity, using a global noise level relative to overall data std or a fixed value
            # noise_std = self.config.noise_level * np.std(X_batch) # Global std based noise
            # Or interpret noise_level as an absolute standard deviation
            noise = np.random.normal(
                loc=0, scale=self.config.noise_level, size=X_augmented_batch.shape
            )
            X_augmented_batch += noise

        if self.config.shift_enabled:
            for k in range(n_samples):  # Shift must be per-sample
                shift_amount = int(
                    n_features
                    * np.random.uniform(
                        -self.config.shift_range, self.config.shift_range
                    )
                )
                if n_features > 0:  # Avoid error on empty features
                    X_augmented_batch[k] = np.roll(X_augmented_batch[k], shift_amount)

        if self.config.scale_enabled:
            for k in range(n_samples):  # Scale per-sample
                scale_factor = np.random.uniform(
                    1 - self.config.scale_range, 1 + self.config.scale_range
                )
                X_augmented_batch[k] *= scale_factor
        return X_augmented_batch

    def augment(self, dataloader: DataLoader) -> DataLoader:
        """Applies data augmentation to the samples in the provided DataLoader.

        If augmentation is enabled in the configuration, this method extracts all
        samples from the input DataLoader, applies the specified augmentations
        (`num_augmentations` times), concatenates the original and augmented data,
        shuffles the combined dataset, and returns a new DataLoader.

        Args:
            dataloader: The original `DataLoader` containing the data to be augmented.

        Returns:
            A new `DataLoader` containing the original and augmented samples.
            If augmentation is disabled or `num_augmentations` is 0, the original
            `DataLoader` is returned unchanged.
        """


class DataProcessor:
    def __init__(
        self, dataset_type: DatasetType, batch_size: int = 64
    ) -> None:  # Removed train_split as it's not used
        """Initializes the DataProcessor with dataset type and batch size.

        Args:
            dataset_type (DatasetType): The type of dataset to process.
            batch_size (int): The batch size for DataLoader. Defaults to 64.
        Raises:
            ValueError: If the dataset type is not recognized.
        """
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.label_encoder_ = None  # To store fitted LabelEncoder if used

        # --- Configuration for filtering and label encoding ---
        self._PART_CATEGORIES = [
            "Fillet",
            "Heads",
            "Livers",
            "Skins",
            "Guts",
            "Gonads",
            "Frames",
        ]
        self._OIL_CATEGORIES = [
            "MO 50",
            "MO 25",
            "MO 10",
            "MO 05",
            "MO 01",
            "MO 0.1",
            "MO 0",
        ]  # "MO 0" should be specific enough

        self._FILTER_RULES = {
            # Common exclusions from 'm/z' based on substring
            (DatasetType.SPECIES, DatasetType.PART, DatasetType.OIL): {
                "exclude_mz": ["HM"]
            },
            (DatasetType.SPECIES, DatasetType.PART, DatasetType.CROSS_SPECIES): {
                "exclude_mz": ["MO"]
            },
            # Specific inclusions/exclusions (can be combined)
            DatasetType.PART: {"include_mz_pattern": "|".join(self._PART_CATEGORIES)},
            DatasetType.OIL: {
                "include_mz_pattern": "MO"
            },  # Ensure 'MO' (oil) samples are kept
            # Exclusions from the first column (instance name)
            (DatasetType.INSTANCE_RECOGNITION, DatasetType.INSTANCE_RECOGNITION_HARD): {
                "exclude_instance_pattern": f"QC|HM|MO|{'|'.join(self._PART_CATEGORIES)}"
            },
            DatasetType.CROSS_SPECIES_HARD: {
                "exclude_instance_pattern": f"^H |^M |QC|HM|MO|{'|'.join(self._PART_CATEGORIES)}"
            },
        }
        self._LABEL_ENCODERS_MAP = {
            DatasetType.SPECIES: lambda x: (
                [0.0, 1.0] if "H" in x else ([1.0, 0.0] if "M" in x else None)
            ),
            DatasetType.PART: self._create_one_hot_encoder(self._PART_CATEGORIES),
            DatasetType.OIL: self._create_one_hot_encoder(self._OIL_CATEGORIES),
            DatasetType.OIL_SIMPLE: lambda x: (
                [1.0, 0.0] if "MO" in x else ([0.0, 1.0] if x.strip() else None)
            ),  # Crude check for non-MO
            DatasetType.CROSS_SPECIES: lambda x: (
                [1.0, 0.0, 0.0]
                if "HM" in x
                else (
                    [0.0, 1.0, 0.0]
                    if "H" in x
                    else ([0.0, 0.0, 1.0] if "M" in x else None)
                )
            ),
            DatasetType.INSTANCE_RECOGNITION: "use_sklearn_label_encoder",
            DatasetType.INSTANCE_RECOGNITION_HARD: "use_sklearn_label_encoder",
            DatasetType.CROSS_SPECIES_HARD: "use_sklearn_label_encoder",
        }

    def _create_one_hot_encoder(
        self, categories: List[str]
    ) -> Callable[[str], Optional[List[float]]]:
        """Creates a one-hot encoder function for the given categories.

        Args:
            categories (List[str]): List of category names to encode.

        Returns:
            Callable[[str], Optional[List[float]]]: A function that takes a string and returns a one-hot encoded list
        """
        cat_to_idx = {cat.lower(): i for i, cat in enumerate(categories)}
        num_cat = len(categories)

        def encoder(x_str: str) -> Optional[List[float]]:
            x_str_lower = x_str.lower()
            for cat_name_lower, idx in cat_to_idx.items():
                if cat_name_lower in x_str_lower:
                    one_hot = [0.0] * num_cat
                    one_hot[idx] = 1.0
                    return one_hot
            return None

        return encoder

    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Loads data from a file into a pandas DataFrame.

        Args:
            file_path (Union[str, Path]): Path to the data file (CSV or Excel).

        Returns:
            pd.DataFrame: DataFrame containing the loaded data.
        """
        path = Path(file_path)
        logger.info(f"Loading data from: {path}")
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        if path.suffix.lower() == ".xlsx":
            data = pd.read_excel(path)
        elif path.suffix.lower() == ".csv":
            data = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        logger.info(f"Loaded data with shape: {data.shape}")
        return data

    def filter_data(
        self, data: pd.DataFrame, is_pre_train: bool = False
    ) -> pd.DataFrame:
        """Filters the DataFrame based on dataset type and pre-training status.

        Args:
            data (pd.DataFrame): The DataFrame to filter.
            is_pre_train (bool): Whether the data is for pre-training. If True, no filtering is applied.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        if is_pre_train:
            return data

        df = data.copy()
        # Always exclude QC samples from 'm/z' or the first column (more robust)
        # Assuming 'm/z' is the primary identifier column for these QC tags
        qc_pattern = "QC"
        df = df[~df["m/z"].astype(str).str.contains(qc_pattern, case=False, na=False)]
        # Also check first column if 'm/z' might not be it, or if QC pattern applies elsewhere
        if data.columns[0] != "m/z":
            df = df[
                ~df.iloc[:, 0]
                .astype(str)
                .str.contains(qc_pattern, case=False, na=False)
            ]

        for key_tuple, rules in self._FILTER_RULES.items():
            dataset_types_in_rule = (
                key_tuple if isinstance(key_tuple, tuple) else (key_tuple,)
            )
            if self.dataset_type in dataset_types_in_rule:
                if "exclude_mz" in rules:
                    for pattern in rules["exclude_mz"]:
                        df = df[
                            ~df["m/z"]
                            .astype(str)
                            .str.contains(pattern, case=False, na=False)
                        ]
                if "include_mz_pattern" in rules:
                    df = df[
                        df["m/z"]
                        .astype(str)
                        .str.contains(rules["include_mz_pattern"], case=False, na=False)
                    ]
                if (
                    "exclude_instance_pattern" in rules
                ):  # Assumes first column is instance identifier
                    df = df[
                        ~df.iloc[:, 0]
                        .astype(str)
                        .str.contains(
                            rules["exclude_instance_pattern"], case=False, na=False
                        )
                    ]

        logger.info(f"Filtered data shape: {df.shape}")
        return df

    def encode_labels(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Encodes labels based on dataset type and returns features and labels.

        Args:
            data (pd.DataFrame): The DataFrame containing the data to encode.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - X (np.ndarray): Features array of shape (num_samples, num_features).
                - y (np.ndarray): Labels array of shape (num_samples, num_classes).
        """
        label_action = self._LABEL_ENCODERS_MAP.get(self.dataset_type)

        if data.empty:
            logger.warning(
                f"Cannot encode labels for empty DataFrame (dataset type: {self.dataset_type.name})."
            )
            # Infer feature dimension if possible, otherwise a generic empty array
            num_features = (
                data.shape[1] - 1 if data.shape[1] > 1 and "m/z" in data.columns else 0
            )
            num_features = (
                data.shape[1]
                if data.shape[1] > 0
                and "m/z" not in data.columns
                and label_action == "use_sklearn_label_encoder"
                else num_features
            )
            return np.empty((0, num_features)), np.empty((0, 0))

        if label_action == "use_sklearn_label_encoder":
            # Assumes labels are in the first column, features are the rest
            X = data.iloc[:, 1:].to_numpy(dtype=np.float32)
            y_raw = (
                data.iloc[:, 0].astype(str).to_numpy()
            )  # Ensure string type for LabelEncoder

            self.label_encoder_ = (
                LabelEncoder()
            )  # Store for potential inverse transform or class inspection
            y_indices = self.label_encoder_.fit_transform(y_raw)
            y = np.eye(len(self.label_encoder_.classes_), dtype=np.float32)[y_indices]
        elif callable(label_action):
            # Assumes labels derived from 'm/z', features are other columns
            if "m/z" not in data.columns:
                raise ValueError(
                    "Column 'm/z' not found for label encoding when expected."
                )
            y_series = data["m/z"].astype(str).apply(label_action)
            valid_mask = y_series.notna()

            if not valid_mask.any():
                logger.warning(
                    f"No valid labels produced by encoder for {self.dataset_type.name} from 'm/z' column."
                )
                return data.drop("m/z", axis=1, errors="ignore").to_numpy(
                    dtype=np.float32
                ), np.empty((0, 0))

            X = data[valid_mask].drop("m/z", axis=1).to_numpy(dtype=np.float32)
            y = np.array(y_series[valid_mask].tolist(), dtype=np.float32)
        else:
            raise ValueError(
                f"No label encoding action defined for dataset type: {self.dataset_type.name}"
            )

        if y.ndim == 1:  # Ensure labels are at least 2D (N, num_classes)
            y = y[:, np.newaxis]

        return X, y

    def extract_groups(self, data: pd.DataFrame) -> np.ndarray:
        """Extracts group labels from the sample names in the 'm/z' column."""
        # Assumes group is the part of the name before the first '_'
        sample_names = data['m/z'].astype(str)
        groups = sample_names.str.split('_').str[0]
        return groups.to_numpy()


def preprocess_data_pipeline(  # Renamed from preprocess_dataset to avoid conflict with torch.utils.data.Dataset
    data_processor: DataProcessor,
    file_path: Union[str, Path],
    is_pre_train: bool = False,
    augmentation_cfg: Optional[AugmentationConfig] = None,
) -> Tuple[DataLoader, pd.DataFrame, pd.DataFrame]:
    """Preprocesses data and returns a DataLoader and raw DataFrame.

    Args:
        data_processor (DataProcessor): The DataProcessor instance to use for loading and processing data.
        file_path (Union[str, Path]): Path to the data file (CSV or Excel).
        is_pre_train (bool): Whether the data is for pre-training. If True, no filtering is applied.
        augmentation_cfg (Optional[AugmentationConfig]): Configuration for data augmentation. If None, no augmentation is applied.

    Returns:
        Tuple[DataLoader, pd.DataFrame, pd.DataFrame]: A tuple containing:
            - DataLoader: The DataLoader containing the preprocessed data.
            - pd.DataFrame: The raw DataFrame containing the loaded data before filtering.
            - pd.DataFrame: The filtered DataFrame used for processing.
    """

    raw_df = data_processor.load_data(file_path)
    filtered_df = data_processor.filter_data(raw_df, is_pre_train)

    if filtered_df.empty:
        logger.error(
            f"Dataframe is empty after filtering for {data_processor.dataset_type.name}. No DataLoader created."
        )
        # Return an empty DataLoader and the raw (unfiltered) DataFrame
        empty_torch_dataset = CustomDataset(np.array([]), np.array([]))
        return (
            DataLoader(empty_torch_dataset, batch_size=data_processor.batch_size),
            raw_df,
            filtered_df,
        )

    X, y = data_processor.encode_labels(filtered_df)

    if X.size == 0:  # Could happen if all labels were None or invalid
        logger.error(
            f"No features remain after label encoding for {data_processor.dataset_type.name}. No DataLoader created."
        )
        empty_torch_dataset = CustomDataset(
            np.array([]), np.array([])
        )  # X and y must be np.ndarray
        return (
            DataLoader(empty_torch_dataset, batch_size=data_processor.batch_size),
            raw_df,
            filtered_df,
        )

    # Determine dataset class (e.g. Siamese for instance recognition)
    dataset_name_str = data_processor.dataset_type.name.lower().replace("_", "-")
    dataset_class = (
        SiameseDataset if "instance-recognition" in dataset_name_str else CustomDataset
    )

    torch_dataset = dataset_class(X, y)

    data_loader = DataLoader(
        torch_dataset,
        batch_size=data_processor.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,  # num_workers=0 for simplicity, can be configured
    )

    if augmentation_cfg and augmentation_cfg.enabled:
        augmenter = DataAugmenter(augmentation_cfg)
        data_loader = augmenter.augment(data_loader)  # Returns a new DataLoader

    return data_loader, raw_df, filtered_df


class DataModule:
    """High-level interface for data management."""

    def __init__(
        self,
        dataset_name: str,  # String name, e.g., "species"
        file_path: Union[str, Path],
        batch_size: int = 64,
        is_pre_train: bool = False,
        augmentation_config: Optional[AugmentationConfig] = None,
    ) -> None:
        """Initializes the DataModule with dataset name, file path, and configuration.

        Args:
            dataset_name (str): Name of the dataset (e.g., "species", "part").
            file_path (Union[str, Path]): Path to the data file (CSV or Excel).
            batch_size (int): Batch size for DataLoader. Defaults to 64.
            is_pre_train (bool): Whether the data is for pre-training. If True, no filtering is applied.
            augmentation_config (Optional[AugmentationConfig]): Configuration for data augmentation. If None, no augmentation is applied.

        Raises:
            ValueError: If the dataset name is invalid or not recognized.
            TypeError: If the file path is not a string or Path object.
        """
        self.dataset_name_str = (
            dataset_name  # Store original string for convenience (e.g. get_num_classes)
        )
        self.file_path = file_path
        self.batch_size = batch_size
        self.is_pre_train = is_pre_train
        self.augmentation_config = augmentation_config

        dataset_type_enum = DatasetType.from_string(dataset_name)
        self.processor = DataProcessor(dataset_type_enum, batch_size)
        self.train_loader: Optional[DataLoader] = None
        self.raw_data: Optional[pd.DataFrame] = None
        self.filtered_data: Optional[pd.DataFrame] = None

    def setup(self) -> None:  # Changed to not return, but set attributes
        """Loads and preprocesses data, setting up DataLoaders."""
        self.train_loader, self.raw_data, self.filtered_data = preprocess_data_pipeline(
            data_processor=self.processor,
            file_path=self.file_path,
            is_pre_train=self.is_pre_train,
            augmentation_cfg=self.augmentation_config,
        )

    def get_dataset(self) -> Dataset:
        """Returns the dataset used by the DataLoader.

        Returns:
            Dataset: The dataset used by the train DataLoader, or an empty CustomDataset if not set.
        """
        if self.train_loader is None:
            logger.warning("Train DataLoader not set up. Call setup() first.")
            return CustomDataset(np.array([]), np.array([]))
        return self.train_loader.dataset

    def get_train_dataframe(self) -> pd.DataFrame:
        """Returns the raw DataFrame used to create the DataLoader.

        Returns:
            pd.DataFrame: The raw DataFrame containing the loaded data before filtering.
        """
        if self.raw_data is None:
            logger.warning("Raw DataFrame not set up. Call setup() first.")
            return pd.DataFrame()
        return self.raw_data

    def get_train_dataloader(self) -> DataLoader:
        """Returns the DataLoader for training data.

        Raises:
            Warning: If the DataLoader is not set up, a warning is logged.

        Returns:
            DataLoader: The DataLoader for training data, or an empty DataLoader if not set up.
        """
        if self.train_loader is None:
            logger.warning("Train DataLoader not set up. Call setup() first.")
            # Return an empty loader to prevent None errors downstream
            empty_dataset = CustomDataset(np.array([]), np.array([]))
            return DataLoader(empty_dataset, batch_size=self.batch_size)
        return self.train_loader

    def get_input_dim(self) -> int:
        """Returns the input dimension of the dataset."""
        if self.train_loader is None:
            self.setup()
        if self.train_loader is not None and len(self.train_loader.dataset) > 0:
            return self.train_loader.dataset.samples.shape[1]
        return 0

    @staticmethod
    def get_num_output_features(
        dataset_name_str: str, data_processor: Optional[DataProcessor] = None
    ) -> int:
        """
        Get number of output features (classes) for a dataset.
        For 'use_sklearn_label_encoder' types, it's dynamic if a processor is available.

        Args:
            dataset_name_str (str): Name of the dataset (e.g., "species", "part").
            data_processor (Optional[DataProcessor]): DataProcessor instance if available, to check fitted label encoder.

        Returns:
            int: Number of output features (classes) for the dataset.
        """
        dt = DatasetType.from_string(dataset_name_str)
        # This static map defines the *expected* output dimension for models.
        # It might not always match the number of unique raw labels for 'use_sklearn_label_encoder'
        # if the model is fixed for a certain number of classes (e.g. Siamese binary output).
        static_class_counts = {
            DatasetType.SPECIES: 2,
            DatasetType.PART: 7,
            DatasetType.OIL: 7,
            DatasetType.OIL_SIMPLE: 2,
            DatasetType.CROSS_SPECIES: 3,
            DatasetType.INSTANCE_RECOGNITION: 1,  # Binary output (0 or 1) for Siamese pairs (usually for BCEWithLogitsLoss)
            # The original Siamese created (N,1) labels
        }
        if dt in static_class_counts:
            return static_class_counts[dt]

        # For types that use sklearn.preprocessing.LabelEncoder
        if (
            data_processor
            and data_processor.label_encoder_
            and dt
            in {DatasetType.INSTANCE_RECOGNITION_HARD, DatasetType.CROSS_SPECIES_HARD}
        ):
            return len(data_processor.label_encoder_.classes_)

        # Fallback/default for unmapped or dynamic types where processor isn't available/fitted
        # These were hardcoded to 24 in the original, which is risky if data changes.
        # It's better to raise an error or have a clear policy.
        if dt == DatasetType.INSTANCE_RECOGNITION_HARD:
            logger.warning(
                f"Num classes for {dt.name} is dynamic. Using fallback 24. Fit DataProcessor or provide static map."
            )
            return 24
        if dt == DatasetType.CROSS_SPECIES_HARD:
            logger.warning(
                f"Num classes for {dt.name} is dynamic. Using fallback 24. Fit DataProcessor or provide static map."
            )
            return 24

        raise ValueError(
            f"Number of output features for dataset '{dataset_name_str}' is not defined or determinable."
        )


def create_data_module(
    dataset_name: str,
    file_path: Union[str, Path],
    batch_size: int = 64,
    is_pre_train: bool = False,
    augmentation_enabled: bool = False,
    **kwargs_for_augmentation,  # Catch all other kwargs
) -> DataModule:
    """Creates a DataModule instance for the specified dataset.

    Args:
        dataset_name (str): Name of the dataset (e.g., "species", "part").
        file_path (Union[str, Path]): Path to the data file (CSV or Excel).
        batch_size (int): Batch size for DataLoader. Defaults to 64.
        is_pre_train (bool): Whether the data is for pre-training. If True, no filtering is applied.
        augmentation_enabled (bool): Whether to enable data augmentation.
        **kwargs_for_augmentation: Additional keyword arguments for AugmentationConfig.

    Returns:
        DataModule: An instance of DataModule configured for the specified dataset.
    Raises:
        ValueError: If the dataset name is invalid or not recognized.
        TypeError: If the file path is not a string or Path object.
    """
    aug_config = None
    if augmentation_enabled:
        # Filter kwargs to only those valid for AugmentationConfig
        valid_aug_keys = {f.name for f in dataclass_fields(AugmentationConfig)}
        actual_aug_kwargs = {
            k: v for k, v in kwargs_for_augmentation.items() if k in valid_aug_keys
        }
        aug_config = AugmentationConfig(enabled=True, **actual_aug_kwargs)

    return DataModule(
        dataset_name=dataset_name,
        file_path=file_path,
        batch_size=batch_size,
        is_pre_train=is_pre_train,
        augmentation_config=aug_config,
    )


# Usage example:
if __name__ == "__main__":
    """Main entry point for testing the DataModule and DataProcessor functionality."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Define file path (ensure this path is correct for your system)
    # data_file = "/vol/ecrg-solar/woodj4/fishy-business/data/REIMS.xlsx" # Example server path
    data_file = Path(
        "./REIMS.xlsx"
    )  # Local example - create a dummy if needed for testing
    if not data_file.exists():
        logger.warning(
            f"Data file {data_file} not found. Creating a dummy CSV for demonstration."
        )
        # Create a dummy CSV file for testing purposes
        dummy_data = {
            "m/z": [
                f"Sample{i} H" if i % 2 == 0 else f"Sample{i} M PartFillet"
                for i in range(20)
            ],
            **{f"feature_{j}": np.random.rand(20) for j in range(10)},
        }
        pd.DataFrame(dummy_data).to_csv(data_file, index=False)

    data_module = create_data_module(
        dataset_name="part",  # Try "species", "part", "instance-recognition"
        file_path=data_file,  # Pass the actual file path
        batch_size=4,  # Smaller batch for testing
        augmentation_enabled=True,
        num_augmentations=1,  # Fewer augmentations for faster testing
        noise_level=0.02,
        shift_enabled=True,  # Test shift
        scale_enabled=True,  # Test scale
    )

    data_module.setup()  # Load and process data

    train_loader = data_module.get_train_dataloader()

    if train_loader and len(train_loader.dataset) > 0:
        logger.info(
            f"Dataset '{data_module.dataset_name_str}' loaded with {len(train_loader.dataset)} samples in DataLoader."
        )

        # Inspect a batch
        try:
            samples, labels = next(iter(train_loader))
            logger.info(
                f"  Sample batch - Samples shape: {samples.shape}, Labels shape: {labels.shape}"
            )
            logger.info(f"  Sample Pytorch tensor dtype: {samples.dtype}")
            logger.info(f"  Labels Pytorch tensor dtype: {labels.dtype}")

        except StopIteration:
            logger.warning("DataLoader is empty, cannot fetch a batch.")

        num_classes = DataModule.get_num_output_features(
            data_module.dataset_name_str, data_module.processor
        )
        logger.info(
            f"Number of output features for '{data_module.dataset_name_str}': {num_classes}"
        )

    else:
        logger.error(
            f"Failed to load data for {data_module.dataset_name_str}. DataLoader is empty."
        )

    # Example for instance-recognition
    logger.info("\n --- Testing instance-recognition ---")
    data_module_siamese = create_data_module(
        dataset_name="instance-recognition", file_path=data_file, batch_size=4
    )
    data_module_siamese.setup()
    siamese_loader = data_module_siamese.get_train_dataloader()
    if siamese_loader and len(siamese_loader.dataset) > 0:
        logger.info(
            f"Dataset 'instance-recognition' loaded with {len(siamese_loader.dataset)} PAIRS in DataLoader."
        )
        samples, labels = next(iter(siamese_loader))
        logger.info(
            f"  Sample batch - Paired Samples shape: {samples.shape}, Pair Labels shape: {labels.shape}"
        )
        num_classes = DataModule.get_num_output_features(
            data_module_siamese.dataset_name_str, data_module_siamese.processor
        )
        logger.info(
            f"Number of output features for 'instance-recognition': {num_classes}"
        )  # Should be 1 for BCE
    else:
        logger.error(
            f"Failed to load data for instance-recognition. DataLoader is empty."
        )
