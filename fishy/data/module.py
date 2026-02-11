# -*- coding: utf-8 -*-
"""
Data module for managing loading, filtering, and preprocessing of datasets.

This module provides the `DataModule` class, which serves as the high-level interface
for all data-related operations in the training pipeline. It handles:
- Loading raw data from files (Excel/CSV).
- applying dataset-specific filtering rules defined in configuration.
- Encoding labels (One-Hot, Ordinal, etc.).
- Creating PyTorch DataLoaders.
- Applying data augmentation.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

from .datasets import DatasetType, CustomDataset, SiameseDataset
from .augmentation import AugmentationConfig, DataAugmenter
from fishy._core.config_loader import load_config

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Handles the low-level processing of raw data into features and labels.

    Examples:
        >>> from fishy.data.datasets import DatasetType
        >>> processor = DataProcessor(DatasetType.SPECIES)
        >>> processor.dataset_type.name
        'SPECIES'

    This class interprets the dataset configuration to apply specific filtering
    and label encoding logic.

    Attributes:
        dataset_type (DatasetType): The type/name of the dataset.
        batch_size (int): The batch size for DataLoaders.
        label_encoder_ (Optional[LabelEncoder]): scikit-learn encoder instance (if used).
        config (Dict): The configuration dictionary for this specific dataset.
    """

    def __init__(self, dataset_type: DatasetType, batch_size: int = 64) -> None:
        """
        Initializes the DataProcessor.

        Args:
            dataset_type (DatasetType): The type/enum of the dataset.
            batch_size (int, optional): Batch size for processing. Defaults to 64.
        """
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.label_encoder_ = None

        # Load dataset configurations
        all_configs = load_config("datasets")
        dataset_name = dataset_type.name.lower().replace("_", "-")
        self.config = all_configs.get(dataset_name, {})

        # Fallback for specific naming variants
        if not self.config and dataset_name == "oil-regression":
            self.config = all_configs.get("oil_regression", {})
        if not self.config and dataset_name == "oil-simple":
            self.config = all_configs.get("oil_simple", {})

    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Loads data from an Excel or CSV file.

        Args:
            file_path (Union[str, Path]): Path to the data file.

        Returns:
            pd.DataFrame: The loaded raw data.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        return (
            pd.read_excel(path) if path.suffix.lower() == ".xlsx" else pd.read_csv(path)
        )

    def filter_data(
        self, data: pd.DataFrame, is_pre_train: bool = False
    ) -> pd.DataFrame:
        """
        Applies dataset-specific filtering rules to the raw DataFrame.

        Filters rows based on 'm/z' patterns or instance identifiers as defined
        in the dataset configuration.

        Args:
            data (pd.DataFrame): The raw input DataFrame.
            is_pre_train (bool, optional): If True, skips filtering. Defaults to False.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        if is_pre_train:
            return data
        df = data.copy()
        df = df[~df["m/z"].astype(str).str.contains("QC", case=False, na=False)]

        rules = self.config.get("filter_rules", {})
        if "exclude_mz" in rules:
            for p in rules["exclude_mz"]:
                df = df[~df["m/z"].astype(str).str.contains(p, case=False, na=False)]
        if "include_mz_pattern" in rules:
            df = df[
                df["m/z"]
                .astype(str)
                .str.contains(rules["include_mz_pattern"], case=False, na=False)
            ]
        if "exclude_instance_pattern" in rules:
            df = df[
                ~df.iloc[:, 0]
                .astype(str)
                .str.contains(rules["exclude_instance_pattern"], case=False, na=False)
            ]

        return df

    def encode_labels(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encodes labels based on the configuration (e.g., sklearn, one_hot, map).

        Args:
            data (pd.DataFrame): The filtered DataFrame.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - X (np.ndarray): The feature matrix (float32).
                - y (np.ndarray): The target array (float32).
        """
        encoding_cfg = self.config.get("label_encoding", {})
        enc_type = encoding_cfg.get("type")

        if data.empty:
            return np.empty((0, 0)), np.empty((0, 0))

        if enc_type == "sklearn":
            X = data.iloc[:, 1:].to_numpy(dtype=np.float32)
            self.label_encoder_ = LabelEncoder()
            y_indices = self.label_encoder_.fit_transform(data.iloc[:, 0].astype(str))
            y = np.eye(len(self.label_encoder_.classes_), dtype=np.float32)[y_indices]

        elif enc_type == "one_hot":
            categories = self.config.get("categories", [])
            cat_to_idx = {cat.lower(): i for i, cat in enumerate(categories)}

            def encode_one_hot(x_str: str):
                x_str_lower = x_str.lower()
                for cat, idx in cat_to_idx.items():
                    if cat in x_str_lower:
                        one_hot = [0.0] * len(categories)
                        one_hot[idx] = 1.0
                        return one_hot
                return None

            y_series = data["m/z"].astype(str).apply(encode_one_hot)
            mask = y_series.notna()
            X = data[mask].drop("m/z", axis=1).to_numpy(dtype=np.float32)
            y = np.array(y_series[mask].tolist(), dtype=np.float32)

        elif enc_type == "map":
            mapping = encoding_cfg.get("map", {})

            def encode_map(x_str: str):
                for key, val in mapping.items():
                    if key != "default" and key in x_str:
                        return val
                return mapping.get("default")

            y_series = data["m/z"].astype(str).apply(encode_map)
            mask = y_series.notna()
            X = data[mask].drop("m/z", axis=1).to_numpy(dtype=np.float32)
            y = np.array(y_series[mask].tolist(), dtype=np.float32)

        elif enc_type == "regex_float":
            pattern = encoding_cfg.get("pattern")

            def encode_regex(x_str: str):
                match = re.search(pattern, x_str)
                return float(match.group(1)) if match else None

            y_series = data["m/z"].astype(str).apply(encode_regex)
            mask = y_series.notna()
            X = data[mask].drop("m/z", axis=1).to_numpy(dtype=np.float32)
            y = np.array(y_series[mask].tolist(), dtype=np.float32)

        else:
            raise ValueError(
                f"Unknown or missing label encoding type for {self.dataset_type}"
            )

        if y.ndim == 1:
            y = y[:, np.newaxis]
        return X, y

    def extract_groups(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extracts group identifiers from the data, usually from the 'm/z' column.

        Args:
            data (pd.DataFrame): The filtered DataFrame.

        Returns:
            np.ndarray: Array of group identifiers.
        """
        if "m/z" not in data.columns:
            return np.arange(len(data))

        # Common logic for most fish datasets: group is the part before the first underscore
        groups = data["m/z"].astype(str).apply(lambda x: x.split("_")[0])
        
        # For instance recognition, each row (instance) is its own group/identity
        dataset_name = self.dataset_type.name.lower().replace("_", "-")
        if "instance-recognition" in dataset_name:
            groups = data.iloc[:, 0].astype(str)

        return groups.to_numpy()


def preprocess_data_pipeline(
    data_processor: DataProcessor,
    file_path: Union[str, Path],
    is_pre_train: bool = False,
    augmentation_cfg: Optional[AugmentationConfig] = None,
) -> Tuple[DataLoader, pd.DataFrame, pd.DataFrame]:
    """
    Runs the full data loading, filtering, encoding, and dataset creation pipeline.

    Args:
        data_processor (DataProcessor): Configured processor instance.
        file_path (Union[str, Path]): Path to the raw data file.
        is_pre_train (bool, optional): Whether to skip filtering. Defaults to False.
        augmentation_cfg (Optional[AugmentationConfig], optional): Config for augmentation. Defaults to None.

    Returns:
        Tuple[DataLoader, pd.DataFrame, pd.DataFrame]:
            - DataLoader: The ready-to-use PyTorch DataLoader.
            - raw_df (pd.DataFrame): The raw loaded DataFrame.
            - filtered_df (pd.DataFrame): The DataFrame after filtering.
    """
    raw_df = data_processor.load_data(file_path)
    filtered_df = data_processor.filter_data(raw_df, is_pre_train)
    if filtered_df.empty:
        return (
            DataLoader(
                CustomDataset(np.array([]), np.array([])),
                batch_size=data_processor.batch_size,
            ),
            raw_df,
            filtered_df,
        )

    X, y = data_processor.encode_labels(filtered_df)
    dataset_class = (
        SiameseDataset
        if "instance-recognition"
        in data_processor.dataset_type.name.lower().replace("_", "-")
        else CustomDataset
    )
    torch_dataset = dataset_class(X, y)

    data_loader = DataLoader(
        torch_dataset, batch_size=data_processor.batch_size, shuffle=True, pin_memory=True
    )
    if augmentation_cfg and augmentation_cfg.enabled:
        data_loader = DataAugmenter(augmentation_cfg).augment(data_loader)
    return data_loader, raw_df, filtered_df


class DataModule:
    """
    High-level interface for data management in the training pipeline.

    Encapsulates the DataProcessor and pipeline execution to provide a clean
    API for accessing DataLoaders and dataset statistics.

    Attributes:
        dataset_name_str (str): The name of the dataset.
        file_path (Union[str, Path]): Path to the data file.
        batch_size (int): Batch size.
        is_pre_train (bool): Whether this is for pre-training.
        augmentation_config (Optional[AugmentationConfig]): Augmentation settings.
        processor (DataProcessor): Internal processor instance.

    Examples:
        >>> # Note: Setup requires a valid data file, so we just show initialization
        >>> dm = DataModule(dataset_name="species", file_path="data/REIMS.xlsx", batch_size=32)
        >>> dm.dataset_name_str
        'species'
        >>> dm.batch_size
        32
    """

    def __init__(
        self,
        dataset_name: str,
        file_path: Union[str, Path],
        batch_size: int = 64,
        is_pre_train: bool = False,
        augmentation_config: Optional[AugmentationConfig] = None,
    ) -> None:
        """
        Initializes the DataModule.

        Args:
            dataset_name (str): Name of the dataset (e.g., "species").
            file_path (Union[str, Path]): Path to data file.
            batch_size (int, optional): Batch size. Defaults to 64.
            is_pre_train (bool, optional): Skip filtering if True. Defaults to False.
            augmentation_config (Optional[AugmentationConfig], optional): Augmentation. Defaults to None.
        """
        self.dataset_name_str = dataset_name
        self.file_path = file_path
        self.batch_size = batch_size
        self.is_pre_train = is_pre_train
        self.augmentation_config = augmentation_config
        self.processor = DataProcessor(DatasetType.from_string(dataset_name), batch_size)
        self.train_loader, self.raw_data, self.filtered_data = None, None, None

    def setup(self) -> None:
        """
        Triggers the loading and preprocessing pipeline.
        Must be called before accessing dataloaders or statistics.
        """
        self.train_loader, self.raw_data, self.filtered_data = preprocess_data_pipeline(
            self.processor,
            self.file_path,
            self.is_pre_train,
            self.augmentation_config,
        )

    def get_groups(self) -> Optional[np.ndarray]:
        """
        Returns group identifiers for the filtered data (usually from the first column).
        Useful for GroupKFold validation.

        Returns:
            Optional[np.ndarray]: Array of group labels or None if empty.
        """
        if self.filtered_data is None:
            self.setup()
        if self.filtered_data.empty:
            return None
        return self.processor.extract_groups(self.filtered_data)

    def get_dataset(self):
        """
        Returns the underlying PyTorch Dataset.

        Returns:
            Dataset: The PyTorch dataset used by the DataLoader.
        """
        return (
            self.train_loader.dataset
            if self.train_loader
            else CustomDataset(np.array([]), np.array([]))
        )

    def get_train_dataframe(self) -> pd.DataFrame:
        """
        Returns the raw loaded DataFrame.

        Returns:
            pd.DataFrame: The raw dataframe.
        """
        return self.raw_data if self.raw_data is not None else pd.DataFrame()

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the configured DataLoader.

        Returns:
            DataLoader: The training data loader.
        """
        return (
            self.train_loader
            if self.train_loader
            else DataLoader(
                CustomDataset(np.array([]), np.array([])), batch_size=self.batch_size
            )
        )

    def get_numpy_data(self, labels_as_indices: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the dataset as numpy arrays (X, y).

        Args:
            labels_as_indices (bool, optional): If True and y is one-hot, returns class indices. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (Features, Labels) as numpy arrays.
        """
        dataset = self.get_dataset()
        X = dataset.samples.cpu().numpy()
        y = dataset.labels.cpu().numpy()

        if labels_as_indices and y.ndim > 1 and y.shape[1] > 1:
            y = np.argmax(y, axis=1)
        elif labels_as_indices and y.ndim > 1 and y.shape[1] == 1:
            y = y.flatten()

        return X, y

    def get_input_dim(self) -> int:
        """
        Calculates the input feature dimension from the loaded data.

        Returns:
            int: The size of the input feature vector.
        """
        if not self.train_loader:
            self.setup()
        return (
            self.train_loader.dataset.samples.shape[1]
            if self.train_loader and len(self.train_loader.dataset) > 0
            else 0
        )

    def get_num_classes(self) -> int:
        """
        Determines number of classes dynamically.

        Returns:
            int: Number of output classes.
        """
        if not self.train_loader:
            self.setup()
        dataset = self.train_loader.dataset
        if isinstance(dataset, SiameseDataset):
            return 2
        labels = dataset.labels
        return labels.shape[1] if labels.ndim > 1 else 1

    def get_class_names(self) -> List[str]:
        """
        Returns class names from config or encoder.

        Returns:
            List[str]: List of class names.
        """
        categories = self.processor.config.get("categories")
        if categories:
            return categories

        if self.processor.label_encoder_:
            return list(self.processor.label_encoder_.classes_)

        num_classes = self.get_num_classes()
        return [str(i) for i in range(num_classes)]


def create_data_module(
    dataset_name: str,
    file_path: Union[str, Path],
    batch_size: int = 64,
    is_pre_train: bool = False,
    augmentation_enabled: bool = False,
    **kwargs,
) -> DataModule:
    """
    Factory function to create a DataModule instance.

    Examples:
        >>> dm = create_data_module("species", "data/REIMS.xlsx", batch_size=32)
        >>> isinstance(dm, DataModule)
        True
        >>> dm.batch_size == 32
        True

    Args:
        dataset_name (str): Name of the dataset.
        file_path (Union[str, Path]): Path to data.
        batch_size (int, optional): Batch size. Defaults to 64.
        is_pre_train (bool, optional): Pre-training mode. Defaults to False.
        augmentation_enabled (bool, optional): Enable augmentation. Defaults to False.
        **kwargs: Additional augmentation arguments.

    Returns:
        DataModule: The created DataModule.
    """
    aug_config = (
        AugmentationConfig(enabled=True, **kwargs) if augmentation_enabled else None
    )
    return DataModule(dataset_name, file_path, batch_size, is_pre_train, aug_config)
