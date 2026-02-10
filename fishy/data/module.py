# -*- coding: utf-8 -*-
"""
Data module for managing loading, filtering, and preprocessing of datasets.

This module encapsulates all data-related operations, including reading files (Excel/CSV),
applying filtering rules based on dataset types (e.g., removing QC samples, filtering by 'm/z'),
encoding labels (one-hot, integer), and preparing PyTorch DataLoaders. It serves as the
primary interface for feeding data into the training pipeline.
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

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Handles the low-level processing of raw data into features and labels.

    This class manages specific rules for different dataset types, such as which 'm/z' patterns
    to include or exclude, and how to encode labels for classification or regression tasks.

    Args:
        dataset_type (DatasetType): The specific type of dataset being processed.
        batch_size (int): The batch size to be used for subsequent DataLoaders.
    """
    def __init__(self, dataset_type: DatasetType, batch_size: int = 64) -> None:
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.label_encoder_ = None
        self._PART_CATEGORIES = ["Fillet", "Heads", "Livers", "Skins", "Guts", "Gonads", "Frames"]
        self._OIL_CATEGORIES = ["MO 50", "MO 25", "MO 10", "MO 05", "MO 01", "MO 0.1", "MO 0"]

        self._FILTER_RULES = {
            (DatasetType.SPECIES, DatasetType.PART, DatasetType.OIL): {"exclude_mz": ["HM"]},
            (DatasetType.SPECIES, DatasetType.PART, DatasetType.CROSS_SPECIES): {"exclude_mz": ["MO"]},
            DatasetType.PART: {"include_mz_pattern": "|".join(self._PART_CATEGORIES)},
            DatasetType.OIL: {"include_mz_pattern": "MO"},
            (DatasetType.INSTANCE_RECOGNITION, DatasetType.INSTANCE_RECOGNITION_HARD): {
                "exclude_instance_pattern": f"QC|HM|MO|{'|'.join(self._PART_CATEGORIES)}"
            },
            DatasetType.CROSS_SPECIES_HARD: {
                "exclude_instance_pattern": f"^H |^M |QC|HM|MO|{'|'.join(self._PART_CATEGORIES)}"
            },
        }
        self._LABEL_ENCODERS_MAP = {
            DatasetType.SPECIES: lambda x: ([0.0, 1.0] if "H" in x else ([1.0, 0.0] if "M" in x else None)),
            DatasetType.PART: self._create_one_hot_encoder(self._PART_CATEGORIES),
            DatasetType.OIL: self._create_one_hot_encoder(self._OIL_CATEGORIES),
            DatasetType.OIL_REGRESSION: lambda x: (float(re.search(r"MO\s*([\d\.]+)", x).group(1)) if re.search(r"MO\s*([\d\.]+)", x) else None),
            DatasetType.OIL_SIMPLE: lambda x: ([1.0, 0.0] if "MO" in x else ([0.0, 1.0] if x.strip() else None)),
            DatasetType.CROSS_SPECIES: lambda x: ([1.0, 0.0, 0.0] if "HM" in x else ([0.0, 1.0, 0.0] if "H" in x else ([0.0, 0.0, 1.0] if "M" in x else None))),
            DatasetType.INSTANCE_RECOGNITION: "use_sklearn_label_encoder",
            DatasetType.INSTANCE_RECOGNITION_HARD: "use_sklearn_label_encoder",
            DatasetType.CROSS_SPECIES_HARD: "use_sklearn_label_encoder",
        }

    def _create_one_hot_encoder(self, categories: List[str]):
        """Creates a closure for one-hot encoding a fixed list of categories."""
        cat_to_idx = {cat.lower(): i for i, cat in enumerate(categories)}
        def encoder(x_str: str):
            x_str_lower = x_str.lower()
            for cat, idx in cat_to_idx.items():
                if cat in x_str_lower:
                    one_hot = [0.0] * len(categories)
                    one_hot[idx] = 1.0
                    return one_hot
            return None
        return encoder

    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Loads data from an Excel or CSV file."""
        path = Path(file_path)
        if not path.exists(): raise FileNotFoundError(f"Data file not found: {path}")
        return pd.read_excel(path) if path.suffix.lower() == ".xlsx" else pd.read_csv(path)

    def filter_data(self, data: pd.DataFrame, is_pre_train: bool = False) -> pd.DataFrame:
        """Applies dataset-specific filtering rules to the raw DataFrame."""
        if is_pre_train: return data
        df = data.copy()
        df = df[~df["m/z"].astype(str).str.contains("QC", case=False, na=False)]
        for key, rules in self._FILTER_RULES.items():
            if self.dataset_type in (key if isinstance(key, tuple) else (key,)):
                if "exclude_mz" in rules:
                    for p in rules["exclude_mz"]: df = df[~df["m/z"].astype(str).str.contains(p, case=False, na=False)]
                if "include_mz_pattern" in rules:
                    df = df[df["m/z"].astype(str).str.contains(rules["include_mz_pattern"], case=False, na=False)]
                if "exclude_instance_pattern" in rules:
                    df = df[~df.iloc[:, 0].astype(str).str.contains(rules["exclude_instance_pattern"], case=False, na=False)]
        return df

    def encode_labels(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Encodes the labels/targets from the DataFrame into numpy arrays."""
        label_action = self._LABEL_ENCODERS_MAP.get(self.dataset_type)
        if data.empty: return np.empty((0, 0)), np.empty((0, 0))

        if label_action == "use_sklearn_label_encoder":
            X = data.iloc[:, 1:].to_numpy(dtype=np.float32)
            self.label_encoder_ = LabelEncoder()
            y_indices = self.label_encoder_.fit_transform(data.iloc[:, 0].astype(str))
            y = np.eye(len(self.label_encoder_.classes_), dtype=np.float32)[y_indices]
        elif callable(label_action):
            y_series = data["m/z"].astype(str).apply(label_action)
            mask = y_series.notna()
            X = data[mask].drop("m/z", axis=1).to_numpy(dtype=np.float32)
            y = np.array(y_series[mask].tolist(), dtype=np.float32)
        else:
            raise ValueError(f"No label encoding for {self.dataset_type}")
        
        if y.ndim == 1: y = y[:, np.newaxis]
        return X, y

def preprocess_data_pipeline(
    data_processor: DataProcessor,
    file_path: Union[str, Path],
    is_pre_train: bool = False,
    augmentation_cfg: Optional[AugmentationConfig] = None,
) -> Tuple[DataLoader, pd.DataFrame, pd.DataFrame]:
    """
    Runs the full data loading, filtering, encoding, and DataLoader creation pipeline.

    Args:
        data_processor: The processor containing rules for the specific dataset.
        file_path: Path to the data file.
        is_pre_train: Flag to skip filtering for pre-training tasks.
        augmentation_cfg: Optional configuration for data augmentation.

    Returns:
        Tuple[DataLoader, pd.DataFrame, pd.DataFrame]: The final DataLoader, raw DataFrame, and filtered DataFrame.
    """
    raw_df = data_processor.load_data(file_path)
    filtered_df = data_processor.filter_data(raw_df, is_pre_train)
    if filtered_df.empty:
        return DataLoader(CustomDataset(np.array([]), np.array([])), batch_size=data_processor.batch_size), raw_df, filtered_df

    X, y = data_processor.encode_labels(filtered_df)
    dataset_class = SiameseDataset if "instance-recognition" in data_processor.dataset_type.name.lower().replace("_", "-") else CustomDataset
    torch_dataset = dataset_class(X, y)
    
    data_loader = DataLoader(torch_dataset, batch_size=data_processor.batch_size, shuffle=True, pin_memory=True)
    if augmentation_cfg and augmentation_cfg.enabled:
        data_loader = DataAugmenter(augmentation_cfg).augment(data_loader)
    return data_loader, raw_df, filtered_df

class DataModule:
    """
    High-level interface for data management in the training pipeline.

    This class coordinates the initialization of the DataProcessor and the execution
    of the preprocessing pipeline to provide ready-to-use DataLoaders.

    Args:
        dataset_name (str): Name of the dataset type.
        file_path (Union[str, Path]): Path to the data file.
        batch_size (int): Batch size.
        is_pre_train (bool): Whether this is for a pre-training task.
        augmentation_config (Optional[AugmentationConfig]): Config for augmentation.
    """
    def __init__(self, dataset_name: str, file_path: Union[str, Path], batch_size: int = 64, is_pre_train: bool = False, augmentation_config: Optional[AugmentationConfig] = None) -> None:
        self.dataset_name_str = dataset_name
        self.file_path = file_path
        self.batch_size = batch_size
        self.is_pre_train = is_pre_train
        self.augmentation_config = augmentation_config
        self.processor = DataProcessor(DatasetType.from_string(dataset_name), batch_size)
        self.train_loader, self.raw_data, self.filtered_data = None, None, None

    def setup(self) -> None:
        """Triggers the loading and preprocessing pipeline."""
        self.train_loader, self.raw_data, self.filtered_data = preprocess_data_pipeline(self.processor, self.file_path, self.is_pre_train, self.augmentation_config)

    def get_dataset(self):
        """Returns the underlying PyTorch Dataset."""
        return self.train_loader.dataset if self.train_loader else CustomDataset(np.array([]), np.array([]))

    def get_train_dataframe(self):
        """Returns the raw loaded DataFrame."""
        return self.raw_data if self.raw_data is not None else pd.DataFrame()

    def get_train_dataloader(self):
        """Returns the configured DataLoader."""
        return self.train_loader if self.train_loader else DataLoader(CustomDataset(np.array([]), np.array([])), batch_size=self.batch_size)

    def get_input_dim(self) -> int:
        """Calculates the input feature dimension from the loaded data."""
        if not self.train_loader: self.setup()
        return self.train_loader.dataset.samples.shape[1] if self.train_loader and len(self.train_loader.dataset) > 0 else 0

def create_data_module(dataset_name: str, file_path: Union[str, Path], batch_size: int = 64, is_pre_train: bool = False, augmentation_enabled: bool = False, **kwargs) -> DataModule:
    """Factory function to create a DataModule instance."""
    aug_config = AugmentationConfig(enabled=True, **kwargs) if augmentation_enabled else None
    return DataModule(dataset_name, file_path, batch_size, is_pre_train, aug_config)