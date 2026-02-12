# -*- coding: utf-8 -*-
"""
Data module for managing loading, filtering, and preprocessing of datasets.
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

from .datasets import CustomDataset, SiameseDataset
from .augmentation import AugmentationConfig, DataAugmenter
from fishy._core.config_loader import load_config

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles the low-level processing of raw data into features and labels."""

    def __init__(self, dataset_name: str, batch_size: int = 64) -> None:
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.label_encoder_ = None
        all_configs = load_config("datasets")
        self.config = all_configs.get(dataset_name, {})
        if not self.config and dataset_name == "oil-regression": self.config = all_configs.get("oil_regression", {})
        if not self.config and dataset_name == "oil-simple": self.config = all_configs.get("oil_simple", {})

    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        path = Path(file_path)
        if not path.exists(): raise FileNotFoundError(f"Data file not found: {path}")
        return pd.read_excel(path) if path.suffix.lower() == ".xlsx" else pd.read_csv(path)

    def filter_data(self, data: pd.DataFrame, is_pre_train: bool = False) -> pd.DataFrame:
        if is_pre_train: return data
        df = data.copy()
        df = df[~df["m/z"].astype(str).str.contains("QC", case=False, na=False)]
        rules = self.config.get("filter_rules", {})
        if "exclude_mz" in rules:
            for p in rules["exclude_mz"]: df = df[~df["m/z"].astype(str).str.contains(p, case=False, na=False)]
        if "include_mz_pattern" in rules:
            df = df[df["m/z"].astype(str).str.contains(rules["include_mz_pattern"], case=False, na=False)]
        if "exclude_instance_pattern" in rules:
            df = df[~df.iloc[:, 0].astype(str).str.contains(rules["exclude_instance_pattern"], case=False, na=False)]
        return df

    def encode_labels(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        encoding_cfg = self.config.get("label_encoding", {})
        enc_type = encoding_cfg.get("type")
        if data.empty: return np.empty((0, 0)), np.empty((0, 0))

        if enc_type == "sklearn":
            X = data.iloc[:, 1:].to_numpy(dtype=np.float32)
            self.label_encoder_ = LabelEncoder()
            y_indices = self.label_encoder_.fit_transform(data.iloc[:, 0].astype(str))
            y = np.eye(len(self.label_encoder_.classes_), dtype=np.float32)[y_indices]
        elif enc_type == "one_hot":
            categories = self.config.get("categories", [])
            cat_to_idx = {cat.lower(): i for i, cat in enumerate(categories)}
            def encode_one_hot(x_str: str):
                for cat, idx in cat_to_idx.items():
                    if cat in x_str.lower():
                        one_hot = [0.0] * len(categories); one_hot[idx] = 1.0; return one_hot
                return None
            y_series = data["m/z"].astype(str).apply(encode_one_hot)
            mask = y_series.notna(); X = data[mask].drop("m/z", axis=1).to_numpy(dtype=np.float32)
            y = np.array(y_series[mask].tolist(), dtype=np.float32)
        elif enc_type == "map":
            mapping = encoding_cfg.get("map", {})
            def encode_map(x_str: str):
                for key, val in mapping.items():
                    if key != "default" and key in x_str: return val
                return mapping.get("default")
            y_series = data["m/z"].astype(str).apply(encode_map)
            mask = y_series.notna(); X = data[mask].drop("m/z", axis=1).to_numpy(dtype=np.float32)
            y = np.array(y_series[mask].tolist(), dtype=np.float32)
        elif enc_type == "regex_float":
            pattern = encoding_cfg.get("pattern")
            def encode_regex(x_str: str):
                match = re.search(pattern, x_str); return float(match.group(1)) if match else None
            y_series = data["m/z"].astype(str).apply(encode_regex)
            mask = y_series.notna(); X = data[mask].drop("m/z", axis=1).to_numpy(dtype=np.float32)
            y = np.array(y_series[mask].tolist(), dtype=np.float32)
        else: raise ValueError(f"Unknown label encoding type for {self.dataset_name}")
        if y.ndim == 1: y = y[:, np.newaxis]
        return X, y

    def extract_groups(self, data: pd.DataFrame) -> np.ndarray:
        if "m/z" not in data.columns: return np.arange(len(data))
        groups = data["m/z"].astype(str).apply(lambda x: x.split("_")[0])
        if "instance-recognition" in self.dataset_name: groups = data.iloc[:, 0].astype(str)
        return groups.to_numpy()

def preprocess_data_pipeline(data_processor: DataProcessor, file_path: Union[str, Path], is_pre_train: bool = False, augmentation_cfg: Optional[AugmentationConfig] = None) -> Tuple[DataLoader, pd.DataFrame, pd.DataFrame]:
    raw_df = data_processor.load_data(file_path); filtered_df = data_processor.filter_data(raw_df, is_pre_train)
    if filtered_df.empty: return (DataLoader(CustomDataset(np.array([]), np.array([])), batch_size=data_processor.batch_size), raw_df, filtered_df)
    X, y = data_processor.encode_labels(filtered_df)
    dataset_class = SiameseDataset if "instance-recognition" in data_processor.dataset_name else CustomDataset
    torch_dataset = dataset_class(X, y)
    data_loader = DataLoader(torch_dataset, batch_size=data_processor.batch_size, shuffle=True, pin_memory=True)
    if augmentation_cfg and augmentation_cfg.enabled: data_loader = DataAugmenter(augmentation_cfg).augment(data_loader)
    return data_loader, raw_df, filtered_df

class DataModule:
    """High-level interface for data management."""

    def __init__(self, dataset_name: str, file_path: Union[str, Path], batch_size: int = 64, is_pre_train: bool = False, augmentation_config: Optional[AugmentationConfig] = None) -> None:
        self.dataset_name_str = dataset_name; self.file_path = file_path; self.batch_size = batch_size; self.is_pre_train = is_pre_train; self.augmentation_config = augmentation_config
        self.processor = DataProcessor(dataset_name, batch_size); self.train_loader, self.raw_data, self.filtered_data = None, None, None

    def setup(self) -> None:
        self.train_loader, self.raw_data, self.filtered_data = preprocess_data_pipeline(self.processor, self.file_path, self.is_pre_train, self.augmentation_config)

    def get_groups(self) -> Optional[np.ndarray]:
        if self.filtered_data is None: self.setup()
        return self.processor.extract_groups(self.filtered_data) if not self.filtered_data.empty else None

    def get_dataset(self): return self.train_loader.dataset if self.train_loader else CustomDataset(np.array([]), np.array([]))
    def get_train_dataframe(self) -> pd.DataFrame: return self.raw_data if self.raw_data is not None else pd.DataFrame()

    def get_filtered_dataframe(self) -> pd.DataFrame:
        if self.filtered_data is None: self.setup()
        df = self.filtered_data.copy()
        if not df.empty:
            # Add human-readable 'Class Name' column
            names = self.get_class_names(); X, y = self.get_numpy_data(labels_as_indices=True)
            # Match lengths (in case some rows were dropped during encoding)
            if len(y) == len(df): df["Class Name"] = [names[i] for i in y]
            else:
                # Fallback: re-encode just to get names for the filtered df
                _, y_full = self.processor.encode_labels(df)
                y_idx = np.argmax(y_full, axis=1) if y_full.shape[1] > 1 else y_full.flatten().astype(int)
                df["Class Name"] = [names[i] for i in y_idx]
        return df

    def get_train_dataloader(self) -> DataLoader: return self.train_loader if self.train_loader else DataLoader(CustomDataset(np.array([]), np.array([])), batch_size=self.batch_size)

    def get_numpy_data(self, labels_as_indices: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        dataset = self.get_dataset(); X = dataset.samples.cpu().numpy(); y = dataset.labels.cpu().numpy()
        if labels_as_indices and y.ndim > 1 and y.shape[1] > 1: y = np.argmax(y, axis=1)
        elif labels_as_indices and y.ndim > 1 and y.shape[1] == 1: y = y.flatten().astype(int)
        return X, y

    def get_input_dim(self) -> int:
        if not self.train_loader: self.setup()
        return self.train_loader.dataset.samples.shape[1] if self.train_loader and len(self.train_loader.dataset) > 0 else 0

    def get_num_classes(self) -> int:
        if not self.train_loader: self.setup()
        dataset = self.train_loader.dataset; labels = dataset.labels
        return labels.shape[1] if labels.ndim > 1 else 1

    def get_class_names(self) -> List[str]:
        # Handle 'map' encoding human readable names
        if self.processor.config.get("label_encoding", {}).get("type") == "map":
            mapping = self.processor.config["label_encoding"]["map"]
            # species: "H": [0, 1] -> 1 is Hoki? Let's check config. 
            # Actually, standardizing: return the keys of the map or specific labels
            if self.dataset_name_str == "species": return ["Mackerel", "Hoki"]
            if self.dataset_name_str == "cross-species": return ["Mix", "Hoki", "Mackerel"]
            return sorted(list(mapping.keys()))
        
        categories = self.processor.config.get("categories")
        if categories: return categories
        if self.processor.label_encoder_: return list(self.processor.label_encoder_.classes_)
        return [str(i) for i in range(self.get_num_classes())]

def create_data_module(dataset_name: str, file_path: Union[str, Path], batch_size: int = 64, is_pre_train: bool = False, augmentation_enabled: bool = False, **kwargs) -> DataModule:
    aug_config = AugmentationConfig(enabled=True, **kwargs) if augmentation_enabled else None
    return DataModule(dataset_name, file_path, batch_size, is_pre_train, aug_config)
