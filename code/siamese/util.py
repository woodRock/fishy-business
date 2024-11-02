import logging
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Union, Any
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "siamese_dataset.log")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    test_size: float = 0.5
    data_path: List[str] = None
    
    def __post_init__(self):
        if self.data_path is None:
            self.data_path = ["~/", "Desktop", "fishy-business", "data", "REIMS.xlsx"]

class SiameseDataset(Dataset):
    """Generate paired instances for contrastive learning."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, pairs_per_sample: int = 50):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.from_numpy(np.vstack(labels).astype(float))
        self.pairs_per_sample = pairs_per_sample
        self.features = F.normalize(self.features, dim=1)
        self.class_indices = self._create_class_indices()
        self.pairs, self.pair_labels = self._generate_pairs()
    
    def _create_class_indices(self) -> Dict[tuple, List[int]]:
        class_indices = {}
        for idx, label in enumerate(self.labels):
            label_tuple = tuple(label.tolist())
            if label_tuple not in class_indices:
                class_indices[label_tuple] = []
            class_indices[label_tuple].append(idx)
        return class_indices
    
    def _generate_pairs(self) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor]]:
        pairs = []
        labels = []
        
        for idx1, label1 in enumerate(self.labels):
            feat1 = self.features[idx1]
            label1_tuple = tuple(label1.tolist())
            
            for _ in range(self.pairs_per_sample):
                if torch.rand(1) < 0.5:
                    same_class_indices = self.class_indices[label1_tuple]
                    if len(same_class_indices) > 1:
                        idx2 = np.random.choice([i for i in same_class_indices if i != idx1])
                    else:
                        idx2 = np.random.choice(len(self.features))
                else:
                    idx2 = np.random.choice(len(self.features))
                
                feat2 = self.features[idx2]
                label2 = self.labels[idx2]
                pair_label = torch.tensor([float(torch.all(label1 == label2))])
                
                pairs.append((feat1, feat2))
                labels.append(pair_label)
        
        return pairs, labels
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.pairs[idx][0], self.pairs[idx][1], self.pair_labels[idx]

class DataPreprocessor:
    """Handle data loading and preprocessing for Siamese networks."""
    
    @staticmethod
    def load_data(config: DataConfig) -> pd.DataFrame:
        path = Path(*config.data_path).expanduser()
        return pd.read_excel(path)
    
    @staticmethod
    def filter_data(data: pd.DataFrame, dataset: str) -> pd.DataFrame:
        data = data[~data['m/z'].str.contains('QC', case=False, na=False)]
        
        if dataset in ["species", "part", "oil", "instance-recognition"]:
            data = data[~data['m/z'].str.contains('HM', case=False, na=False)]
        
        if dataset in ["species", "part", "cross-species"]:
            data = data[~data['m/z'].str.contains('MO', case=False, na=False)]
            
        if dataset == "instance-recognition":
            pattern = r'QC|HM|MO|fillet|frames|gonads|livers|skins|guts|frame|heads'
            data = data[~data.iloc[:, 0].astype(str).str.contains(pattern, case=False, na=False)]
            
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
            "species": {"H": [0,1], "default": [1,0]},
            "part": {
                "Fillet": [1,0,0,0,0,0], "Heads": [0,1,0,0,0,0],
                "Livers": [0,0,1,0,0,0], "Skins": [0,0,0,1,0,0],
                "Guts": [0,0,0,0,1,0], "Frames": [0,0,0,0,0,1]
            },
            "cross-species": {
                "HM": [1,0,0], "H": [0,1,0], "M": [0,0,1]
            }
        }
        
        if dataset not in encoding_patterns:
            raise ValueError(f"Invalid dataset: {dataset}")
            
        pattern = encoding_patterns[dataset]
        labels = data['m/z'].apply(lambda x: [
            pattern.get(key)[i] 
            for key in pattern.keys() 
            for i in range(len(pattern[key])) 
            if key in x
        ])
        
        if len(labels) == 0:
            raise ValueError(f"No valid labels found for dataset {dataset}")
            
        return labels.values

def prepare_dataset(config: DataConfig) -> Tuple[DataLoader, DataLoader]:
    preprocessor = DataPreprocessor()
    
    data = preprocessor.load_data(config)
    data = preprocessor.filter_data(data, config.dataset_name)
    
    features = data.drop('m/z', axis=1).to_numpy()
    labels = preprocessor.encode_labels(data, config.dataset_name)
    
    if len(features) < 2:
        raise ValueError(f"Not enough samples ({len(features)}) to split into train/val sets")
    
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels,
        stratify=labels,
        test_size=config.test_size,
        shuffle=True
    )
    
    train_dataset = SiameseDataset(X_train, y_train, config.pairs_per_sample)
    val_dataset = SiameseDataset(X_val, y_val, config.pairs_per_sample)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
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