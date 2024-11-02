import logging
import os 
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd 
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Iterable, Tuple, Union

class SiameseDataset(Dataset):
    """ Generate a dataset of paired instances for contrastive learning. """
    def __init__(self, samples, labels, pairs_per_sample=50):
        self.samples = torch.tensor(samples, dtype=torch.float32)
        self.labels = torch.from_numpy(np.vstack(labels).astype(float))
        self.samples = F.normalize(self.samples, dim=1)
        self.pairs_per_sample = pairs_per_sample
        
        # Create dictionaries to store indices for each class
        self.class_indices = {}
        for idx, label in enumerate(self.labels):
            label_tuple = tuple(label.tolist())
            if label_tuple not in self.class_indices:
                self.class_indices[label_tuple] = []
            self.class_indices[label_tuple].append(idx)

        samples = [] 
        labels = [] 

        # For each sample in the dataset.
        for sample_idx, _ in enumerate(self.samples):
            X1, y1 = self.samples[sample_idx], self.labels[sample_idx]
            
            # Generate 50 pairs per sample.
            for _ in range(50):
                # 50% chance to choose a pair of the same class
                if np.random.random() < 0.5:
                    same_class_indices = self.class_indices[tuple(y1.tolist())]
                    if len(same_class_indices) > 1:  # Ensure there's at least one other sample in the same class
                        idx2 = np.random.choice([i for i in same_class_indices if i != sample_idx])
                    else:
                        idx2 = np.random.choice(len(self.samples))  # If no other samples in the same class, choose randomly
                else:
                    idx2 = np.random.choice(len(self.samples))  # Choose a random sample
                
                X2, y2 = self.samples[idx2], self.labels[idx2]
                
                # 1 if same class, 0 if different class
                pair_label = torch.FloatTensor([int(torch.all(y1 == y2))])

                samples.append((X1, X2))
                labels.append(pair_label)

        # Store the newly generated samples and labels.
        self.samples = samples 
        self.labels = labels 

    def __len__(self):
        return len(self.samples) 

    def __getitem__(self, idx):
        # Determine the original sample index and pair number
        sample_idx = idx // self.pairs_per_sample
        (X1,X2), y1 = self.samples[sample_idx], self.labels[sample_idx]
        return X1, X2, y1

def load_from_file(
        path: Iterable = ["~/", "Desktop", "fishy-business", "data", "REIMS.xlsx"]
        # path: Iterable = ["/vol","ecrg-solar","woodj4","fishy-business","data", "REIMS_data.xlsx"]
    ) -> pd.DataFrame:
    path = os.path.join(*path)
    data = pd.read_excel(path)
    return data

def filter_dataset(dataset: str, data: pd.DataFrame) -> pd.DataFrame:
    data = data[~data['m/z'].str.contains('QC')]
    
    if dataset in ["species", "part", "oil", "instance-recognition"]:
        data = data[~data['m/z'].str.contains('HM')]
    
    if dataset in ["species", "part", "cross-species"]:
        data = data[~data['m/z'].str.contains('MO')]

    if dataset == "instance-recognition":
        data = data[~data.iloc[:, 0].astype(str).str.contains('QC|HM|MO|fillet|frames|gonads|livers|skins|guts|frame|heads', case=False, na=False)]
    return data

def one_hot_encoded_labels(dataset, data):
    if dataset == "species":
        y = data['m/z'].apply(lambda x: [0,1] if 'H' in x else [1,0])
    elif dataset == "part":
        y = data['m/z'].apply(lambda x:
                          [1,0,0,0,0,0] if 'Fillet' in x
                    else ([0,1,0,0,0,0] if 'Heads' in x
                    else ([0,0,1,0,0,0] if 'Livers' in x
                    else ([0,0,0,1,0,0] if 'Skins' in x
                    else ([0,0,0,0,1,0] if 'Guts' in x
                    else ([0,0,0,0,0,1] if 'Frames' in x
                    else None))))))
    elif dataset == "oil":
        y = data['m/z'].apply(lambda x:
                          [1,0,0,0,0,0,0] if 'MO 50' in x
                    else ([0,1,0,0,0,0,0] if 'MO 25' in x
                    else ([0,0,1,0,0,0,0] if 'MO 10' in x
                    else ([0,0,0,1,0,0,0] if 'MO 05' in x
                    else ([0,0,0,0,1,0,0] if 'MO 01' in x
                    else ([0,0,0,0,0,1,0] if 'MO 0.1' in x
                    else ([0,0,0,0,0,0,1] if 'MO 0' in x
                    else None)))))))
    elif dataset == "cross-species":
        y = data['m/z'].apply(lambda x: 
                              [1,0,0] if 'HM' in x 
                        else ([0,1,0] if 'H' in x
                        else ([0,0,1] if 'M' in x
                        else None)))
    elif dataset == "instance-recognition":
        X = data.iloc[:, 1:].to_numpy() 
        # Take only the class label column.
        y = data.iloc[:, 0].to_numpy()
        X,y = np.array(X), np.array(y)
        le = LabelEncoder()
        y = le.fit_transform(y)
        n_classes = len(np.unique(y))
        y = np.eye(n_classes)[y]
    else: 
        raise ValueError(f"No valid dataset was specified: {dataset}")
    return y

def remove_instances_with_none_labels(X: Iterable, y: Iterable) -> Union[np.array, np.array]:
    xs = []
    ys = []
    for (x,y) in zip(X.to_numpy(),y):
        if y is not None:
            xs.append(x)
            ys.append(y)
    X = np.array(xs)
    y = np.array(ys)
    return X,y

def preprocess_dataset(dataset: str ="species", batch_size: int = 64) -> Union[DataLoader, DataLoader, int, int, pd.DataFrame]:
    data = load_from_file()
    data = filter_dataset(dataset=dataset, data=data)
    y = one_hot_encoded_labels(dataset=dataset, data=data)
    X = data.drop('m/z', axis=1)
    X, y = remove_instances_with_none_labels(X, y)

    # Split your dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.5, shuffle=True)

    train_dataset = SiameseDataset(X_train, y_train)
    val_dataset = SiameseDataset(X_val, y_val)

    # Create PyTorch DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


if __name__ == "__main__":
    def inspect_data(train_loader):
       class_counts = {0: 0, 1: 0}
       feature_means = []
       feature_stds = []
       
       for X1, X2, labels in train_loader:
           class_counts[0] += (labels == 0).sum().item()
           class_counts[1] += (labels == 1).sum().item()
           
           # Combine X1 and X2 for feature statistics
           X = torch.cat((X1, X2), dim=0)
           feature_means.append(X.mean(dim=0))
           feature_stds.append(X.std(dim=0))
       
       feature_means = torch.stack(feature_means).mean(dim=0)
       feature_stds = torch.stack(feature_stds).mean(dim=0)
       
       print(f"Class distribution: {class_counts}")
       print(f"Feature means range: [{feature_means.min().item():.2f}, {feature_means.max().item():.2f}]")
       print(f"Feature stds range: [{feature_stds.min().item():.2f}, {feature_stds.max().item():.2f}]")

    train_loader, val_loader = preprocess_dataset(dataset="instance-recognition")
    # Call this function before training
    inspect_data(train_loader)
    inspect_data(val_loader)