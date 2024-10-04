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
    def __init__(self, samples: Iterable, labels: Iterable) -> None:
        self.samples = torch.tensor(samples, dtype=torch.float32)
        self.labels = torch.from_numpy(np.vstack(labels).astype(float))
        self.samples = F.normalize(self.samples, dim=1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Iterable, Iterable, Iterable]:
        # Randomly choose another sample
        idx2 = np.random.choice(len(self.labels))
        
        X1, y1 = self.samples[idx], self.labels[idx]
        X2, y2 = self.samples[idx2], self.labels[idx2]
        
        # 1 if same class, 0 if different class
        pair_label = torch.FloatTensor([int(torch.all(y1 == y2))])
        
        return X1, X2, pair_label

def load_from_file(
        # path: Iterable = ["~/", "Desktop", "fishy-business", "data", "REIMS_data.xlsx"]
        path: Iterable = ["/vol","ecrg-solar","woodj4","fishy-business","data", "REIMS_data.xlsx"]
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
    print(f"len(data): {len(data)}")
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
        print(f"y: {y}")
        y = np.eye(24)[y]
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
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    train_dataset = SiameseDataset(X_train, y_train)
    val_dataset = SiameseDataset(X_val, y_val)

    # Create PyTorch DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader