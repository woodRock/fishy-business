import logging
import os 
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd 
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformer import Transformer
from typing import Iterable, Tuple, Union

class CustomDataset(Dataset):
    def __init__(self, 
            samples: Iterable, 
            labels: Iterable
        ) -> None:
        self.samples = torch.tensor(samples, dtype=torch.float32)
        # Credit: https://stackoverflow.com/a/70323486
        self.labels = torch.from_numpy(np.vstack(labels).astype(float))
        # Normalize the features to be between in [0,1]
        self.samples = F.normalize(self.samples, dim = 0)

    def __len__(self
    ) -> int:
        return len(self.samples)

    def __getitem__(self, 
        idx: int
    ) -> Tuple[Iterable, Iterable]:
        return self.samples[idx], self.labels[idx]

def random_augmentation(
        X: Iterable, 
        y: Iterable, 
        num_augmentations: int = 5,
        is_noise: bool = True, 
        is_shift: bool = False, 
        is_scale: bool = False,
        noise_level: float = 0.1, 
        shift_range: float = 0.1, 
        scale_range: float = 0.1
    ) -> Union[Iterable, Iterable]:
    xs = []
    ys = []
    for (x,y) in tqdm(zip(X,y), desc="Data augmentation"):
        # Include the orginal instance.
        xs.append(x)
        ys.append(y)
        for _ in range(num_augmentations):
            augmented = x
            if is_noise:
                # Add random noise
                noise = np.random.normal(scale=noise_level, size=x.shape)
                augmented = x + noise
            if is_shift:
                # Apply random shift
                shift_amount = np.random.uniform(-shift_range, shift_range)
                augmented = np.roll(augmented, int(shift_amount * len(x)))
            if is_scale:
                # Apply random scaling
                scale_factor = np.random.uniform(1 - scale_range, 1 + scale_range)
                augmented = augmented * scale_factor
            # Append the augmented data and label to the dataset.
            xs.append(augmented)
            ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys

def preprocess_dataset(
        dataset: str ="species", 
        is_data_augmentation: bool = True, 
        batch_size: int = 64
    ) -> Union[DataLoader, DataLoader, DataLoader, int, int, pd.DataFrame]:
    path = ['~/','Desktop', 'fishy-business', 'data','REIMS_data.xlsx']
    path = os.path.join(*path)

    data = pd.read_excel(path)
    y = []

    # Remove the quality control samples.
    data = data[~data['m/z'].str.contains('QC')]
    
    # Exclude cross-species samples from the dataset.
    if dataset == "species" or dataset == "part" or dataset == "oil":
        data = data[~data['m/z'].str.contains('HM')]
    
    # Exclude mineral oil samples from the dataset.
    if dataset == "species" or dataset == "part" or dataset == "cross-species":
        data = data[~data['m/z'].str.contains('MO')]
    
    # Either fish "species" or "part" dataset.
    if dataset == "species":
        # Onehot encoding for the class labels, e.g. [0,1] for Hoki, [1,0] for Mackeral.
        y = data['m/z'].apply(lambda x: [0,1] if 'H' in x else [1,0])
    elif dataset == "part":
        y = data['m/z'].apply(lambda x:
                          [1,0,0,0,0,0] if 'Fillet' in x
                    else ([0,1,0,0,0,0] if 'Heads' in x
                    else ([0,0,1,0,0,0] if 'Livers' in x
                    else ([0,0,0,1,0,0] if 'Skins' in x
                    else ([0,0,0,0,1,0] if 'Guts' in x
                    else ([0,0,0,0,0,1] if 'Frames' in x
                    else None ))))))  # Labels (0 for Hoki, 1 for Moki)
    elif dataset == "oil":
        # Onehot encodings for class labels (1 for Oil, 0 for No Oil)
        # Oil contaminated samples contain 'MO' in their class label.
        y = data['m/z'].apply(lambda x: [1,0] if 'MO' in x else [0,1])
    elif dataset == "cross-species":
        print(f"I get here")
        # Onehot encodings for class labels (1 for HM, 0 for Not Cross-species)
        # Cross-species contaminated samples contain 'HM' in their class label.
        y = data['m/z'].apply(lambda x: 
                              [1,0,0] if 'HM' in x 
                        else ([0,1,0] if 'H' in x
                        else ([0,0,1] if 'M' 
                        else None)))
    else: 
        # Return an excpetion if the dataset is not valid.
        raise ValueError(f"No valid dataset was specified: {dataset}")

    # X contains only the features.
    X = data.drop('m/z', axis=1)

    # Remove the "None" values from the dataset. 
    # Discard instances not related to the current problem.
    xs = []
    ys = []
    for (x,y) in zip(X.to_numpy(),y):
        if y is not None:
            xs.append(x)
            ys.append(y)
    X = np.array(xs)
    y = np.array(ys)
    
    # Convert to numpy arrays.
    X = np.array(X)
    y = np.array(y)

    # Evaluation parameters.
    train_split = 0.8
    val_split = 0.5 # 1/2 of 20%, validation and test, 10% and 10%, respectively.

    # Step 2: Split your dataset into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, stratify=y, test_size=(1-train_split))
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_split)
    
    # Data augmentation - adding random noise.
    if is_data_augmentation:
        X_train, y_train = random_augmentation(X_train,y_train)

    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    test_dataset = CustomDataset(X_test, y_test)

    assert train_dataset.samples.shape[0] == train_dataset.labels.shape[0] , "train_dataset samples and labels should have same length."
    assert val_dataset.samples.shape[0] == val_dataset.labels.shape[0] , "train_dataset samples and labels should have same length."
    assert test_dataset.samples.shape[0] == test_dataset.labels.shape[0] , "train_dataset samples and labels should have same length."

    # Step 4: Create PyTorch DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # calculate steps per epoch for training and validation set
    train_steps = len(train_loader.dataset) // batch_size
    val_steps = len(val_loader.dataset) // batch_size
    # when batch_size greater than dataset size, avoid division by zero.
    train_steps = max(1, train_steps)
    val_steps = max(1, val_steps)
    return train_loader, val_loader, test_loader, train_steps, val_steps, data


class EarlyStopping:
    def __init__(self,
        patience: int = 5,
        delta: float = 0, 
        path: str = 'checkpoint.pt'
    ) -> None:
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'transformer_checkpoint.pth'
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, 
        train_acc: Iterable, 
        val_loss: Iterable, 
        model: Transformer, 
        verbose: bool=False
    ) -> None:
        logger = logging.getLogger(__name__)
        """
        Args:
            val_loss (float): Validation loss
            model (torch.nn.Module): Transformer model
        """
        score = -val_loss

        # Check if the model has fit the training set.
        if train_acc == 1:
            # Employ early stopping once the model has fitted training set.
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                if verbose:
                    logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

    def save_checkpoint(self, 
            val_loss: Iterable, 
            model: Transformer
        ) -> None:
        """
        Args:
            val_loss (float): Validation loss
            model (torch.nn.Module): Transformer model
        """
        if val_loss < self.val_loss_min:
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss
