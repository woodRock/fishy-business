import os 
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd 
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = torch.tensor(samples, dtype=torch.float32)
        # Credit: https://stackoverflow.com/a/70323486
        self.labels = torch.from_numpy(np.vstack(labels).astype(float))
        # Normalize the features to be between in [0,1]
        self.samples = F.normalize(self.samples, dim = 0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

def random_augmentation(X, y, num_augmentations=5,
                        is_noise = True, is_shift = False, is_scale = False,
                        noise_level=0.1, shift_range=0.1, scale_range=0.1):
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

def preprocess_dataset(is_data_augmentation=True, batch_size=64):
    path = ['/vol','ecrg-solar', 'woodj4', 'fishy-business', 'data','REIMS_data.xlsx']
    path = os.path.join(*path)

    raw = pd.read_excel(path)

    data = raw[~raw['m/z'].str.contains('HM')]
    data = data[~data['m/z'].str.contains('QC')]
    data = data[~data['m/z'].str.contains('HM')]
    X = data.drop('m/z', axis=1) # X contains only the features.
    # Onehot encoding for the class labels, e.g. [0,1] for Hoki, [1,0] for Mackeral.
    y = data['m/z'].apply(lambda x: [0,1] if 'H' in x else [1,0])
    
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
    def __init__(self, patience=5, delta=0, path='checkpoint.pt'):
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

    def __call__(self, train_acc, val_loss, model, verbose=False):
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
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Args:
            val_loss (float): Validation loss
            model (torch.nn.Module): Transformer model
        """
        if val_loss < self.val_loss_min:
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss