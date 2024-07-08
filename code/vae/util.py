import logging
import os 
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd 
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from typing import Iterable, Tuple, Union


class CustomDataset(Dataset):
    """ A custom data loader that convert numpy arrays to tensors."""
    def __init__(self, 
            samples: Iterable, 
            labels: Iterable
        ) -> None:
        """
        CustomDataset is a tailored DataSet for loading fish data.

        Args:
            samples (Iterable): the input features
            labels (Iterable): the class labels.
        """
        self.samples = torch.tensor(samples, dtype=torch.float32)
        # Credit: https://stackoverflow.com/a/70323486
        self.labels = torch.from_numpy(np.vstack(labels).astype(float))
        # Normalize the features to be between [0,1]
        self.samples = F.normalize(self.samples, dim = 0)

    def __len__(self
    ) -> int:
        """Return the length of the dataset.
        """
        return len(self.samples)

    def __getitem__(self, 
        idx: int
    ) -> Tuple[Iterable, Iterable]:
        """Retrieve an instance from the dataset.

        Args:
            idx (int): the index of the element to retrive.
        """
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
    """
    Perform random augmentation on the dataset.

    Args: 
        X (Iterable): the input features.
        y (Iterable): the class labels.
        num_augmentations (int): the number of augmentations per instance.
        is_noise (bool): conditional flag for random noise.
        is_shift (bool): conditional flag for random shift.
        is_scale (bool): conditional flag for random scale.
        noise_level (float): the factor to generate noise by.
        shift_range (float): the factor to shift by.
        scale_range (float): the factor to scale by.

    Returns:
        X,y (Iterable, Iterable): the augmented dataset.
    """
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

def load_from_file(
        path: Iterable = ['~/','Desktop', 'fishy-business', 'data','REIMS_data.xlsx']
    ) -> pd.DataFrame:
    """ Load the dataset from a file path.

    We use `os.path.join` so this code will run across platforms, both Windows, Mac and Linux.
    
    Args: 
        path (Iterable): Filepath where the dataset is stored. Defaults to ['~/','Desktop', 'fishy-business', 'data','REIMS_data.xlsx'].

    Returns 
        data (pd.DataFrame): the dataset is stored as a pandas dataframe.
    """
    path = os.path.join(*path)
    data = pd.read_excel(path)
    return data

def filter_dataset(
        dataset: str, 
        data: pd.DataFrame
    ) -> Union[Iterable, Iterable]:
    """ Remove the extra instances that are not needed for each downstream task.

    Args: 
        dataset (str): the name of the dataset. Can be "species", "part", "oil", or "cross-species".
        data (pd.DataFrame): the pandas dataframe containgin the data.

    Returns: 
        data (pd.DataFrame): the dataset is stored as a pandas dataframe.
    """
     # Remove the quality control samples.
    data = data[~data['m/z'].str.contains('QC')]
    
    # Exclude cross-species samples from the dataset.
    if dataset == "species" or dataset == "part" or dataset == "oil":
        data = data[~data['m/z'].str.contains('HM')]
    
    # Exclude mineral oil samples from the dataset.
    if dataset == "species" or dataset == "part" or dataset == "cross-species":
        data = data[~data['m/z'].str.contains('MO')]
    return data

def one_hot_encoded_labels(dataset, data):
    """One-hot encodings for the class labels.
    
    Depending on which downstream task is specified as dataset.
    This code encodes the class labels as one-hot encoded vectors.

    Args: 
        dataset (str): the name of the dataset. Can be "species", "part", "oil", or "cross-species".
        data (pd.DataFrame): the pandas dataframe containgin the data.

    Returns: 
        y (pd.DataFrame): the class lables stored as a pandas dataframe.
    """
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
    elif dataset == "oil_simple":
        # Onehot encodings for class labels (1 for Oil, 0 for No Oil)
        # Oil contaminated samples contain 'MO' in their class label.
        y = data['m/z'].apply(lambda x: [1,0] if 'MO' in x else [0,1])
    elif dataset == "oil_regression":
        # Regression outputs for the amount of oil contamination.
        y = data['m/z'].apply(lambda x:
                          0.5 if 'MO 50' in x
                    else (0.25 if 'MO 25' in x
                    else (0.1 if 'MO 10' in x
                    else (0.05 if 'MO 05' in x
                    else (0.01 if 'MO 01' in x
                    else (0.001 if 'MO 0.1' in x
                    else (0.0 if 'MO 0' in x
                    else 0.0)))))))
    elif dataset == "oil":
        # Onehpot encodings for class lables.
        # Class labels for different concentrations of mineral oil.
        y = data['m/z'].apply(lambda x:
                          [1,0,0,0,0,0,0] if 'MO 50' in x
                    else ([0,1,0,0,0,0,0] if 'MO 25' in x
                    else ([0,0,1,0,0,0,0] if 'MO 10' in x
                    else ([0,0,0,1,0,0,0] if 'MO 05' in x
                    else ([0,0,0,0,1,0,0] if 'MO 01' in x
                    else ([0,0,0,0,0,1,0] if 'MO 0.1' in x
                    else ([0,0,0,0,0,0,1] if 'MO 0' in x
                    else None )))))))  # Labels (0 for Hoki, 1 for Moki))
    elif dataset == "cross-species":
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
    return y

def remove_instances_with_none_labels(
        X: Iterable, 
        y: Iterable
    ) -> Union[np.array, np.array]:
    """ Removes any uneeded instances for downstream tasks.
    
    Args: 
        X (Iterable): the feature set.
        y (Iterable): the class labels.

    Returns 
        X (np.array): the feature set.
        y (np.array): the class labels.
    """

    xs = []
    ys = []
    for (x,y) in zip(X.to_numpy(),y):
        if y is not None:
            xs.append(x)
            ys.append(y)
    X = np.array(xs)
    y = np.array(ys)
    return X,y

def train_test_split_to_data_loader(
        X: Iterable, 
        y: Iterable, 
        is_data_augmentation: bool = False, 
        batch_size: int = 64
    )  -> Union[DataLoader, DataLoader, DataLoader, int, int]:
    """ Converts from a train_test_split to DataLoaders.
    
    Args: 
        X (Iterable): the feature set.
        y (Iterable): the class labels. 
        is_data_augmentation: Whether or not to perform data augementation. Defaults to False. 
        batch_size (int): The size of each batch in the DataLoader.
    
    Returns: 
        train_loader (DataLoader): the training set DataLoader.
        val_loader (DataLoader): the validation set DataLoader. 
        test_loader (TestLoader): the test set DataLoader.
    """
    # Evaluation parameters.
    train_split = 0.8

    # Step 2: Split your dataset into training, validation, and testing sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=(1-train_split))
    
    # Data augmentation - adding random noise.
    if is_data_augmentation:
        X_train, y_train = random_augmentation(X_train,y_train)

    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)

    assert train_dataset.samples.shape[0] == train_dataset.labels.shape[0], "train_dataset samples and labels should have same length."
    assert val_dataset.samples.shape[0] == val_dataset.labels.shape[0], "val_dataset samples and labels should have same length."

    # Step 4: Create PyTorch DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # calculate steps per epoch for training and validation set
    train_steps = len(train_loader.dataset) // batch_size
    val_steps = len(val_loader.dataset) // batch_size
    # when batch_size greater than dataset size, avoid division by zero.
    train_steps = max(1, train_steps)
    val_steps = max(1, val_steps)
    return train_loader, val_loader, train_steps, val_steps

    
def preprocess_dataset(
        dataset: str ="species", 
        is_data_augmentation: bool = True, 
        batch_size: int = 64,
        is_pre_train = False
    ) -> Union[DataLoader, DataLoader]:
    """Preprocess the dataset for the downstream task of pre-training.
    
    If pre-training, include quality control, mixed species, and oil contaminated instances.
    All these instances are included to inflate the number of training instances for pre-training.
    Otherwise, discard these values.
    
    Args: 
        dataset (str): Fish species, part, oil or cross-species. Defaults to species.
        is_data_augmentation (bool): Conditional flag to perform data augmentation, or not.
        batch_size (int): The batch_size for the DataLoaders.
        is_pre_train (bool): Flag to specify if dataset is being loaded for pre-training or training. Defaults to False.
    
    Returns:
        train_loader (DataLoader), : the training set. 
        val_loader (DataLoader), : the validation set.
        test_loader (DataLoader), : the test set.
        train_steps (int), : the number of training steps.
        val_steps (int), : the number of validation steps.
        data (pd.DataFrame): the dataframe storing the entire dataset.
    """
    data = load_from_file()
    # For pre-training, keep all instances.
    if not is_pre_train:
        data = filter_dataset(dataset=dataset, data=data)
    y = one_hot_encoded_labels(dataset=dataset, data=data)
    X = data.drop('m/z', axis=1)
    X,y = remove_instances_with_none_labels(X,y)
    train_loader, val_loader, train_steps, val_steps = train_test_split_to_data_loader(
        X,
        y,
        is_data_augmentation=is_data_augmentation,
        batch_size=batch_size
    )
    return train_loader, val_loader