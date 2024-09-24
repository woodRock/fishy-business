import os 
import numpy as np
import pandas as pd
import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split arrays or matrices into random train and test subsets.
    
    Parameters:
    X : array-like, shape (n_samples, n_features)
        The data to be split.
    y : array-like, shape (n_samples,)
        The target variable.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default=None
        Controls the shuffling applied to the data before applying the split.
    
    Returns:
    X_train, X_test, y_train, y_test : arrays
        The train and test subsets of X and y.
    """
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Ensure X and y have the same number of samples
    assert len(X) == len(y), "X and y must have the same number of samples"
    
    # Calculate the number of test samples
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Create an array of indices and shuffle it
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Split the indices into train and test sets
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Use the indices to create train and test sets
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return (X_train, y_train) , (X_test, y_test)

def preprocess_data(path = os.path.join("~","Desktop", "fishy-business", "data", "REIMS_data.xlsx")):
    # Import the dataset.
    data = pd.read_excel(path, header=1)
    # Filter out the quality control, hoki-mackerel mix, and body parts samples.
    data = data[~data.iloc[:, 0].astype(str).str.contains('QC|HM|MO|fillet|frames|gonads|livers|skins|guts|frame|heads', case=False, na=False)]
    # Remove the class label column.
    X = data.iloc[:, 1:].to_numpy() 
    # Take only the class label column.
    y = data.iloc[:, 0].to_numpy()

    # Preprocessing
    # Center the data
    X = X - X.mean()
    # Normalize between 0 and 1
    X = (X - X.min()) / (X.max() - X.min())

    features = list() 
    labels = list() 

    for i, (x_1, x_2) in enumerate(zip(X, X[1:])):
        concatenated = np.concatenate((x_1, x_2))
        features.append(concatenated)
        label = int(y[i] == y[i+1])
        labels.append(label)

    # Convert to numpy arrays.
    X,y = np.array(features), np.array(labels)
    # Train test split
    (X_train, y_train) , (X_test, y_test) = train_test_split(X,y)
    # One hot encoding 
    # y_train, y_test = np.eye(2)[y_train], np.eye(2)[y_test]
    return (X_train, y_train) , (X_test, y_test)
