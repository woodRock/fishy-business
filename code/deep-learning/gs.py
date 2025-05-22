# Grid Search - gs.py 

from transformer import Transformer 
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import make_scorer, balanced_accuracy_score
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from util import create_data_module

def load_data(
    dataset: str = "species"
) -> (TensorDataset, list):
    """ 
    Load the specified dataset and scale the features using StandardScaler.

    Args: 
        dataset (str): The name of the dataset to load. One of ["species", "part", "oil", "cross-species"].

    Returns: 
        scaled_dataset (TensorDataset): The scaled dataset.
        targets (list): The target labels.
    """
    data_module = create_data_module(
        dataset_name=dataset,
        batch_size=32,
    )

    data_loader, _ = data_module.setup()
    dataset = data_loader.dataset
    features = torch.stack([sample[0] for sample in dataset])
    labels = torch.stack([sample[1] for sample in dataset])

    features_np = features.numpy()
    scaler = StandardScaler()
    scaled_features_np = scaler.fit_transform(features_np)
    scaled_features = torch.tensor(scaled_features_np, dtype=torch.float32)
    
    scaled_dataset = TensorDataset(scaled_features, labels)
    targets = [sample[1].argmax(dim=0) for sample in dataset]
    
    return scaled_dataset, targets

def grid_search(
    dataset: str = "species",
    param_grid: dict = None,
    n_splits: int = 5,
    random_state: int = 42
) -> dict:
    """ 
    Perform grid search for hyperparameter tuning.

    Args: 
        dataset (str): The name of the dataset to load. One of ["species", "part", "oil", "cross-species"].
        param_grid (dict): The parameter grid for grid search.
        n_splits (int): Number of splits for cross-validation.
        random_state (int): Random state for reproducibility.

    Returns: 
        best_params (dict): The best parameters found during grid search.
    """
    if param_grid is None: 
        param_grid = {
            'num_heads': [1, 2, 4],
            'num_layers': [1, 2],
            'hidden_dim': [32, 64],
            'dropout': [0.1, 0.2]
        }
    # Load the dataset
    dataset, targets = load_data(dataset=dataset)
    # Create the model
    model = Transformer(
        input_dim=dataset[0][0].shape[0],
        output_dim=len(set(targets)),
        num_heads=1,
        hidden_dim=32,
        num_layers=1,
        dropout=0.1
    )
    # Create the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    # Create the scorer
    scorer = make_scorer(balanced_accuracy_score)
    # Create the grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=n_splits,
        n_jobs=-1,
        verbose=1
    )
    # Fit the grid search
    grid_search.fit(dataset, targets)
    # Get the best parameters
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score}")
    return best_params

if __name__ == "__main__":
    dataset = "species"
    best_params = grid_search(dataset=dataset)
    print(f"Best parameters for {dataset} dataset: {best_params}")