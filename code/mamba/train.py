from tqdm import tqdm
import logging
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from mamba import Mamba
from plot import plot_accuracy
from typing import Union

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int = 100,
    patience: int = 10,
    n_splits: int = 5,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> nn.Module:
    logger = logging.getLogger(__name__)

    model_copy = copy.deepcopy(model)
    
    # Extract dataset from DataLoader
    dataset = train_loader.dataset
    
    # Get all labels for stratification
    all_labels = []
    for _, labels in dataset:
        if isinstance(labels, torch.Tensor):
            if labels.dim() > 1:
                # For multi-label classification, use argmax to get the primary label
                all_labels.append(labels.argmax().item())
            else:
                all_labels.append(labels.argmax())
        else:
            all_labels.append(labels)
    all_labels = np.array(all_labels)

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Lists to store metrics
    fold_train_losses = []
    fold_val_losses = []
    fold_train_accuracies = []
    fold_val_accuracies = []
    best_val_accuracies = []

    # Perform k-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), all_labels)):

        logger.info(f"Fold {fold}/{n_splits}")

        # Reset model to initial state
        model = copy.deepcopy(model_copy)

        # Create data loaders for this fold
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        fold_train_loader = DataLoader(train_subset, batch_size=train_loader.batch_size, shuffle=True)
        fold_val_loader = DataLoader(val_subset, batch_size=train_loader.batch_size, shuffle=True)

        # Initialize model, criterion, and optimizer
        model = model.to(device)
        criterion = criterion.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.001)

        # Training loop
        best_val_acc = float('-inf')
        epochs_without_improvement = 0
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(num_epochs):
            # Train
            model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []
            for inputs, labels in tqdm(fold_train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                _, actual = labels.max(1)
                train_preds.extend(predicted.cpu().numpy())
                train_labels.extend(actual.cpu().numpy())
            
            train_loss /= len(fold_train_loader)
            train_acc = balanced_accuracy_score(train_labels, train_preds)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            # Validate
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for inputs, labels in tqdm(fold_val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    _, actual = labels.max(1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(actual.cpu().numpy())
            
            val_loss /= len(fold_val_loader)
            val_acc = balanced_accuracy_score(val_labels, val_preds)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                best_model = model.state_dict()
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
                    # break

            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Store the best results for this fold
        fold_train_losses.append(train_losses)
        fold_val_losses.append(val_losses)
        fold_train_accuracies.append(train_accuracies)
        fold_val_accuracies.append(val_accuracies)
        best_val_accuracies.append(best_val_acc)

    # Calculate average performance across folds
    avg_train_losses = np.mean(fold_train_losses, axis=0)
    avg_val_losses = np.mean(fold_val_losses, axis=0)
    avg_train_accuracies = np.mean(fold_train_accuracies, axis=0)
    avg_val_accuracies = np.mean(fold_val_accuracies, axis=0)

    # Plot average performance
    plot_accuracy(
        train_losses=avg_train_losses,
        val_losses=avg_val_losses,
        train_accuracies=avg_train_accuracies,
        val_accuracies=avg_val_accuracies
    )

    logger.info(f"Average final validation accuracy: {avg_val_accuracies[-1]:.4f}")

    # Print the best validation accuracies
    for fold, val_acc in enumerate(best_val_accuracies, 1):
        print(f"{val_acc:.4f},")
    
    # Load the best model state
    model.load_state_dict(best_model)
    return model


def evaluate_model(
        model : Mamba, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        dataset: str, 
        device: Union[torch.device,str]
    ) -> None:
    """Evaluate the model on the training and evaluation datasets.

    Args: 
        model (CNN): the model to evaluate.
        train_loader (DataLoader): the training set. 
        val_loader (DataLoader): the validation set.
        dataset (str): the name of the dataset to be evaluated.
        device: (torch.device, str): the device to evaluate the model/dataset on.
    """
    logger = logging.getLogger(__name__)
    
    model.eval()
    with torch.no_grad():
        datasets = [("train", train_loader), ("validation", val_loader)]
        for name, data_loader in datasets:
            startTime = time.time()
            all_preds = []
            all_labels = []
            for x,y in data_loader:
                x,y = x.to(device), y.to(device)
                pred = model(x,x)
                _, predicted = pred.max(1)
                _, actual = y.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(actual.cpu().numpy())
            
            balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
            logger.info(f"{name} set balanced accuracy: {balanced_accuracy:.4f}")
            plot_confusion_matrix(dataset, name, np.array(all_labels), np.array(all_preds))
            endTime = time.time()
            logger.info(f"Total time taken to evaluate on {name} set: {(endTime - startTime):.2f}s")