import logging
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from lstm_with_attention import LSTM
from plot import plot_confusion_matrix, plot_accuracy
from typing import Union
from sklearn.metrics import balanced_accuracy_score
import numpy as np

def train_model(
        model: LSTM, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        criterion: nn.CrossEntropyLoss,
        optimizer: optim.AdamW, 
        num_epochs: int = 100, 
        patience: int  = 10
    ) -> LSTM:
    """ Train the model
    
    Args: 
        model (LSTM): the model to train.
        train_loader (DataLoader): the training set.
        val_loader (DataLoader): the validation set.
        criterion (nn.CrossEntropyLoss): the loss function. Defaults to CrossEntropyLoss.
        optimizer (optim.AdamW): the optimizer. Defaults to AdamW.
        num_epochs (int): the number of epochs to train for.
        patience (int): the patience for the early stopping mechanism.

    Returns:
        model (LSTM): the trained model - with early stopping - that has best validation performance.
    """
    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    training_balanced_accuracies = []
    training_losses = []
    validation_balanced_accuracies = []
    validation_losses = []

    best_val_balanced_acc = float('-inf')
    epochs_without_improvement = 0
    best_model = None

    for epoch in (pbar := tqdm(range(num_epochs), desc="Training")):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            _, actual = y.max(1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(actual.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_balanced_acc = balanced_accuracy_score(train_labels, train_preds)

        training_balanced_accuracies.append(train_balanced_acc)
        training_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)

                val_loss += loss.item() * x.size(0)
                _, predicted = outputs.max(1)
                _, actual = y.max(1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(actual.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_balanced_acc = balanced_accuracy_score(val_labels, val_preds)

        validation_balanced_accuracies.append(val_balanced_acc)
        validation_losses.append(val_loss)

        message = f'Epoch {epoch+1}/{num_epochs} \tTrain Loss: {train_loss:.4f}, Train Balanced Acc: {train_balanced_acc:.4f}\t Val Loss: {val_loss:.4f}, Val Balanced Acc: {val_balanced_acc:.4f}'
        pbar.set_description(message)
        logger.info(message)

        # Early stopping
        if val_balanced_acc > best_val_balanced_acc:
            best_val_balanced_acc = val_balanced_acc
            epochs_without_improvement = 0
            best_model = model.state_dict()
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                message = f'Early stopping triggered after {epoch + 1} epochs'
                logger.info(message)
                print(message)
                message = f"Best validation balanced accuracy: {best_val_balanced_acc:.4f}"
                logger.info(message)
                print(message)
                break

    # Plot the accuracy curve.
    plot_accuracy(
        train_losses=training_losses, 
        val_losses=validation_losses, 
        train_accuracies=training_balanced_accuracies, 
        val_accuracies=validation_balanced_accuracies
    ) 

    # Retrieve weights for the model that performs best on the validation set.
    if best_model is not None:
        model.load_state_dict(best_model)
    return model

def evaluate_model(
        model : LSTM, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        dataset: str, 
        device: Union[torch.device,str]
    ) -> None:
    """Evaluate the model on the training and evaluation datasets.

    Args: 
        model (LSTM): the model to evaluate.
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
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                _, predicted = pred.max(1)
                _, actual = y.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(actual.cpu().numpy())
            
            balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
            logger.info(f"{name} set balanced accuracy: {balanced_accuracy:.4f}")
            plot_confusion_matrix(dataset, name, np.array(all_labels), np.array(all_preds))
            endTime = time.time()
            logger.info(f"Total time taken to evaluate on {name} set: {(endTime - startTime):.2f}s")