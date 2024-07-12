
import logging
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from kan import KAN
from plot import plot_confusion_matrix, plot_accuracy
from typing import Union


def train_model(
        model: KAN, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        criterion: nn.CrossEntropyLoss,
        optimizer: optim.AdamW, 
        num_epochs: int = 100, 
        patience: int  = 10
    ) -> KAN:
    """ Train the model
    
    Args: 
        model (KAN): the model to train.
        train_loader (DataLoader): the training set.
        val_loader (DataLoader): the validation set.
        criterion (nn.CrossEntropyLoss): the loss function. Defaults to CrossEntropyLoss.
        optimizer (optim.AdamW): the optimizer. Defaults to AdamW.
        num_epochs (int): the number of epochs to train for.
        patience (int): the patience for the early stopping mechanism.

    Returns:
        model (KAN): the trained model - with early stopping - that has best validation performance.
    """
   # Logging output to a file.
    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    training_accuracies = []
    training_losses = []
    validation_accuracies = []
    validation_losses = []

    best_val_acc = float('-inf')
    epochs_without_improvement = 0
    best_model = None

    for epoch in (pbar := tqdm(range(num_epochs), desc="Training")):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

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
            train_correct += predicted.eq(actual).sum().item()
            train_total += y.size(0)

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total

        # Telemetry for loss curve.
        training_accuracies.append(train_acc)
        training_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)

                val_loss += loss.item() * x.size(0)
                _, predicted = outputs.max(1)
                _, actual = y.max(1)
                val_correct += predicted.eq(actual).sum().item()
                val_total += y.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total

        # Telemetry for loss curve.
        validation_accuracies.append(val_acc)
        validation_losses.append(val_loss)

        message = f'Epoch {epoch+1}/{num_epochs} \tTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}\t Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}'
        pbar.set_description(message)
        logger.info(message)

        # Early stopping
        if train_acc == 1:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                best_model = model.state_dict()
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    message = f'Early stopping triggered after {epoch + 1} epochs'
                    logger.info(message)
                    print(message)
                    print(f"Validation accuracy: {best_val_acc}")
                    break
        else: 
            epochs_without_improvement = 0

    # Plot the loss curve.
    plot_accuracy(
        train_losses=training_losses, 
        val_losses=validation_losses, 
        train_accuracies=training_accuracies, 
        val_accuracies=validation_accuracies
    ) 

    # Retrieve weights for the model that performs best on the validation set.
    if best_model is not None:
        model.load_state_dict(best_model)
    return model


def evaluate_model(
        model : KAN, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        dataset: str, 
        device: Union[torch.device,str]
    ) -> None:
    """Evaluate the model on the training and evaluation datasets.

    Args: 
        model (KAN): the model to evaluate.
        train_loader (DataLoader): the training set. 
        val_loader (DataLoader): the validation set.
        dataset (str): the name of the dataset to be evaluated.
        device: (torch.device, str): the device to evaluate the model/dataset on.
    """
    # Logging output to a file.
    logger = logging.getLogger(__name__)
    
    model.eval()
    # switch off autograd
    with torch.no_grad():
        # loop over the test set
        datasets = [("train", train_loader), ("validation", val_loader)]
        for name, dataset_x_y in datasets:
            startTime = time.time()
            # finish measuring how long training too
            for (x,y) in dataset_x_y:
                (x,y) = (x.to(device), y.to(device))
                pred = model(x)
                test_correct = (pred.argmax(1) == y.argmax(1)).sum().item()
                accuracy = test_correct / len(x)
                logger.info(f"{name} got {test_correct} / {len(x)} correct, accuracy: {accuracy}")
                plot_confusion_matrix(dataset, name, y.argmax(1).cpu(), pred.argmax(1).cpu())
            endTime = time.time()
            logger.info(f"Total time taken evaluate on {name} set the model: {(endTime - startTime):.2f}s")