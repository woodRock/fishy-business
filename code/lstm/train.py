import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from lstm_with_attention import LSTM


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
    # Logging output to a file.
    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_val_acc = float('-inf')
    epochs_without_improvement = 0
    best_model = None

    for epoch in (pbar := tqdm(range(num_epochs), desc="Training")):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            _, actual = labels.max(1)
            train_correct += predicted.eq(actual).sum().item()
            train_total += labels.size(0)

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                _, actual = labels.max(1)
                val_correct += predicted.eq(actual).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total

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

    if best_model is not None:
        model.load_state_dict(best_model)
    return model