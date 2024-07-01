import logging
from tqdm import tqdm
import torch
from plot import plot_accuracy
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from typing import Union, Optional, Iterable
from transformer import Transformer
from util import EarlyStopping


def train(
        model: Transformer, 
        dataloader: DataLoader, 
        criterion: CrossEntropyLoss, 
        optimizer: AdamW , 
        device: Optional[Union[str, torch.device]] = None
    ) -> Union[Iterable, Iterable]:
    """
    Evaluate the training performance.

    Args: 
        model (Transformer): the transformer model to evaluate training performance for.
        data_loader (DataLoader): the pytorch DataLoader for the training set.
        criterion (CrossEntropyLoss): the loss function to evaluate the training set with.
        optimizer (AdamW): the AdamW optimizer for gradient descent. 
        device (torch.device, str): the device to perform the training evaluation on.
    
    Returns: 
        epoch_loss (Iterable), : the loss for this epoch.
        epoch_accuracy (Iterable), : the accuracy for this epoch.
    """
    logger = logging.getLogger(__name__)

    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, inputs, src_mask=None, tgt_mask=None)  # Assuming no masking is needed for now
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == targets.argmax(1)).sum().item()
        total_samples += targets.size(0)

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    return epoch_loss, epoch_accuracy

def evaluate(
        model: Transformer, 
        dataloader: DataLoader, 
        criterion: CrossEntropyLoss, 
        device: Optional[Union[str, torch.device]] = None
    ) -> Union[Iterable, Iterable]:
    """Evaluate the validation performance.

    Args: 
        model (Transformer): the transformer model to evaluate.
        data_loader (DataLoader): the pytorch DataLoader for the validation set.
        criterion (CrossEntropyLoss): the loss function to evaluate the validation set with.
        device (torch.device, str): the device to perform the evaluation on.
    
    Returns: 
        epoch_loss (Iterable), : the loss for this epoch.
        epoch_accuracy (Iterable), : the accuracy for this epoch.
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs, inputs, src_mask=None, tgt_mask=None)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets.argmax(1)).sum().item()
            total_samples += targets.size(0)

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    return epoch_loss, epoch_accuracy

def train_model(
        model: Transformer, 
        num_epochs: int = 100, 
        train_loader: DataLoader = None, 
        val_loader: DataLoader = None, 
        device: Optional[Union[str, torch.device]]= None,
        criterion: CrossEntropyLoss = None,
        optimizer: AdamW = None,
        is_early_stopping: bool = False,
        early_stopping: EarlyStopping = None,
    ) -> Union[Iterable, Iterable, Iterable, Iterable]:
    """
    Train the transformer model.

    Args: 
        model (Transformer): the transformer model to be trained.
        num_epochs (int): the number of epochs to train it for.
        train_loader (DataLoader): pytorch DataLoader with the training set.
        val_loader (DataLoader): pytorch DataLaoder with the validation set.
        device (torch.device, str): the device to perform the computation on.
        criterion (CrossEntropyLoss): the loss function to optimize.
        optimizer (AdamW): the AdamW optimizer for gradient descent. 
        is_early_stopping (bool): whether or not early stopping is enabled.
        early_stopping (EarlyStopping): a class to checkpoint models with best validation loss.

    Returns:
        train_losses: Iterable, 
        train_accuracies: Iterable, 
        val_losses Iterable, 
        val_accuracies: Iterable,
    """
    logger = logging.getLogger(__name__)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in tqdm(range(num_epochs), desc="Training"):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

        # Early stopping (Morgan 1989)
        if is_early_stopping:
            # Check if early stopping criteria met
            early_stopping(train_accuracy, val_loss, model)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        logger.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

    plot_accuracy(train_losses, val_losses, train_accuracies, val_accuracies)

    return train_losses, train_accuracies, val_losses, val_accuracies

def transfer_learning(
        dataset: str, 
        model: Transformer, 
        file_path: str = 'transformer_checkpoint.pth'
    ) -> Transformer:
    """
    Transfer learning loads weights from a pre-training task.

    This method edits the final layer of the transformer to be ammenable to downstream tasks.

    Args":
        dataset (str): the dataset is either species, part, oil or cross-species.
        model (Transformer): the tranformer model to transfer the weights to.
        file_path (str): the file path to store the transformer checkpoint at.

    Returns:
        model (Transformer): the model with the pre-trained weights tranferred to it.
    """
    if dataset == "species":
        # There are 2 classes in the fish species, oil and cross-species dataset.
        output_dim = 2
        checkpoint = torch.load(file_path)
        checkpoint['fc.weight'] = checkpoint['fc.weight'][:output_dim]  # Keep only the first 2 rows
        checkpoint['fc.bias'] = checkpoint['fc.bias'][:output_dim] # Keep only the first 2 elements
        
    elif dataset == "part":
        # There are 6 classes in the fish parts dataset.
        output_dim = 6
        checkpoint = torch.load(file_path)
        checkpoint['fc.weight'] = torch.zeros(output_dim, checkpoint['fc.weight'].shape[1])
        checkpoint['fc.bias'] = torch.zeros(output_dim)

    elif dataset == "oil":
        # There are 7 classes in the fish oil dataset.
        output_dim = 7
        checkpoint = torch.load(file_path)
        checkpoint['fc.weight'] = torch.zeros(output_dim, checkpoint['fc.weight'].shape[1])
        checkpoint['fc.bias'] = torch.zeros(output_dim)
    
    elif dataset == "cross-species":
        output_dim = 3
        checkpoint = torch.load(file_path)
        checkpoint['fc.weight'] = torch.zeros(output_dim, checkpoint['fc.weight'].shape[1])
        checkpoint['fc.bias'] = torch.zeros(output_dim)

    else: 
        raise ValueError(f"Invalid dataset specified: {dataset}")

    # Load the modified state dictionary into the model.
    model.load_state_dict(checkpoint, strict=False)

    return model