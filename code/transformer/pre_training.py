import logging
from tqdm import tqdm
import random
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformer import Transformer
from typing import Union, Optional

def pre_train_masked_spectra(
        model: Transformer, 
        num_epochs: int = 100, 
        train_loader: DataLoader = None, 
        val_loader: DataLoader = None, 
        file_path: str = "transformer_checkpoint.pth", 
        device: Optional[Union[str, torch.device]] = None,
        criterion: CrossEntropyLoss = None,
        optimizer: AdamW = None,
        mask_prob: float = 0.2,
        n_features = 1023
    ) -> Transformer:
    """ Masked spectra modelling.

    Randomly masks spectra with a given probability, and pre-trains the model in predicting those masked spectra.

    Args: 
        model (Transformer): the nn.Module for the transformer.
        num_epochs (int): The number of epochs to pre-train for. Defaults to 100.
        train_loader (DataLoader): the torch DataLoader containing the training set.
        val_loader (DataLoader) the torch DataLoader containing the validation set.
        file_path (str): the file path to store the model checkpoints to. Defaults to "transformer_checkpoint.pth"
        device (str, torch,device): the device to perform the operations on. Defaults to None.
        criterion (CrossEntropyLoss): the cross entropy loss function to measure loss by.
        optimizer (AdamW): the AdamW optimizer to perform gradient descent with.
        mask_prob (float): the probability of masking a spectra. Defaults to 0.2
        n_features (int): the number of features. Defaults to 1023.

    Returns:
        model (Transformer): returns the pre-trained model.
    """
    
    logger = logging.getLogger(__name__)
    
    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Pre-training: Masked Spectra Modelling"):
        total_loss = 0.0
        model.train()

        for (x,_) in train_loader:
            # Generate batch of data
            tgt_x, x = x.to(device), x.to(device)

            batch_size = x.shape[0]
            mask = torch.rand(batch_size, n_features) < mask_prob
            mask = mask.to(device)
            x[mask] = 0

            optimizer.zero_grad()
            outputs  = model(x, x)
            loss = criterion(outputs, tgt_x)  # Compare predicted spectra with true spectra
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        total_val_loss = 0.0
        model.eval()
        for (x,_) in val_loader:
            tgt_x, x = x.to(device), x.to(device)

            val_batch_size = x.shape[0]
            mask = torch.rand(val_batch_size, n_features) < mask_prob
            mask = mask.to(device)
            x[mask] = 0

            outputs = model(x, x)
            val_loss = criterion(outputs, tgt_x)
            total_val_loss += val_loss.item()

        # Print average loss for the epoch
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/batch_size:.4f}, Val: {val_loss/val_batch_size:.4f}')

    masked_spectra_prediction = model
    torch.save(masked_spectra_prediction.state_dict(), file_path)
    return model


def mask_left_side(
        input_spectra: torch.Tensor, 
        quarter: bool = True
    ) -> torch.Tensor:
    """
    Masks the left-hand side of the input spectra tensor.

    Args:
        input_spectra (torch.Tensor): Input spectra tensor of shape (batch_size, 1023).

    Returns:
        torch.Tensor: Masked input spectra tensor.
    """
    # Calculate the index to split the tensor
    split_index = input_spectra.shape[0] // 2
    # Mask the left half of the input tensor
    input_spectra[:split_index] = 0
    return input_spectra

def mask_right_side(
        input_spectra: torch.Tensor
    ) -> torch.Tensor:
    """
    Masks the right-hand side of the input spectra tensor.

    Args:
        input_spectra (torch.Tensor): Input spectra tensor of shape (batch_size, 1023).

    Returns:
        torch.Tensor: Masked input spectra tensor.
    """
    # Calculate the index to split the tensor
    split_index = input_spectra.shape[0] // 2
    # Mask the left half of the input tensor
    input_spectra[split_index:] = 0
    return input_spectra


def pre_train_model_next_spectra(
        model: Transformer, 
        num_epochs: int = 100, 
        train_loader: DataLoader = None, 
        val_loader: DataLoader = None, 
        file_path: str = "transformer_checkpoint.pth", 
        device: Optional[Union[str, torch.device]] = None,
        criterion: CrossEntropyLoss = None,
        optimizer: AdamW = None
    ) -> Transformer:
    """
    Pre-trains the model with Next Spectra Prediction (NSP).
    This is a variant of Next Sentence Prediction (NSP) from (Devlin 2018).

    Args:
        model (torch.nn.Module): The pre-trained model.
        num_epochs (int): The number of epochs to pre-train for. Defaults to 100.
        train_loader (DataLoader): the torch DataLoader for the training set.
        val_loader (DataLoader): the torch DataLoader for the validation set.
        file_path (str): The path to save the model weights.
        device (str, torch,device): the device to perform the operations on. Defaults to None.
        criterion (CrossEntropyLoss): the cross entropy loss function to measure loss by.
        optimizer (AdamW): the AdamW optimizer to perform gradient descent with.

    Returns: 
        model (Transformer): the pre-trained model
    """
    logger = logging.getLogger(__name__)
    # Generate the training set of contrastive pairs.
    X_train = []
    y_train = []
    # Iterate over batches in the training loader.
    for (x,_) in train_loader:
        # Randomly choose pairs of adjacent spectra from the same index or different indexes
        for i in range(len(x)):
            if random.random() < 0.5:
                # Choose two adjacent spectra from the same index
                if i < len(x) - 1:
                    # Mask the right side of the spectra
                    left = mask_right_side(x[i])
                    right = mask_left_side(x[i])
                    X_train.append((left, right))
                    y_train.append([1,0])
            else:
                # Choose two spectra from different indexes
                j = random.randint(0, len(x) - 1)
                # Exhaustive search for two different indexes.
                while (j == i):
                    j = random.randint(0, len(x) - 1)
                left = mask_right_side(x[i])
                right = mask_left_side(x[j])
                X_train.append((left, right))
                y_train.append([0,1])

    # Generate the validation set of contrastive pairs.
    X_val = []
    y_val = []
    # Iterate over batches in the validation loader.
    for (x,_) in val_loader: 
        # Randomly choose pairs of adjacent spectra from the same index or different indexes
        for i in range(len(x)):
            if random.random() < 0.5:
                # Choose two adjacent spectra from the same index
                if i < len(x) - 1:
                    # Mask the right side of the spectra
                    left = mask_right_side(x[i])
                    right = mask_right_side(x[i])
                    X_val.append((left, right))
                    y_val.append([0,1])
            else:
                # Choose two spectra from different indexes
                j = random.randint(0, len(x) - 1)
                # Exhaustive search for two different indexes.
                while (j == i):
                    j = random.randint(0, len(x) - 1)
                left = mask_left_side(x[i])
                right = mask_right_side(x[j])
                X_val.append((left, right))
                y_val.append([1,0])


    for epoch in tqdm(range(num_epochs), desc="Pre-training: Next Spectra Prediction"):
        model.train()
        total_loss = 0.0

        for (left, right), label in zip(X_train, y_train):
            # Forward pass
            left = left.to(device)
            right = right.to(device)
            label = torch.tensor(label).to(device)

            optimizer.zero_grad()
            output = model(left.unsqueeze(0), right.unsqueeze(0))
            label = label.float()
            
            loss = criterion(output, label.unsqueeze(0))
            total_loss += loss.item()
            # Backpropagation
            loss.backward()
            optimizer.step()

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(X_train)
        
        model.eval()
        val_total_loss = 0.0

        for (left, right), label in zip(X_val, y_val):
            # Forward pass
            left = left.to(device)
            right = right.to(device)
            label = torch.tensor(label).to(device)

            optimizer.zero_grad()
            output = model(left.unsqueeze(0), right.unsqueeze(0))
            label = label.float()   

            loss = criterion(output, label.unsqueeze(0))
            val_total_loss += loss.item()

        val_avg_loss = val_total_loss / len(X_val)
        logger.info(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f} Validation: {val_avg_loss:.4f}")

    next_spectra_model = model
    torch.save(next_spectra_model.state_dict(), file_path)
    return model


def pre_train_transfer_learning(
        model: Transformer, 
        file_path: str = 'transformer_checkpoint.pth', 
        output_dim: int = 2
    ) -> Transformer:
    """ Loads the weights from a pre-trained model.

    This method handles the differences in dimensions for the output dimension between pre-training and training tasks.

    Args: 
        model (Transformer): the model to load the pre-trained weights to
        file_path (str): the filepath where the checkpoint is stored.
        output_dim (int): the number of classes for the output dimension. Defaults to 2 for next spectra prediction.

    Returns:
        model (Transformer): the model is returned with the pre-trained weights loaded into it.
    """
    # Load the state dictionary from the checkpoint.
    checkpoint = torch.load(file_path)
    # Modify the 'fc.weight' and 'fc.bias' parameters
    checkpoint['fc.weight'] = checkpoint['fc.weight'][:output_dim]  # Keep only the first 2 rows
    checkpoint['fc.bias'] = checkpoint['fc.bias'][:output_dim] # Keep only the first 2 elements
    # Load the modified state dictionary into the model.
    model.load_state_dict(checkpoint, strict=False)
    return model 