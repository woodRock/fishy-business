import logging
from tqdm import tqdm
import random
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from cnn import CNN
from typing import Union, Optional

def pre_train_masked_spectra(
        model: CNN, 
        num_epochs: int = 100, 
        train_loader: DataLoader = None, 
        val_loader: DataLoader = None, 
        file_path: str = "transformer_checkpoint.pth", 
        device: Optional[Union[str, torch.device]] = None,
        criterion: CrossEntropyLoss = None,
        optimizer: AdamW = None,
        mask_prob: float = 0.2
    ) -> CNN:
    """ Masked spectra modelling.

    Randomly masks spectra with a given probability, and pre-trains the model in predicting those masked spectra.

    Args: 
        model (CNN): the nn.Module for the CNN.
        num_epochs (int): The number of epochs to pre-train for. Defaults to 100.
        train_loader (DataLoader): the torch DataLoader containing the training set.
        val_loader (DataLoader) the torch DataLoader containing the validation set.
        file_path (str): the file path to store the model checkpoints to. Defaults to "transformer_checkpoint.pth"
        device (str, torch,device): the device to perform the operations on. Defaults to None.
        criterion (CrossEntropyLoss): the cross entropy loss function to measure loss by.
        optimizer (AdamW): the AdamW optimizer to perform gradient descent with.
        mask_prob (float): the probability of masking a spectra. Defaults to 0.2

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
            mask = torch.rand(batch_size, 1023) < mask_prob
            mask = mask.to(device)
            x[mask] = 0

            optimizer.zero_grad()
            outputs  = model(x)
            loss = criterion(outputs, tgt_x)  # Compare predicted spectra with true spectra
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        total_val_loss = 0.0
        model.eval()
        for (x,_) in val_loader:
            tgt_x, x = x.to(device), x.to(device)

            val_batch_size = x.shape[0]
            mask = torch.rand(val_batch_size, 1023) < mask_prob
            mask = mask.to(device)
            x[mask] = 0

            outputs = model(x)
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


def pre_train_transfer_learning(
        model: CNN, 
        file_path: str = 'transformer_checkpoint.pth', 
        output_dim: int = 2
    ) -> CNN:
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
    print(f"checkpoint.keys :{checkpoint.keys()}")
    # Modify the 'fc.weight' and 'fc.bias' parameters
    checkpoint['fc_layers.3.weight'] = checkpoint['fc_layers.3.weight'][:output_dim]  # Keep only the first 2 rows
    checkpoint['fc_layers.3.bias'] = checkpoint['fc_layers.3.bias'][:output_dim] # Keep only the first 2 elements
    # Load the modified state dictionary into the model.
    model.load_state_dict(checkpoint, strict=False)
    return model 