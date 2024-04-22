import logging
import numpy as np
from tqdm import tqdm
import random
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from util import EarlyStopping
from transformer import Transformer


def pre_train_masked_spectra(model, 
                            num_epochs=100, 
                            train_loader=None, 
                            val_loader=None, 
                            file_path="transformer_checkpoint.pth", 
                            device = None,
                            criterion=None,
                            optimizer=None,
                            is_early_stopping=False,
                            early_stopping = None,
                            mask_prob=0.2):
    
    logger = logging.getLogger(__name__)
    
    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Pre-training: Masked Spectra Modelling"):
        total_loss = 0.0
        model.train()

        for (x,y) in train_loader:
            # Generate batch of data
            tgt_x, x = x.to(device), x.to(device)

            batch_size = x.shape[0]
            mask = torch.rand(batch_size, 1023) < mask_prob
            mask = mask.to(device)
            x[mask] = 0

            optimizer.zero_grad()
            outputs = model(x, x)
            loss = criterion(outputs, tgt_x)  # Compare predicted spectra with true spectra
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        total_val_loss = 0.0
        model.eval()
        for (x,y) in val_loader:
            tgt_x, x = x.to(device), x.to(device)

            val_batch_size = x.shape[0]
            mask = torch.rand(val_batch_size, 1023) < mask_prob
            mask = mask.to(device)
            x[mask] = 0

            outputs = model(x, x)
            val_loss = criterion(outputs, tgt_x)
            total_val_loss += val_loss.item()

        # Early stopping (Morgan 1989)
        if is_early_stopping:
            # Check if early stopping criteria met
            early_stopping(1, val_loss, model)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

        # Print average loss for the epoch
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/batch_size:.4f}, Val: {val_loss/val_batch_size:.4f}')

    masked_spectra_prediction = model
    torch.save(masked_spectra_prediction.state_dict(), file_path)
    return model


def mask_left_side(input_spectra, mask_prob=0.5):
    """
    Masks the left-hand side of the input spectra tensor.

    Args:
        input_spectra (torch.Tensor): Input spectra tensor of shape (batch_size, 1023).
        mask_prob (float): Probability of masking each element of the left-hand side.

    Returns:
        torch.Tensor: Masked input spectra tensor.
    """
    # Calculate the index to split the tensor
    split_index = input_spectra.shape[0] // 2
    # Mask the left half of the input tensor
    input_spectra[:split_index] = 0
    return input_spectra

def mask_right_side(input_spectra, mask_prob=0.5):
    """
    Masks the right-hand side of the input spectra tensor.

    Args:
        input_spectra (torch.Tensor): Input spectra tensor of shape (batch_size, 1023).
        mask_prob (float): Probability of masking each element of the right-hand side.

    Returns:
        torch.Tensor: Masked input spectra tensor.
    """
    # Calculate the index to split the tensor
    split_index = input_spectra.shape[0] // 2
    # Mask the left half of the input tensor
    input_spectra[split_index:] = 0
    return input_spectra


def pre_train_model_next_spectra(model, 
                            num_epochs=100, 
                            train_loader=None, 
                            val_loader=None, 
                            file_path="transformer_checkpoint.pth", 
                            device = None,
                            criterion=None,
                            optimizer=None,
                            is_early_stopping=False,
                            early_stopping = None):
    """
    Pre-trains the model with Next Spectra Prediction (NSP).
    This is a variant of Next Sentence Prediction (NSP) from (Devlin 2018).

    Args:
        model (torch.nn.Module): The pre-trained model.
        file_path (str): The path to save the model weights.
    """
    logger = logging.getLogger(__name__)
    # Assume train_loader is your DataLoader containing spectra data
    for epoch in tqdm(range(num_epochs), desc="Pre-training: Next Spectra Prediction"):
        model.train()
        total_loss = 0.0
        num_pairs = 0

        # Iterate over batches in the train_loader
        for (x,y) in train_loader:
            # Randomly choose pairs of adjacent spectra from the same index or different indexes
            pairs = []
            labels = []
            for i in range(len(x)):
                if random.random() < 0.5:
                    # Choose two adjacent spectra from the same index
                    if i < len(x) - 1:
                        # Mask the right side of the spectra
                        left = mask_left_side(x[i])
                        right = mask_right_side(x[i])
                        pairs.append((left, right))
                        labels.append([0,1])
                else:
                    # Choose two spectra from different indexes
                    j = random.randint(0, len(x) - 1)
                    if j != i:
                        left = mask_left_side(x[i])
                        right = mask_right_side(x[j])
                        pairs.append((left, right))
                        labels.append([1,0])

            for (input_spectra, target_spectra), label in zip(pairs, labels):
                # Forward pass
                input_spectra = input_spectra.to(device)
                target_spectra = target_spectra.to(device)
                label = torch.tensor(label).to(device)

                optimizer.zero_grad()
                output = model(input_spectra.unsqueeze(0), target_spectra.unsqueeze(0))
                label = label.float()

                loss = criterion(output, label.unsqueeze(0))
                total_loss += loss.item()
                # Backpropagation
                loss.backward()
                optimizer.step()
                num_pairs += 1

        # Calculate average loss for the epoch
        avg_loss = total_loss / num_pairs

        model.eval()
        val_total_loss = 0.0
        num_pairs = 0

        for (x,y) in val_loader:# Randomly choose pairs of adjacent spectra from the same index or different indexes
            pairs = []
            labels = []

            for i in range(len(x)):
                if random.random() < 0.5:
                    # Choose two adjacent spectra from the same index
                    if i < len(x) - 1:
                        # Mask the right side of the spectra
                        left = mask_left_side(x[i])
                        right = mask_right_side(x[i])
                        pairs.append((left, right))
                        labels.append([0,1])
                else:
                    # Choose two spectra from different indexes
                    j = random.randint(0, len(x) - 1)
                    if j != i:
                        left = mask_left_side(x[i])
                        right = mask_right_side(x[j])
                        pairs.append((left, right))
                        labels.append([1,0])

            for (input_spectra, target_spectra), label in zip(pairs, labels):
                # Forward pass
                input_spectra = input_spectra.to(device)
                target_spectra = target_spectra.to(device)
                label = torch.tensor(label).to(device)

                optimizer.zero_grad()
                output = model(input_spectra.unsqueeze(0), target_spectra.unsqueeze(0))
                label = label.float()

                loss = criterion(output, label.unsqueeze(0))
                val_total_loss += loss.item()
                num_pairs += 1

        num_pairs = max(1, num_pairs)
        val_avg_loss = total_loss / num_pairs
        logger.info(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f} Validation: {val_avg_loss:.4f}")

        # Early stopping (Morgan 1989)
        if is_early_stopping:
            # Check if early stopping criteria met
            early_stopping(1, val_avg_loss, model)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

    next_spectra_model = model
    torch.save(next_spectra_model.state_dict(), file_path)
    return model


def transfer_learning(model, file_path='transformer_checkpoint.pth', output_dim=2):
    # Load the state dictionary from the checkpoint.
    checkpoint = torch.load(file_path)
    # Modify the 'fc.weight' and 'fc.bias' parameters
    checkpoint['fc.weight'] = checkpoint['fc.weight'][:output_dim]  # Keep only the first 2 rows
    checkpoint['fc.bias'] = checkpoint['fc.bias'][:output_dim] # Keep only the first 2 elements
    # Load the modified state dictionary into the model.
    model.load_state_dict(checkpoint, strict=False)

    return model