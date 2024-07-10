import logging
from tqdm import tqdm 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from vae import vae_classifier_loss
from vae import VAE 
from plot import plot_confusion_matrix, plot_accuracy
from typing import Union


def train(
        model: VAE, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        num_epochs: int, 
        alpha: float = 0.1, 
        beta: float = 0.9, 
        device: Union[torch.device, str] = 'cpu',
        optimizer=optim.AdamW
    ) -> VAE:
    """ Train the VAE model.

    Args: 
        model (VAE): The VAE model to train.
        data_loader (DataLoader): The data loader.
        num_epochs (int): The number of epochs to train.
        alpha (float): The weight for the reconstruction loss.
        beta (float): The weight for the KLD loss.
        device (torch.device, str): The device to train on.
        optimizer (torch.optim): The optimizer to use.

    Returns: 
        model (VAE): The trained VAE model.
    """
    logger = logging.getLogger(__name__)
    model.train()

    training_accuracies = []
    training_losses = []
    validation_accuracies = []
    validation_losses = []

    best_val_acc = float('-inf')
    epochs_without_improvement = 0
    best_model = None

    for epoch in (pbar := tqdm(range(num_epochs), desc="Training")):
        model.train()
        train_acc = 0
        train_loss = 0
        for x, y in train_loader:
            # Move data to GPU
            x = x.float().to(device)
            y = y.long().to(device)
            
            # Forward pasbatchs
            recon_batch, mu, logvar, class_probs = model(x)
            loss = vae_classifier_loss(recon_batch, x, mu, logvar, class_probs, y, alpha, beta)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Classification accuracy.
            _, predicted = torch.max(class_probs, 1)
            _, actual = torch.max(y, 1)
            correct = (predicted == actual).sum().item()
            total = y.size(0)
            train_acc += correct / total
        
        train_acc = train_acc / len(train_loader)
        training_accuracies.append(train_acc)  
        train_loss = train_loss / len(train_loader)      
        training_losses.append(train_loss)
            
        model.eval()
        with torch.no_grad():
            correct = 0
            val_acc = 0
            val_loss = 0
            for x, y in val_loader:
                x = x.float().to(device)
                y = y.long().to(device)
                recon_batch, mu, logvar, class_probs = model(x)
                loss = vae_classifier_loss(recon_batch, x, mu, logvar, class_probs, y, alpha, beta)
                val_loss += loss.item()
                _, predicted = torch.max(class_probs, 1)
                _, actual = torch.max(y, 1)

                total += y.size(0)
                correct += (predicted == actual).sum().item()
                val_acc += correct / total
            
            val_acc = val_acc / len(val_loader)
            validation_accuracies.append(val_acc)
            val_loss = val_loss / len(val_loader)
            validation_losses.append(val_loss)
        
        # Print average loss for the epoch(val_loader
        message = f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f} \t Train Accuracy: {train_acc:.4f} \t Val Loss: {val_loss:.4f} \t Val Accuracy: {val_acc:.4f}'
        pbar.set_description(message)
        logger.info(message)

    plot_accuracy(
        training_losses, 
        validation_losses, 
        training_accuracies, 
        validation_accuracies
    )

    return model

# Function to get encoded representation and class prediction
def encode_and_classify(
        model: VAE, 
        data: torch.Tensor, 
        device: Union[torch.device, str]
    ) -> [torch.Tensor, torch.Tensor]:
    """ Encode the input data and predict the class.

    Args:
        model (VAE): The trained VAE model.
        data (torch.Tensor): The input data.
        device (torch.device, str): The device to run the inference on.
    
    Returns: 
        z (torch.Tensor): The encoded representation.
        class_probs (torch.Tensor): The class probabilities.

    """
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        mu, logvar = model.encode(data)
        z = model.reparameterize(mu, logvar)
        class_probs = F.softmax(model.classifier(z), dim=1)
    return z.cpu(), class_probs.cpu()

# Function to generate new samples of a specific class
def generate(
        model: VAE, 
        num_samples: int, 
        target_class: int, 
        device: Union[torch.device, str]
    ) -> torch.Tensor:
    """ Generate new samples of a specific class.

    Args:
        model (VAE): The trained VAE model.
        num_samples (int): The number of samples to generate.
        target_class (int): The target class to generate samples for.
        device (torch.device, str): The device to run the generation on.

    Returns:
        samples (torch.Tensor): The generated samples.
    """
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        c = F.one_hot(torch.tensor([target_class] * num_samples), num_classes=model.num_classes).float().to(device)
        samples = model.decode(z, c)
    return samples.cpu()

def evaluate_classification(
        model: VAE, 
        data_loader: DataLoader, 
        train_val_test: str, 
        dataset: str = "species",
        device: Union[torch.device, str] = "cpu"   
    ) -> None:
    """ Evaluate the classification accuracy of the VAE on a dataset.

    Args: 
        model (VAE): The trained VAE model.
        data_loader (DataLoader): The data loader for the dataset.
        train_val_test (str):  Specify if training, validation or test set.
        dataset (str): The dataset to evaluate on. Default is "species".
        device (torch.device, str): The device to run the evaluation on.
    """
    logger = logging.getLogger(__name__)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            # Move data to GPU
            x = x.float().to(device)
            y = y.long().to(device)
            
            # Forward pass to get class probabilities
            _, _, _, class_probs = model(x)
            
            # Get the predicted class
            _, predicted = torch.max(class_probs, 1)
            _, actual = torch.max(y, 1)
            
            # Update total and correct counts
            total += y.size(0)
            correct += (predicted == actual).sum().item()

            plot_confusion_matrix(
                dataset=dataset, 
                name=train_val_test, 
                actual=actual, 
                predicted=predicted,
                color_map = "Blues"
            )
    
    # Calculate accuracy
    accuracy = 100 * correct / total
    message = f'{dataset} classification Accuracy: {accuracy:.2f}%'
    print(message)
    logger.info(message)