from tqdm import tqdm
import torch
from plot import plot_accuracy
import logging

def train(model, dataloader, criterion, optimizer, device):
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

def evaluate(model, dataloader, criterion, device):
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

def train_model(model, 
                num_epochs=100, 
                train_loader=None, 
                val_loader=None, 
                device = None,
                criterion=None,
                optimizer=None,
                is_early_stopping=False,
                early_stopping = None,
                file_path="transformer_checkpoint.pth"):
    
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

def transfer_learning(dataset, model, file_path='transformer_checkpoint.pth'):
    if dataset == "species":
        # There are 2 classes in the fish species dataset.
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

    # Load the modified state dictionary into the model.
    model.load_state_dict(checkpoint, strict=False)

    return model