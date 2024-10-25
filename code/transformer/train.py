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
from sklearn.metrics import precision_score, recall_score, f1_score, auc, roc_curve
from transformer import Transformer
from plot import plot_accuracy
from typing import Union

from tqdm import tqdm
import logging
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score, precision_score, recall_score, 
    f1_score, auc, roc_curve
)
from transformer import Transformer
from plot import plot_accuracy
from typing import Union, Dict, List, Tuple

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
    """Calculate multiple classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Probability predictions for ROC AUC (optional)
        
    Returns:
        Dictionary containing calculated metrics
    """
    metrics = {
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
    
    # Calculate ROC AUC if probabilities are provided
    if y_prob is not None:
        # For binary classification
        if y_prob.shape[1] == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            metrics['auc_roc'] = auc(fpr, tpr)
        # For multiclass, calculate macro average
        else:
            n_classes = y_prob.shape[1]
            # One-hot encode true labels
            y_true_onehot = np.eye(n_classes)[y_true]
            aucs = []
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_prob[:, i])
                aucs.append(auc(fpr, tpr))
            metrics['auc_roc'] = np.mean(aucs)
    
    return metrics

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
                all_labels.append(labels.argmax().item())
            else:
                all_labels.append(labels.argmax(dim=0))
        elif isinstance(labels, np.ndarray):
            if labels.ndim > 1:
                all_labels.append(np.argmax(labels))
            else:
                all_labels.append(np.argmax(labels))
        else:
            all_labels.append(labels)
    all_labels = np.array(all_labels)

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Lists to store metrics
    fold_metrics = {
        'train_losses': [], 'val_losses': [],
        'train_metrics': [], 'val_metrics': []
    }
    best_val_metrics = []

    # Perform k-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), all_labels), 1):
        model = copy.deepcopy(model_copy)
        logger.info(f"Fold {fold}/{n_splits}")

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
        best_val_f1 = float('-inf')
        epochs_without_improvement = 0
        epoch_metrics = {
            'train_losses': [], 'val_losses': [],
            'train_metrics': [], 'val_metrics': []
        }

        for epoch in range(num_epochs):
            # Train
            model.train()
            train_loss = 0.0
            train_preds = []
            train_probs = []
            train_labels = []
            
            for inputs, labels in tqdm(fold_train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs, inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                _, actual = labels.max(1)
                
                train_preds.extend(predicted.cpu().numpy())
                train_probs.extend(probs.detach().cpu().numpy())
                train_labels.extend(actual.cpu().numpy())
            
            train_loss /= len(fold_train_loader)
            train_metrics = calculate_metrics(
                np.array(train_labels), 
                np.array(train_preds),
                np.array(train_probs)
            )
            
            # Validate
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_probs = []
            val_labels = []
            
            with torch.no_grad():
                for inputs, labels in tqdm(fold_val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs, inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = outputs.max(1)
                    _, actual = labels.max(1)
                    
                    val_preds.extend(predicted.cpu().numpy())
                    val_probs.extend(probs.cpu().numpy())
                    val_labels.extend(actual.cpu().numpy())
            
            val_loss /= len(fold_val_loader)
            val_metrics = calculate_metrics(
                np.array(val_labels), 
                np.array(val_preds),
                np.array(val_probs)
            )

            # Store metrics
            epoch_metrics['train_losses'].append(train_loss)
            epoch_metrics['val_losses'].append(val_loss)
            epoch_metrics['train_metrics'].append(train_metrics)
            epoch_metrics['val_metrics'].append(val_metrics)

            # Early stopping based on F1 score
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_model = model.state_dict()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    logger.info("Best validation metrics:")
                    for metric, value in val_metrics.items():
                        logger.info(f"{metric}: {value:.4f}")
                    # break

            # Log metrics
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Train - Loss: {train_loss:.4f}")
            logger.info("Train Metrics:")
            for metric, value in train_metrics.items():
                logger.info(f"{metric}: {value:.4f}")
            logger.info(f"Val - Loss: {val_loss:.4f}")
            logger.info("Validation Metrics:")
            for metric, value in val_metrics.items():
                logger.info(f"{metric}: {value:.4f}")

        # Store fold metrics
        fold_metrics['train_losses'].append(epoch_metrics['train_losses'])
        fold_metrics['val_losses'].append(epoch_metrics['val_losses'])
        fold_metrics['train_metrics'].append(epoch_metrics['train_metrics'])
        fold_metrics['val_metrics'].append(epoch_metrics['val_metrics'])
        best_val_metrics.append(val_metrics)

    # Calculate and log average performance across folds
    avg_train_losses = 0
    avg_val_losses = 0
    
    # Plot average performance
    # plot_accuracy(
    #     train_losses=avg_train_losses,
    #     val_losses=avg_val_losses,
    #     train_accuracies=np.mean([[m['balanced_accuracy'] for m in fold] for fold in fold_metrics['train_metrics']], axis=0),
    #     val_accuracies=np.mean([[m['balanced_accuracy'] for m in fold] for fold in fold_metrics['val_metrics']], axis=0)
    # )

    # Print final metrics for each fold
    logger.info("\nBest validation metrics for each fold:")
    for fold, metrics in enumerate(best_val_metrics, 1):
        logger.info(f"\nFold {fold}:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
    
    # Load the best model state
    model.load_state_dict(best_model)
    return model

def evaluate_model(
        model: Transformer, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        dataset: str, 
        device: Union[torch.device,str]
    ) -> None:
    """Evaluate the model on the training and evaluation datasets with extended metrics."""
    logger = logging.getLogger(__name__)
    
    model.eval()
    with torch.no_grad():
        datasets = [("train", train_loader), ("validation", val_loader)]
        for name, data_loader in datasets:
            startTime = time.time()
            all_preds = []
            all_probs = []
            all_labels = []
            
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x, x)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                _, actual = y.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(actual.cpu().numpy())
            
            # Calculate all metrics
            metrics = calculate_metrics(
                np.array(all_labels),
                np.array(all_preds),
                np.array(all_probs)
            )
            
            logger.info(f"\n{name} set metrics:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.4f}")
            
            plot_confusion_matrix(dataset, name, np.array(all_labels), np.array(all_preds))
            endTime = time.time()
            logger.info(f"Total time taken to evaluate on {name} set: {(endTime - startTime):.2f}s")

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
    if dataset == "species" or dataset == "oil_simple":
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