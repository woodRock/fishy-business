""" This module implements a sequential transfer learning framework for deep learning models.
It allows for training models on multiple datasets sequentially, with the final dataset being used for fine-tuning.
The framework is designed to work with PyTorch and includes functions for training, evaluation,
and visualization of results."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
)
import seaborn as sns
from collections import defaultdict
import copy


def sequential_transfer_learning(
    model_class,
    model_params,
    transfer_datasets,
    target_dataset,
    num_epochs_transfer=10,
    num_epochs_finetune=20,
    batch_size=32,
    learning_rate=1e-3,
    finetune_lr=5e-4,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_intermediate=True,
    val_split=0.2,
):
    """
    Perform sequential transfer learning across multiple datasets, then fine-tune on target dataset.

    Args:
        model_class: The model class to instantiate
        model_params: Dictionary of parameters for model initialization
        transfer_datasets: List of dataset names to use for sequential transfer learning
        target_dataset: Name of the final dataset to fine-tune on
        num_epochs_transfer: Number of epochs for each transfer learning phase
        num_epochs_finetune: Number of epochs for final fine-tuning
        batch_size: Batch size for training
        learning_rate: Learning rate for transfer learning phases
        finetune_lr: Learning rate for fine-tuning phase
        device: Device to train on
        save_intermediate: Whether to save intermediate models
        val_split: Validation split ratio

    Returns:
        final_model: The fine-tuned model
        history: Dictionary containing training history
    """
    history = {
        "transfer": {},
        "finetune": {},
    }

    # Initialize model
    model = model_class(**model_params)
    model = model.to(device)
    print(f"Model initialized on {device}")

    # Sequential transfer learning on multiple datasets
    for i, dataset_name in enumerate(transfer_datasets):
        print(f"\n{'='*50}")
        print(f"Transfer Learning Phase {i+1}: Dataset '{dataset_name}'")
        print(f"{'='*50}\n")

        # Load dataset
        dataset, targets = load_data(dataset=dataset_name)

        # Split into train and validation
        dataset_size = len(dataset)
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Update output layer to match current dataset's classes
        num_classes = dataset[0][1].shape[0]
        old_output_layer = model.fc_out

        # Keep the weights for the shared dimensions if possible
        if hasattr(model, "fc_out") and i > 0:
            old_weights = model.fc_out.weight.data
            old_bias = model.fc_out.bias.data
            model.fc_out = nn.Linear(model_params["input_dim"], num_classes).to(device)

            # Transfer weights for shared dimensions
            min_classes = min(old_weights.shape[0], num_classes)
            model.fc_out.weight.data[:min_classes, :] = old_weights[:min_classes, :]
            model.fc_out.bias.data[:min_classes] = old_bias[:min_classes]
        else:
            model.fc_out = nn.Linear(model_params["input_dim"], num_classes).to(device)

        # Train on this dataset
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

        dataset_history = train_with_tracking(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs_transfer,
            device=device,
        )

        history["transfer"][dataset_name] = dataset_history

        # Save intermediate model if requested
        if save_intermediate:
            torch.save(model.state_dict(), f"model_transfer_{dataset_name}.pt")

        print(f"Completed transfer learning on '{dataset_name}'")

    # Fine-tuning on target dataset
    print(f"\n{'='*50}")
    print(f"Fine-tuning on target dataset: '{target_dataset}'")
    print(f"{'='*50}\n")

    # Load target dataset
    target_data, target_targets = load_data(dataset=target_dataset)
    dataset_size = len(target_data)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(target_data, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Update output layer for target dataset
    num_classes = target_data[0][1].shape[0]
    old_weights = model.fc_out.weight.data
    old_bias = model.fc_out.bias.data
    model.fc_out = nn.Linear(model_params["input_dim"], num_classes).to(device)

    # Transfer weights for shared dimensions
    min_classes = min(old_weights.shape[0], num_classes)
    model.fc_out.weight.data[:min_classes, :] = old_weights[:min_classes, :]
    model.fc_out.bias.data[:min_classes] = old_bias[:min_classes]

    # Use a smaller learning rate for fine-tuning
    optimizer = AdamW(model.parameters(), lr=finetune_lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

    # Fine-tune the model
    finetune_history = train_with_tracking(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs_finetune,
        device=device,
    )

    history["finetune"][target_dataset] = finetune_history

    # Save final model
    torch.save(model.state_dict(), f"model_final_{target_dataset}.pt")

    # Evaluate on test data
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            true_labels = torch.argmax(y, dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())

    # Calculate balanced accuracy
    final_balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
    print(
        f"\nFinal Balanced Accuracy on {target_dataset}: {final_balanced_accuracy*100:.2f}%"
    )

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title(
        f"Confusion Matrix - {target_dataset} (Balanced Acc: {final_balanced_accuracy*100:.2f}%)"
    )
    plt.savefig(f"confusion_matrix_{target_dataset}.png")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    # Analyze expert utilization patterns (for MoE models)
    if hasattr(model, "moe_layers"):
        print("\nExpert Utilization Analysis:")
        for i, moe_layer in enumerate(model.moe_layers):
            util = moe_layer.get_expert_utilization()
            print(f"Layer {i+1} Expert Utilization: {[f'{u:.4f}' for u in util]}")

    return model, history


def train_with_tracking(
    model, train_loader, val_loader, optimizer, scheduler, num_epochs, device
):
    """Train model with detailed tracking of metrics, including balanced accuracy.
    
    Args: 
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        num_epochs (int): Number of epochs to train.
        device (torch.device): Device to run the training on.

    Returns:
        history (dict): Dictionary containing training and validation metrics.
    """
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_balanced_acc": [],
        "val_balanced_acc": [],
        "learning_rates": [],
    }

    best_val_acc = 0
    best_model = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_all_preds = []
        train_all_labels = []

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, torch.argmax(y, dim=1))

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            true_labels = torch.argmax(y, dim=1)
            train_total += y.size(0)
            train_correct += (predicted == true_labels).sum().item()

            # Collect predictions and labels for balanced accuracy
            train_all_preds.extend(predicted.cpu().numpy())
            train_all_labels.extend(true_labels.cpu().numpy())

            # Print progress
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}, "
                    f"Acc: {100 * train_correct / train_total:.2f}%"
                )

        # Calculate epoch-level metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100 * train_correct / train_total
        epoch_train_balanced_acc = 100 * balanced_accuracy_score(
            train_all_labels, train_all_preds
        )

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_all_preds = []
        val_all_labels = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, torch.argmax(y, dim=1))

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                true_labels = torch.argmax(y, dim=1)
                val_total += y.size(0)
                val_correct += (predicted == true_labels).sum().item()

                # Collect predictions and labels for balanced accuracy
                val_all_preds.extend(predicted.cpu().numpy())
                val_all_labels.extend(true_labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * val_correct / val_total
        epoch_val_balanced_acc = 100 * balanced_accuracy_score(
            val_all_labels, val_all_preds
        )

        # Update learning rate scheduler - now using balanced accuracy
        scheduler.step(epoch_val_balanced_acc)

        # Track history
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)
        history["train_balanced_acc"].append(epoch_train_balanced_acc)
        history["val_balanced_acc"].append(epoch_val_balanced_acc)
        history["learning_rates"].append(optimizer.param_groups[0]["lr"])

        # Save best model based on balanced accuracy
        if epoch_val_balanced_acc > best_val_acc:
            best_val_acc = epoch_val_balanced_acc
            best_model = copy.deepcopy(model.state_dict())

        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {epoch_train_loss:.4f}, "
            f"Val Loss: {epoch_val_loss:.4f}, "
            f"Train Acc: {epoch_train_acc:.2f}%, "
            f"Val Acc: {epoch_val_acc:.2f}%, "
            f"Train Bal Acc: {epoch_train_balanced_acc:.2f}%, "
            f"Val Bal Acc: {epoch_val_balanced_acc:.2f}%, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)

    return history


def visualize_training_history(history):
    """Visualize training and validation metrics over time.
    
    Args: 
        history (dict): Dictionary containing training history with keys:
                        - train_loss, val_loss
                        - train_acc, val_acc
                        - train_balanced_acc, val_balanced_acc
                        - learning_rates
    """
    # Create figure with subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 20), sharex=True)

    # Plot losses
    ax1.plot(history["train_loss"], label="Training Loss")
    ax1.plot(history["val_loss"], label="Validation Loss")
    ax1.set_title("Loss over Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Plot standard accuracies
    ax2.plot(history["train_acc"], label="Training Accuracy")
    ax2.plot(history["val_acc"], label="Validation Accuracy")
    ax2.set_title("Standard Accuracy over Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()

    # Plot balanced accuracies
    ax3.plot(history["train_balanced_acc"], label="Training Balanced Accuracy")
    ax3.plot(history["val_balanced_acc"], label="Validation Balanced Accuracy")
    ax3.set_title("Balanced Accuracy over Epochs")
    ax3.set_ylabel("Balanced Accuracy (%)")
    ax3.legend()

    # Plot learning rate
    ax4.plot(history["learning_rates"], label="Learning Rate")
    ax4.set_title("Learning Rate over Epochs")
    ax4.set_xlabel("Epochs")
    ax4.set_ylabel("Learning Rate")
    ax4.set_yscale("log")
    ax4.legend()

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.close()


def visualize_transfer_learning_performance(history):
    """Visualize performance across different datasets during transfer learning.
    
    Args: 
        history (dict): Dictionary containing transfer learning history with keys:
                        - transfer: Dictionary of datasets with their validation accuracies
                        - finetune: Dictionary of datasets with their validation accuracies 
    """
    datasets = list(history["transfer"].keys()) + list(history["finetune"].keys())
    final_accuracies = []
    final_balanced_accuracies = []

    # Collect final validation accuracies for each dataset
    for dataset in history["transfer"]:
        final_accuracies.append(history["transfer"][dataset]["val_acc"][-1])
        final_balanced_accuracies.append(
            history["transfer"][dataset]["val_balanced_acc"][-1]
        )

    for dataset in history["finetune"]:
        final_accuracies.append(history["finetune"][dataset]["val_acc"][-1])
        final_balanced_accuracies.append(
            history["finetune"][dataset]["val_balanced_acc"][-1]
        )

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot standard accuracies
    bars1 = ax1.bar(datasets, final_accuracies)
    ax1.set_title("Final Standard Validation Accuracy by Dataset")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 100)

    # Add values on top of bars
    for bar, acc in zip(bars1, final_accuracies):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
        )

    # Plot balanced accuracies
    bars2 = ax2.bar(datasets, final_balanced_accuracies)
    ax2.set_title("Final Balanced Validation Accuracy by Dataset")
    ax2.set_xlabel("Dataset")
    ax2.set_ylabel("Balanced Accuracy (%)")
    ax2.set_ylim(0, 100)

    # Add values on top of bars
    for bar, acc in zip(bars2, final_balanced_accuracies):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
        )

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("transfer_learning_performance.png")
    plt.close()


def main():
    """ Main execution function for sequential transfer learning. """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define model parameters (based on previous hyperparameter tuning)
    model_params = {
        "input_dim": 2080,  # Fixed based on the data
        "output_dim": 7,  # Will be updated during training for each dataset
        "num_heads": 8,
        "hidden_dim": 384,
        "num_layers": 2,
        "num_experts": 6,
        "k": 2,
        "dropout": 0.2,
    }

    # Define transfer learning sequence
    transfer_datasets = ["oil"]
    target_dataset = "cross-species"

    # Run sequential transfer learning
    final_model, history = sequential_transfer_learning(
        model_class=MOE,
        model_params=model_params,
        transfer_datasets=transfer_datasets,
        target_dataset=target_dataset,
        num_epochs_transfer=100,
        num_epochs_finetune=100,
        batch_size=32,
        learning_rate=1e-3,
        finetune_lr=5e-4,
        device=device,
        save_intermediate=False,
    )

    # Visualize results
    for dataset, dataset_history in history["transfer"].items():
        visualize_training_history(dataset_history)

    visualize_training_history(history["finetune"][target_dataset])
    visualize_transfer_learning_performance(history)

    print("Sequential transfer learning completed successfully!")

    return final_model, history


if __name__ == "__main__":
    # Import the required functions from the original code
    from MOE import load_data, MOE

    final_model, history = main()
