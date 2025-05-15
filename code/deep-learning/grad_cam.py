import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# Import your original Transformer and MultiHeadAttention implementations
from transformer import Transformer, MultiHeadAttention

class GradCAM:
    """
    1D Grad-CAM implementation for analyzing attention in transformer models
    for mass spectrometry data.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.forward_hook = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()
    
    def generate_cam(self, input_tensor, target_class=None):
        # Ensure model is in eval mode
        self.model.eval()
        
        # Ensure input has 3 dimensions [batch_size, seq_len, features]
        if input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(1)
        
        # Forward pass
        model_output = self.model(input_tensor)
        
        # If target_class is None, use the predicted class
        if target_class is None:
            target_class = torch.argmax(model_output, dim=1)
        
        # Create one-hot encoding for the target class
        one_hot = torch.zeros_like(model_output)
        one_hot.scatter_(1, target_class.unsqueeze(1), 1.0)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        model_output.backward(gradient=one_hot, retain_graph=True)
        
        # Get activation and gradient
        if self.activations is None or self.gradients is None:
            print("Warning: Activations or gradients are None")
            # Return dummy tensor
            return torch.zeros((input_tensor.shape[0], input_tensor.shape[2]), device=input_tensor.device)
        
        # Use the activations and gradients to calculate importance
        weights = torch.mean(self.gradients, dim=1)  # [batch_size, features]
        batch_size = input_tensor.shape[0]
        feature_dim = input_tensor.shape[2]
        
        # Create CAM map of shape [batch_size, features]
        cam = torch.zeros((batch_size, feature_dim), device=input_tensor.device)
        
        # Calculate feature importance
        for i in range(batch_size):
            for j in range(feature_dim):
                cam[i, j] = weights[i, j] * self.activations[i, 0, j]
        
        # Apply ReLU and normalize
        cam = torch.nn.functional.relu(cam)
        
        # Normalize each sample independently
        for i in range(batch_size):
            if torch.max(cam[i]) > torch.min(cam[i]):
                cam[i] = (cam[i] - torch.min(cam[i])) / (torch.max(cam[i]) - torch.min(cam[i]))
        
        return cam


def train_model(model, train_loader, val_loader, device, num_epochs=20, lr=0.001, save_path='trained_model.pt'):
    """
    Train the Transformer model
    
    Args:
        model: Transformer model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on (cuda/cpu)
        num_epochs: Number of training epochs
        lr: Learning rate
        save_path: Path to save the best model
        
    Returns:
        Trained model
    """
    print(f"Starting model training for {num_epochs} epochs...")
    
    # Move model to device
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Track best accuracy and save best model
    best_accuracy = 0.0
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        # Training phase
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(features)
            
            # Get target classes (assuming one-hot encoded labels)
            targets = torch.argmax(labels, dim=1)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            train_loss += loss.item()
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # Average training loss for the epoch
        avg_train_loss = train_loss / len(train_loader)
        training_history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(features)
                
                # Get target classes
                targets = torch.argmax(labels, dim=1)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        # Average validation loss and accuracy
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100.0 * correct / total
        training_history['val_loss'].append(avg_val_loss)
        training_history['val_accuracy'].append(val_accuracy)
        
        # Update learning rate
        scheduler.step(val_accuracy)
        
        # Save the best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with accuracy: {val_accuracy:.2f}%")
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f}s")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        print("-" * 60)
    
    # Load the best model
    model.load_state_dict(torch.load(save_path))
    print(f"Training completed. Best validation accuracy: {best_accuracy:.2f}%")
    
    # Plot training history
    plot_training_history(training_history)
    
    return model


def plot_training_history(history):
    """
    Plot training and validation metrics
    
    Args:
        history: Dictionary containing training metrics
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()


def visualize_gradcam(features, cam_map, idx=0, title="Grad-CAM Analysis"):
    """
    Simple visualization function that works with any input shape
    
    Args:
        features: Feature tensor (will be converted to 1D array)
        cam_map: CAM map tensor (will be converted to 1D array)
        idx: Sample index
        title: Plot title
    """
    # Get the feature tensor as numpy array
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    
    # Get the CAM map as numpy array
    if isinstance(cam_map, torch.Tensor):
        cam_map = cam_map.detach().cpu().numpy()
    
    # Handle different tensor shapes - ensure we have 1D arrays
    if features.ndim > 1:
        # If features is [batch, features] or [batch, seq, features]
        if idx < features.shape[0]:
            features = features[idx]
        # If still multi-dimensional, flatten to 1D
        if features.ndim > 1:
            features = features.flatten()
    
    if cam_map.ndim > 1:
        # If CAM is [batch, features]
        if idx < cam_map.shape[0]:
            cam_map = cam_map[idx]
        # If still multi-dimensional, flatten to 1D
        if cam_map.ndim > 1:
            cam_map = cam_map.flatten()
    
    # Create x-axis
    x = np.arange(len(features))
    
    # Create plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot features
    ax1.plot(x, features, 'b-', alpha=0.7, label='Features')
    ax1.set_xlabel('Feature Index')
    ax1.set_ylabel('Feature Value', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Create second y-axis for CAM
    ax2 = ax1.twinx()
    ax2.plot(x, cam_map, 'r-', alpha=0.7, label='Grad-CAM')
    ax2.fill_between(x, 0, cam_map, color='r', alpha=0.3)
    ax2.set_ylabel('Grad-CAM Value', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title(title)
    plt.tight_layout()
    
    return fig


def analyze_with_gradcam(model, data_loader, device, output_dir='gradcam_results', num_samples=5):
    """
    Analyze mass spectrometry data with Grad-CAM
    
    Args:
        model: Trained Transformer model
        data_loader: DataLoader containing test data
        device: Device to run on (cuda/cpu)
        output_dir: Directory to save results
        num_samples: Number of samples to analyze
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    # Get the last MultiHeadAttention layer
    target_layer = model.attention_layers[-1]
    print(f"Using target layer: {target_layer.__class__.__name__}")
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # Process samples
    sample_count = 0
    class_correct = {}
    class_total = {}
    all_cams = []
    
    for batch_idx, (features, labels) in enumerate(data_loader):
        if sample_count >= num_samples:
            break
            
        features = features.to(device)
        
        # Generate Grad-CAM maps
        cam_maps = grad_cam.generate_cam(features)
        
        # Process each sample in the batch
        for i in range(min(len(features), num_samples - sample_count)):
            # Get predicted class
            with torch.no_grad():
                outputs = model(features[i:i+1])
                _, predicted = torch.max(outputs, 1)
                true_class = torch.argmax(labels[i])
                
                # Track accuracy per class
                class_label = true_class.item()
                if class_label not in class_total:
                    class_total[class_label] = 0
                    class_correct[class_label] = 0
                
                class_total[class_label] += 1
                if predicted.item() == class_label:
                    class_correct[class_label] += 1
            
            # Store CAM map
            all_cams.append({
                'cam': cam_maps[i].cpu().numpy(),
                'true_class': true_class.item(),
                'predicted_class': predicted.item()
            })
            
            # Create title with correct/incorrect prediction
            is_correct = "Correct" if predicted.item() == true_class.item() else "Incorrect"
            title = f"Sample {sample_count+1} - Predicted: {predicted.item()}, True: {true_class.item()} ({is_correct})"
            
            # Create visualization
            fig = visualize_gradcam(
                features=features[i],
                cam_map=cam_maps[i],
                title=title
            )
            
            # Save figure
            plt.figure(fig.number)
            plt.savefig(f"{output_dir}/gradcam_sample_{sample_count}.png")
            plt.close()
            
            print(f"Saved visualization for sample {sample_count+1} ({is_correct} prediction)")
            sample_count += 1
    
    # Clean up hooks
    grad_cam.remove_hooks()
    
    # Calculate and display class-wise accuracy
    print("\nClass-wise Accuracy:")
    for class_idx in sorted(class_total.keys()):
        accuracy = 100 * class_correct[class_idx] / class_total[class_idx]
        print(f"Class {class_idx}: {accuracy:.2f}% ({class_correct[class_idx]}/{class_total[class_idx]})")
    
    # Generate average CAM map for correctly classified samples
    correct_cams = [item['cam'] for item in all_cams if item['predicted_class'] == item['true_class']]
    if correct_cams:
        avg_correct_cam = np.mean(correct_cams, axis=0)
        plt.figure(figsize=(10, 6))
        plt.plot(avg_correct_cam)
        plt.title("Average Grad-CAM for Correctly Classified Samples")
        plt.xlabel("Feature Index")
        plt.ylabel("Avg. Importance")
        plt.savefig(f"{output_dir}/avg_correct_gradcam.png")
        plt.close()
    
    print(f"Analysis complete. Saved {sample_count} visualizations to {output_dir}/")


def prepare_data_loaders(data_loader, val_split=0.2, test_split=0.1, batch_size=None):
    """
    Prepare train, validation, and test data loaders from a single data loader
    
    Args:
        data_loader: Original data loader
        val_split: Validation split ratio
        test_split: Test split ratio
        batch_size: Batch size (if None, uses the original batch size)
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Get the dataset from the data loader
    dataset = data_loader.dataset
    
    # Calculate sizes
    total_size = len(dataset)
    test_size = int(test_split * total_size)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size - test_size
    
    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Get batch size
    if batch_size is None:
        batch_size = data_loader.batch_size
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f"Data split: Train={train_size}, Validation={val_size}, Test={test_size}")
    
    return train_loader, val_loader, test_loader


def main():
    """
    Main function to train the model and run Grad-CAM analysis
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Import your data loading utility
    from util import create_data_module
    
    # Create data module and load dataset
    data_module = create_data_module(
        dataset_name="part",  # Use your desired dataset
        batch_size=32
    )
    
    # Setup the data loader
    data_loader, _ = data_module.setup()
    
    # Split data into train, validation, and test sets
    train_loader, val_loader, test_loader = prepare_data_loaders(
        data_loader, val_split=0.2, test_split=0.1
    )
    
    # Get sample to determine dimensions
    sample_features, sample_labels = next(iter(train_loader))
    feature_dim = sample_features.shape[1]
    output_dim = sample_labels.shape[1]
    
    print(f"Feature dimension: {feature_dim}, Output dimension: {output_dim}")
    
    # Create model output directory
    os.makedirs('model', exist_ok=True)
    
    # Check if trained model exists
    model_path = 'model/trained_transformer.pt'
    
    # Create Transformer model
    model = Transformer(
        input_dim=feature_dim,
        output_dim=output_dim,
        num_heads=4,
        hidden_dim=128,
        num_layers=7,
        dropout=0.1
    )
    
    # Train or load model
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print("Training new model...")
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=20,
            lr=0.001,
            save_path=model_path
        )
    
    # Run Grad-CAM analysis on test set
    analyze_with_gradcam(
        model=model,
        data_loader=test_loader,
        device=device,
        output_dir='gradcam_results',
        num_samples=10
    )


if __name__ == "__main__":
    main()