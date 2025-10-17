"""
This script implements a 1D Grad-CAM analysis for a Transformer model
trained on mass spectrometry data using contrastive learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional

from models import Transformer, SimCLRModel
from deep_learning.grad_cam import GradCAM
from .util import SiameseDataset, DataPreprocessor, DataConfig


class ContrastiveGradCAM:
    """
    Grad-CAM for contrastive models.
    """

    def __init__(self, model: SimCLRModel, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.grad_cam = GradCAM(self.model.encoder, target_layer)

    def generate_cam(self, input1: torch.Tensor, input2: torch.Tensor):
        """
        Generate Grad-CAM for a pair of inputs.
        The CAM is generated for the similarity score between the two inputs.
        """
        # Ensure model is in eval mode
        self.model.eval()

        # Forward pass to get the embeddings
        h1 = self.model.encoder(input1)

        # The "score" is the cosine similarity
        with torch.no_grad():
            h2 = self.model.encoder(input2)

        similarity = torch.nn.functional.cosine_similarity(h1, h2, dim=-1).mean()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        similarity.backward()

        # Get the gradients from the target layer
        gradients = self.grad_cam.gradients
        if gradients is None:
            return None

        # Pool gradients
        pooled_gradients = torch.mean(gradients, dim=[0, 1])

        # Get activations
        activations = self.grad_cam.activations
        if activations is None:
            return None

        # Weight activations with gradients
        for i in range(activations.shape[-1]):
            activations[:, :, i] *= pooled_gradients[i]

        # Sum over the channels
        cam = torch.sum(activations, dim=1)

        # Apply ReLU and normalize
        cam = torch.nn.functional.relu(cam)
        if torch.max(cam) > 0:
            cam = (cam - torch.min(cam)) / torch.max(cam)

        return cam

    def remove_hooks(self):
        self.grad_cam.remove_hooks()


def visualize_gradcam(features, cam_map, title="Grad-CAM Analysis"):
    """
    Simple visualization function that works with any input shape

    Args:
        features: Feature tensor (will be converted to 1D array)
        cam_map: CAM map tensor (will be converted to 1D array)
        title: Plot title

    Returns:
        Matplotlib figure object
    """
    # Get the feature tensor as numpy array
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()

    # Get the CAM map as numpy array
    if isinstance(cam_map, torch.Tensor):
        cam_map = cam_map.detach().cpu().numpy()

    # Handle different tensor shapes - ensure we have 1D arrays
    if features.ndim > 1:
        features = features.flatten()

    if cam_map.ndim > 1:
        cam_map = cam_map.flatten()

    # Create x-axis
    x = np.arange(len(features))

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot Grad-CAM
    ax.plot(x, cam_map, "r-", alpha=0.7, label="Grad-CAM")
    ax.fill_between(x, 0, cam_map, color="r", alpha=0.3)
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Grad-CAM Value")
    ax.tick_params(axis="y")

    plt.title(title)
    plt.tight_layout()

    return fig


def analyze_contrastive_gradcam(
    model: SimCLRModel,
    data_loader: DataLoader,
    device: torch.device,
    output_dir: str = "gradcam_results_contrastive",
    num_samples: int = 5,
):
    """
    Analyze contrastive model with Grad-CAM.
    """
    os.makedirs(output_dir, exist_ok=True)

    model.to(device)
    model.eval()

    target_layer = model.encoder.attention_layers[-1]
    contrastive_grad_cam = ContrastiveGradCAM(model, target_layer)

    positive_cams = []
    sample_count = 0

    for i, (x1, x2, labels) in enumerate(data_loader):
        if sample_count >= num_samples:
            break

        x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)

        # Generate CAM for positive pairs
        for j in range(x1.shape[0]):
            if torch.argmax(labels[j]) == 1:
                cam1 = contrastive_grad_cam.generate_cam(
                    x1[j].unsqueeze(0), x2[j].unsqueeze(0)
                )
                if cam1 is not None:
                    positive_cams.append(cam1.cpu().numpy())
                    sample_count += 1
                    if sample_count >= num_samples:
                        break

    if positive_cams:
        avg_cam = np.mean(positive_cams, axis=0)
        # get a sample feature to visualize
        features, _, _ = next(iter(data_loader))
        fig = visualize_gradcam(
            features.mean(dim=0), avg_cam, title="Average Grad-CAM for Positive Pairs"
        )
        plt.figure(fig.number)
        plt.savefig(f"{output_dir}/average_positive_grad_cam.png")
        plt.close()

    contrastive_grad_cam.remove_hooks()
