# -*- coding: utf-8 -*-
"""
Unified XAI module for deep learning model explanations using LIME and Grad-CAM.

This module provides tools for explaining the predictions of deep learning models trained
on spectral data. It implements two primary explanation methods:

1.  **Grad-CAM (Gradient-weighted Class Activation Mapping)**: Visualizes which parts of the input spectrum
    were most important for a specific prediction by analyzing the gradients flowing into a target layer.
2.  **LIME (Local Interpretable Model-agnostic Explanations)**: Approximates the complex model locally
    with an interpretable model to explain individual predictions.

It also includes helper classes for configuring the explainers and wrapping models for compatibility.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from lime.lime_tabular import LimeTabularExplainer

from fishy.models.deep.transformer import Transformer
from fishy.models.deep.vae import VAE
from fishy.engine.training_loops import train_model
from fishy.data.module import create_data_module
from fishy._core.config import TrainingConfig
from fishy._core.factory import create_model
from fishy._core.utils import RunContext

logger = logging.getLogger(__name__)

from dataclasses import dataclass


@dataclass
class ExplainerConfig:
    """
    Configuration for model explanation methods.

    Attributes:
        num_features (int): Number of top features to show in the explanation (LIME).
        num_samples (int): Number of samples to generate for the local approximation (LIME).
        output_dir (Path): Directory where explanation plots will be saved.
        device (str): Device to run the model on ('cuda' or 'cpu').
        random_seed (int): Seed for reproducibility.
    """

    num_features: int = 5
    num_samples: int = 100
    output_dir: Path = Path("outputs/xai")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed: int = 42


# --- Grad-CAM Implementation ---


class GradCAM:
    """
    1D Grad-CAM implementation for analyzing spectral data models.

    Grad-CAM uses the gradients of any target concept (say, 'dog' in a classification network
    or a sequence of words in captioning) flowing into the final convolutional layer to produce
    a coarse localization map highlighting the important regions in the image (or spectrum) for predicting the concept.

    Args:
        model (nn.Module): The PyTorch model to explain.
        target_layer (nn.Module): The specific layer within the model to analyze (e.g., the last convolutional or attention layer).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.forward_hook = self.target_layer.register_forward_hook(
            self.save_activation
        )
        self.backward_hook = self.target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, input, output):
        """Hook to capture forward activations."""
        self.activations = (
            output[0].detach() if isinstance(output, tuple) else output.detach()
        )

    def save_gradient(self, module, grad_input, grad_output):
        """Hook to capture backward gradients."""
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        """Removes the forward and backward hooks from the model."""
        self.forward_hook.remove()
        self.backward_hook.remove()

    def generate_cam(
        self, input_tensor: torch.Tensor, target_class: Optional[int] = None
    ):
        """
        Generates the Class Activation Map (CAM) for a given input.

        Args:
            input_tensor (torch.Tensor): The input spectrum/sample.
            target_class (Optional[int]): The target class index to explain. If None, uses the predicted class.

        Returns:
            torch.Tensor: The computed CAM, normalized to [0, 1].
        """
        self.model.eval()
        if input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(1)

        model_output = self.model(input_tensor)
        if isinstance(model_output, tuple):
            model_output = model_output[0]

        if target_class is None:
            target_class = torch.argmax(model_output, dim=1)

        one_hot = torch.zeros_like(model_output)
        one_hot.scatter_(1, target_class.unsqueeze(1), 1.0)

        self.model.zero_grad()
        model_output.backward(gradient=one_hot, retain_graph=True)

        if self.activations is None or self.gradients is None:
            return torch.zeros(
                (input_tensor.shape[0], input_tensor.shape[2]),
                device=input_tensor.device,
            )

        # Pool gradients
        if isinstance(self.target_layer, nn.Linear):
            pooled_gradients = torch.mean(self.gradients, dim=0)
            for i in range(self.activations.shape[-1]):
                self.activations[:, i] *= pooled_gradients[i]
            cam = self.activations
        else:
            pooled_gradients = torch.mean(self.gradients, dim=[0, 1])
            for i in range(self.activations.shape[-1]):
                self.activations[:, :, i] *= pooled_gradients[i]
            cam = torch.sum(self.activations, dim=1)

        cam = F.relu(cam)
        # Resize to match input feature dimension
        if cam.shape[1] != input_tensor.shape[2]:
            cam = F.interpolate(
                cam.unsqueeze(1),
                size=input_tensor.shape[2],
                mode="linear",
                align_corners=False,
            ).squeeze(1)

        # Normalize
        for i in range(cam.shape[0]):
            if torch.max(cam[i]) > 0:
                cam[i] = (cam[i] - torch.min(cam[i])) / torch.max(cam[i])
        return cam


# --- LIME Model Wrapper ---


class ModelWrapper:
    """
    Wrapper for neural network models to make them compatible with LIME.

    LIME expects models to have a `predict_proba` method that takes a numpy array
    and returns probabilities. This wrapper adapts PyTorch models to this interface.

    Args:
        model (nn.Module): The PyTorch model.
        device (str): The device to run inference on.
    """

    def __init__(self, model: nn.Module, device: str) -> None:
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def normalize_intensities(self, x: np.ndarray) -> np.ndarray:
        """Min-max normalizes the input intensities."""
        if np.all(x == x[0]):
            return np.zeros_like(x)
        x_norm = np.zeros_like(x)
        for i in range(len(x)):
            low, high = np.min(x[i]), np.max(x[i])
            if high - low > 0:
                x_norm[i] = (x[i] - low) / (high - low)
        return x_norm

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts class probabilities for the given input.

        Args:
            x (np.ndarray): Input samples.

        Returns:
            np.ndarray: Class probabilities.
        """
        try:
            x_normalized = self.normalize_intensities(x)
            x_tensor = torch.tensor(x_normalized, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                outputs = self.model(x_tensor)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                return F.softmax(outputs, dim=-1).cpu().numpy()
        except Exception as e:
            logger.error(f"Error in predict_proba: {e}")
            raise


# --- High-level Analysis Functions ---


def run_lime_explanation(
    dataset_name: str,
    model: nn.Module,
    data_module: Any,
    explainer_config: ExplainerConfig,
    instance_name: str,
    target_label: List[float],
    ctx: RunContext,
):
    """
    Generates LIME explanation for a model prediction.

    Args:
        dataset_name (str): Name of the dataset.
        model (nn.Module): The model to explain.
        data_module (Any): Data module containing the dataset.
        explainer_config (ExplainerConfig): Configuration for the explainer.
        instance_name (str): Identifier for the instance being explained.
        target_label (List[float]): Target label to explain.
        ctx (RunContext): context for saving results.
    """
    wrapper = ModelWrapper(model, explainer_config.device)
    train_loader = data_module.get_train_dataloader()
    features, labels = next(iter(train_loader))

    feature_names = [
        f"{float(x):.4f}" for x in data_module.get_train_dataframe().columns[1:]
    ]
    class_names = ["Hoki", "Mackerel"]  # Default, can be expanded

    explainer = LimeTabularExplainer(
        training_data=features.cpu().numpy(),
        training_labels=labels.cpu().numpy(),
        feature_names=feature_names,
        discretize_continuous=True,
        random_state=explainer_config.random_seed,
    )

    # Find instance logic (simplified)
    instance = features[0]  # Just take first for now or implement find_instance

    exp = explainer.explain_instance(
        instance.cpu().numpy().flatten(),
        wrapper.predict_proba,
        num_features=explainer_config.num_features,
    )

    output_path = ctx.figure_dir / f"lime_{dataset_name}_{instance_name}.png"
    exp.as_pyplot_figure()
    plt.savefig(output_path)
    plt.close()
    ctx.logger.info(f"Saved LIME explanation to {output_path}")


def run_gradcam_analysis(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str,
    output_dir: Path,
    target_layer: Optional[nn.Module] = None,
    ctx: Optional[RunContext] = None,
):
    """
    Performs Grad-CAM analysis on a model using data from the loader.

    Args:
        model (nn.Module): The model to analyze.
        data_loader (DataLoader): DataLoader providing samples for analysis.
        device (str): Computation device.
        output_dir (Path): Directory to save results.
        target_layer (Optional[nn.Module]): Specific layer to target. If None, attempts to infer.
        ctx: RunContext.
    """
    if target_layer is None:
        # Heuristic to find a good target layer (last Conv or last Attention)
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, (nn.Conv1d, nn.Linear)):  # Fallback to Linear
                target_layer = module
                break

    if target_layer is None:
        raise ValueError("Could not automatically find a target layer for Grad-CAM.")

    gc = GradCAM(model, target_layer)
    features, labels = next(iter(data_loader))
    features = features.to(device)

    cam_maps = gc.generate_cam(features)
    avg_cam = cam_maps.mean(dim=0).cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(avg_cam)
    plt.title(f"Average Grad-CAM Importance ({target_layer.__class__.__name__})")

    if ctx:
        ctx.save_figure(plt, "avg_gradcam.png")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "avg_gradcam.png")
        print(f"Saved average Grad-CAM to {output_dir / 'avg_gradcam.png'}")

    plt.close()
    gc.remove_hooks()


def explain_predictions(
    dataset_name: str,
    model_name: str,
    training_config: TrainingConfig,
    explainer_config: ExplainerConfig,
    instance_name: str,
    target_label: List[float],
    method: str = "lime",
) -> None:
    """
    Orchestrates the XAI analysis workflow.

    Args:
        dataset_name (str): Name of the dataset.
        model_name (str): Name of the model architecture.
        training_config (TrainingConfig): Configuration used for model creation/training.
        explainer_config (ExplainerConfig): Configuration for the explainer.
        instance_name (str): Identifier for the instance.
        target_label (List[float]): Target label.
        method (str): Explanation method ('lime' or 'gradcam').
    """
    ctx = RunContext(dataset=dataset_name, method=method, model_name=model_name)
    ctx.save_config(training_config, filename="training_config.json")
    ctx.save_config(explainer_config, filename="explainer_config.json")

    from fishy.experiments.deep_training import ModelTrainer

    num_classes = ModelTrainer.N_CLASSES_PER_DATASET.get(dataset_name, 2)

    data_path = str(
        Path(__file__).resolve().parent.parent.parent / "data" / "REIMS.xlsx"
    )
    data_module = create_data_module(
        file_path=data_path,
        dataset_name=dataset_name,
        batch_size=32,
    )
    data_module.setup()

    model = create_model(training_config, data_module.get_input_dim(), num_classes).to(
        explainer_config.device
    )

    # In real usage, you'd load a pre-trained model here
    # For now, we assume it's trained or we train briefly
    if method == "lime":
        run_lime_explanation(
            dataset_name,
            model,
            data_module,
            explainer_config,
            instance_name,
            target_label,
            ctx,
        )
    elif method == "gradcam":
        run_gradcam_analysis(
            model,
            data_module.get_train_dataloader(),
            explainer_config.device,
            explainer_config.output_dir,
            ctx=ctx,
        )
