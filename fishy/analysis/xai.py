# -*- coding: utf-8 -*-
"""
Unified XAI module for deep learning model explanations using LIME and Grad-CAM.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from lime.lime_tabular import LimeTabularExplainer

from fishy._core.utils import RunContext, get_device

logger = logging.getLogger(__name__)

from dataclasses import dataclass


@dataclass
class ExplainerConfig:
    """Configuration for model explanation methods."""

    num_features: int = 5
    num_samples: int = 100
    output_dir: Path = Path("outputs/xai")
    device: str = str(get_device())
    random_seed: int = 42


class GradCAM:
    """1D Grad-CAM implementation for analyzing spectral data models."""

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
        self.activations = (
            output[0].detach() if isinstance(output, tuple) else output.detach()
        )

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

    def generate_cam(
        self, input_tensor: torch.Tensor, target_class: Optional[int] = None
    ):
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
        if cam.shape[1] != input_tensor.shape[2]:
            cam = F.interpolate(
                cam.unsqueeze(1),
                size=input_tensor.shape[2],
                mode="linear",
                align_corners=False,
            ).squeeze(1)
        for i in range(cam.shape[0]):
            if torch.max(cam[i]) > 0:
                cam[i] = (cam[i] - torch.min(cam[i])) / torch.max(cam[i])
        return cam


class ModelWrapper:
    """Wrapper for models to make them compatible with LIME (handles both Torch and Sklearn)."""

    def __init__(self, model: Any, device: str) -> None:
        self.model = model
        self.device = device
        # Safely handle PyTorch-specific methods
        if hasattr(model, "to"):
            try:
                self.model = model.to(device)
            except:
                pass
        if hasattr(model, "eval"):
            try:
                self.model.eval()
            except:
                pass

    def normalize_intensities(self, x: np.ndarray) -> np.ndarray:
        x_norm = np.zeros_like(x)
        for i in range(len(x)):
            low, high = np.min(x[i]), np.max(x[i])
            if high - low > 0:
                x_norm[i] = (x[i] - low) / (high - low)
        return x_norm

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        try:
            x_norm = self.normalize_intensities(x)
            # 1. Handle as a PyTorch model
            if isinstance(self.model, nn.Module):
                # Process in chunks to avoid memory overflow for large inputs (e.g. from LIME)
                chunk_size = 128
                results = []
                for i in range(0, len(x_norm), chunk_size):
                    chunk = x_norm[i : i + chunk_size]
                    x_tensor = torch.tensor(chunk, dtype=torch.float32).to(self.device)
                    with torch.no_grad():
                        out = self.model(x_tensor)
                        if isinstance(out, tuple):
                            out = out[0]
                        probs = F.softmax(out, dim=-1).cpu().numpy()
                        results.append(probs)
                return np.concatenate(results, axis=0) if results else np.empty((0,))

            # 2. Try as a standard sklearn-like model if it has predict_proba
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(x_norm)

            # 3. Fallback for models that only have predict (e.g. some regressors or SVC)
            if hasattr(self.model, "predict"):
                preds = self.model.predict(x_norm).astype(int)
                # Determine number of classes if possible
                n_classes = 2
                if hasattr(self.model, "classes_"):
                    n_classes = len(self.model.classes_)
                elif hasattr(self.model, "n_classes_"):
                    n_classes = self.model.n_classes_

                # Convert to one-hot probabilities
                return np.eye(n_classes)[preds]

            raise AttributeError(
                "Model has neither PyTorch interface nor predict_proba."
            )
        except Exception as e:
            logger.error(f"Error in predict_proba: {e}")
            raise


def run_lime_explanation(
    dataset_name, model, data_module, explainer_config, instance_name, target_label, ctx
):
    wrapper = ModelWrapper(model, explainer_config.device)
    features, labels = next(iter(data_module.get_train_dataloader()))
    explainer = LimeTabularExplainer(
        features.cpu().numpy(),
        feature_names=[
            f"{float(x):.4f}" for x in data_module.get_train_dataframe().columns[1:]
        ],
        discretize_continuous=True,
        random_state=explainer_config.random_seed,
    )
    exp = explainer.explain_instance(
        features[0].cpu().numpy().flatten(),
        wrapper.predict_proba,
        num_features=explainer_config.num_features,
    )
    output_path = ctx.figure_dir / f"lime_{dataset_name}_{instance_name}.png"
    exp.as_pyplot_figure()
    plt.savefig(output_path)
    plt.close()


def run_gradcam_analysis(
    model, data_loader, device, output_dir, target_layer=None, ctx=None
):
    if target_layer is None:
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                target_layer = module
                break
    if target_layer is None:
        raise ValueError("Could not find target layer.")
    gc = GradCAM(model, target_layer)
    features, _ = next(iter(data_loader))
    cam = gc.generate_cam(features.to(device)).mean(dim=0).cpu().numpy()
    plt.figure(figsize=(10, 6))
    plt.plot(cam)
    plt.title("Grad-CAM Importance")
    if ctx:
        ctx.save_figure(plt, "avg_gradcam.png")
    else:
        (output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "avg_gradcam.png")
    plt.close()
    gc.remove_hooks()


def visualize_feature_interaction(model, data_loader, device, ctx=None):
    """
    Generates a 10x10 Feature Interaction Matrix for the top 10 features identified via saliency.
    This provides an 'attention-like' visualization of feature dependencies even for seq_len=1.
    """
    model.eval()
    
    # 1. Identify top 10 features using Grad-CAM on the final layer norm
    target_layer = None
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            target_layer = module
            break
    
    if target_layer is None:
        logger.warning("Could not find suitable target layer for feature interaction analysis.")
        return

    gc = GradCAM(model, target_layer)
    
    # Get a batch of data
    features, _ = next(iter(data_loader))
    features = features.to(device)
    
    # Compute average saliency across the batch
    cam = gc.generate_cam(features).mean(dim=0).cpu().numpy()
    gc.remove_hooks()
    
    # Get top 10 indices
    top_10_idx = cam.argsort()[-10:][::-1]
    top_10_idx.sort() # Spectral order
    
    # 2. Compute correlation matrix for these features across the batch
    # features shape: (B, F) or (B, 1, F)
    X = features.cpu().numpy()
    if X.ndim == 3:
        X = X.squeeze(1)
    
    top_features_data = X[:, top_10_idx]
    # Handle case with very small batches
    if top_features_data.shape[0] < 2:
        logger.warning("Batch size too small for correlation analysis.")
        return
        
    corr_matrix = np.corrcoef(top_features_data.T)
    
    # 3. Visualize
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap="RdBu_r", 
        center=0,
        xticklabels=[f"F{i}" for i in top_10_idx],
        yticklabels=[f"F{i}" for i in top_10_idx]
    )
    plt.title("Top 10 Feature Interaction Matrix (Saliency-Guided)")
    
    if ctx:
        ctx.save_figure(plt, "feature_interaction_matrix.png")
    else:
        output_path = Path("outputs/xai")
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path / "feature_interaction_matrix.png")
    plt.close()


def visualize_transformer_attention(model, data_loader, device, ctx=None):
    """
    Extracts and visualizes attention maps from a Transformer or AugFormer model.
    """
    model.eval()
    features, _ = next(iter(data_loader))
    features = features.to(device)

    # Forward pass with attention return
    with torch.no_grad():
        try:
            # Handle standard Transformer/AugFormer interface
            res = model(features, return_attention=True)
            if isinstance(res, tuple):
                _, attentions = res
            else:
                return
        except (TypeError, ValueError, AttributeError):
            return

    if not attentions or not isinstance(attentions, list):
        return

    # Take the last layer's attention
    # shape: (batch, num_heads, seq_len, seq_len)
    last_attn = attentions[-1][0].cpu().numpy()
    num_heads = last_attn.shape[0]
    seq_len = last_attn.shape[1]

    plt.figure(figsize=(15, 5))
    
    # Check if it's AugFormer (usually 7 views)
    is_augformer = hasattr(model, "num_views")
    
    if seq_len > 1:
        # For AugFormer, seq_len is num_views + 1
        labels = None
        if is_augformer and seq_len == model.num_views + 1:
            labels = ["Original"] + [f"Aug {i+1}" for i in range(model.num_views)]
            
        for i in range(min(num_heads, 4)):
            plt.subplot(1, 4, i + 1)
            sns.heatmap(
                last_attn[i], 
                cmap="viridis", 
                xticklabels=labels if labels else "auto",
                yticklabels=labels if labels else "auto"
            )
            plt.title(f"Head {i+1}")
            if labels:
                plt.xticks(rotation=45, ha='right')
            else:
                plt.axis("off")
        
        title_prefix = "AugFormer (Multi-View)" if is_augformer else "Transformer"
        plt.suptitle(f"{title_prefix} Attention Maps (Last Layer)")
    else:
        # For seq_len=1, the attention map is a single scalar.
        heads = [f"Head {i+1}" for i in range(num_heads)]
        values = [last_attn[i, 0, 0] for i in range(num_heads)]
        plt.bar(heads, values, color='skyblue')
        plt.title("Multi-Head Attention Scalar Values (seq_len=1)")
        plt.ylabel("Attention Weight")
        
        if ctx:
            ctx.save_figure(plt, "transformer_attention.png")
        else:
            output_path = Path("outputs/xai")
            output_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path / "transformer_attention.png")
        plt.close()
        
        # Add Feature Interaction Matrix for seq_len=1
        visualize_feature_interaction(model, data_loader, device, ctx)
        return

    if ctx:
        ctx.save_figure(plt, "transformer_attention.png")
    else:
        output_path = Path("outputs/xai")
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path / "transformer_attention.png")
    plt.close()

    # --- Add Attention Rollout ---
    if seq_len > 1:
        rollout = compute_attention_rollout(attentions)
        visualize_attention_rollout(rollout, model, ctx)


def compute_attention_rollout(attentions: List[torch.Tensor]) -> np.ndarray:
    """
    Computes the attention rollout across multiple layers.
    Logic: (A + I) recursive multiplication.
    """
    # 1. Average across heads for each layer
    # attentions is list of (batch, num_heads, seq_len, seq_len)
    res_attn = []
    for attn in attentions:
        # Take first sample in batch, average across heads
        a = attn[0].mean(dim=0).cpu().detach()
        # 2. Account for residual connection (add identity)
        I = torch.eye(a.size(-1))
        a = (a + I) / 2
        res_attn.append(a.numpy())

    # 3. Recursively multiply matrices from first to last layer
    rollout = res_attn[0]
    for i in range(1, len(res_attn)):
        rollout = np.matmul(res_attn[i], rollout)

    return rollout


def visualize_attention_rollout(rollout, model, ctx=None):
    """Plots the attention rollout matrix."""
    plt.figure(figsize=(8, 6))
    
    is_augformer = hasattr(model, "num_views")
    seq_len = rollout.shape[0]
    
    labels = None
    if is_augformer and seq_len == model.num_views + 1:
        labels = ["Original"] + [f"Aug {i+1}" for i in range(model.num_views)]
    
    sns.heatmap(
        rollout, 
        cmap="magma", 
        xticklabels=labels if labels else "auto",
        yticklabels=labels if labels else "auto",
        annot=True if seq_len < 10 else False,
        fmt=".2f"
    )
    
    plt.title("Attention Rollout (Global Information Flow)")
    if labels:
        plt.xticks(rotation=45, ha='right')
        plt.xlabel("Source Views")
        plt.ylabel("Destination Views")
    
    if ctx:
        ctx.save_figure(plt, "attention_rollout.png")
    else:
        output_path = Path("outputs/xai")
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path / "attention_rollout.png")
    plt.close()
