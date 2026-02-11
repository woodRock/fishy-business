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
        self.forward_hook = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output[0].detach() if isinstance(output, tuple) else output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.forward_hook.remove(); self.backward_hook.remove()

    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None):
        self.model.eval()
        if input_tensor.dim() == 2: input_tensor = input_tensor.unsqueeze(1)
        model_output = self.model(input_tensor)
        if isinstance(model_output, tuple): model_output = model_output[0]
        if target_class is None: target_class = torch.argmax(model_output, dim=1)
        one_hot = torch.zeros_like(model_output)
        one_hot.scatter_(1, target_class.unsqueeze(1), 1.0)
        self.model.zero_grad(); model_output.backward(gradient=one_hot, retain_graph=True)
        if self.activations is None or self.gradients is None:
            return torch.zeros((input_tensor.shape[0], input_tensor.shape[2]), device=input_tensor.device)
        if isinstance(self.target_layer, nn.Linear):
            pooled_gradients = torch.mean(self.gradients, dim=0)
            for i in range(self.activations.shape[-1]): self.activations[:, i] *= pooled_gradients[i]
            cam = self.activations
        else:
            pooled_gradients = torch.mean(self.gradients, dim=[0, 1])
            for i in range(self.activations.shape[-1]): self.activations[:, :, i] *= pooled_gradients[i]
            cam = torch.sum(self.activations, dim=1)
        cam = F.relu(cam)
        if cam.shape[1] != input_tensor.shape[2]:
            cam = F.interpolate(cam.unsqueeze(1), size=input_tensor.shape[2], mode="linear", align_corners=False).squeeze(1)
        for i in range(cam.shape[0]):
            if torch.max(cam[i]) > 0: cam[i] = (cam[i] - torch.min(cam[i])) / torch.max(cam[i])
        return cam

class ModelWrapper:
    """Wrapper for models to make them compatible with LIME (handles both Torch and Sklearn)."""

    def __init__(self, model: Any, device: str) -> None:
        self.model = model
        self.device = device
        # Safely handle PyTorch-specific methods
        if hasattr(model, "to"):
            try: self.model = model.to(device)
            except: pass
        if hasattr(model, "eval"):
            try: self.model.eval()
            except: pass

    def normalize_intensities(self, x: np.ndarray) -> np.ndarray:
        x_norm = np.zeros_like(x)
        for i in range(len(x)):
            low, high = np.min(x[i]), np.max(x[i])
            if high - low > 0: x_norm[i] = (x[i] - low) / (high - low)
        return x_norm

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        try:
            x_norm = self.normalize_intensities(x)
            # 1. Handle as a PyTorch model
            if isinstance(self.model, nn.Module):
                x_tensor = torch.tensor(x_norm, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    out = self.model(x_tensor)
                    if isinstance(out, tuple): out = out[0]
                    return F.softmax(out, dim=-1).cpu().numpy()
            
            # 2. Try as a standard sklearn-like model if it has predict_proba
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(x_norm)
            
            # 3. Fallback for models that only have predict (e.g. some regressors used as classifiers)
            if hasattr(self.model, "predict"):
                preds = self.model.predict(x_norm)
                # Convert to dummy probabilities if needed
                return np.eye(2)[preds.astype(int)] # Very basic fallback
                
            raise AttributeError("Model has neither PyTorch interface nor predict_proba.")
        except Exception as e:
            logger.error(f"Error in predict_proba: {e}"); raise

def run_lime_explanation(dataset_name, model, data_module, explainer_config, instance_name, target_label, ctx):
    wrapper = ModelWrapper(model, explainer_config.device)
    features, labels = next(iter(data_module.get_train_dataloader()))
    explainer = LimeTabularExplainer(features.cpu().numpy(), feature_names=[f"{float(x):.4f}" for x in data_module.get_train_dataframe().columns[1:]], discretize_continuous=True, random_state=explainer_config.random_seed)
    exp = explainer.explain_instance(features[0].cpu().numpy().flatten(), wrapper.predict_proba, num_features=explainer_config.num_features)
    output_path = ctx.figure_dir / f"lime_{dataset_name}_{instance_name}.png"
    exp.as_pyplot_figure(); plt.savefig(output_path); plt.close()

def run_gradcam_analysis(model, data_loader, device, output_dir, target_layer=None, ctx=None):
    if target_layer is None:
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, (nn.Conv1d, nn.Linear)): target_layer = module; break
    if target_layer is None: raise ValueError("Could not find target layer.")
    gc = GradCAM(model, target_layer); features, _ = next(iter(data_loader))
    cam = gc.generate_cam(features.to(device)).mean(dim=0).cpu().numpy()
    plt.figure(figsize=(10, 6)); plt.plot(cam); plt.title("Grad-CAM Importance")
    if ctx: ctx.save_figure(plt, "avg_gradcam.png")
    else: (output_dir).mkdir(parents=True, exist_ok=True); plt.savefig(output_dir / "avg_gradcam.png")
    plt.close(); gc.remove_hooks()
