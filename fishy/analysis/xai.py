# -*- coding: utf-8 -*-
"""
Unified XAI module for deep learning model explanations using LIME, Grad-CAM,
and the RBN-native relational signature explainer.
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


# ---------------------------------------------------------------------------
# RBN-native explainer: class-conditional relational signature
# ---------------------------------------------------------------------------


def _attn_scores_from_module(attn_module: nn.Module, bindings: torch.Tensor) -> torch.Tensor:
    """
    Recompute softmax attention weights from a SecondOrderRelationalAttention
    or SparseSecondOrderAttention module given already-computed bindings.

    Returns: [B, H, K, K] attention weight tensor.
    """
    B, K, _ = bindings.shape
    H = attn_module.n_heads
    d = attn_module.d_head

    q = attn_module.q_proj(bindings).view(B, K, H, d).transpose(1, 2)  # [B, H, K, d]
    k = attn_module.k_proj(bindings).view(B, K, H, d).transpose(1, 2)

    w1 = attn_module.mlp_rel[0]
    q_p = torch.matmul(q, w1.weight[:, :d].t())
    k_p = torch.matmul(k, w1.weight[:, d : 2 * d].t())
    w1_qk = w1.weight[:, 2 * d :].t()

    res = q_p.unsqueeze(3) + k_p.unsqueeze(2) + w1.bias
    res = res + torch.matmul(q.unsqueeze(3) * k.unsqueeze(2), w1_qk)
    scores = attn_module.mlp_rel[2](attn_module.mlp_rel[1](res)).squeeze(-1)  # [B, H, K, K]
    return torch.softmax(scores / (d**0.5), dim=-1)


class RBNExplainer:
    """
    Generates class-conditional relational signatures for RBN and RBNPlus.

    For each class the explainer:
    1. Selects one representative instance.
    2. Propagates it through all-but-last reasoning layers.
    3. Extracts the [K, K] attention matrix from the final layer (averaged over
       heads) — the model's pairwise peak-relationship fingerprint for that class.
    4. Extracts the task-query readout attention — per-peak contribution to the
       final representation.
    5. Saves a composite figure: heatmap (relational signature) + marginal bar
       (readout weights), both indexed by m/z value.
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader,
        class_names: List[str],
        device: str,
        feature_names=None,
    ):
        self.model = model
        self.data_loader = data_loader
        self.class_names = class_names
        self.device = device
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.model.to(device)
        self.model.eval()

        # Detect variant by attribute name
        self._is_plus = hasattr(model, "encoder")  # RBNPlus uses self.encoder

    def _get_one_per_class(self) -> Dict[int, torch.Tensor]:
        """Return {class_idx: [1, C] tensor} for each class."""
        n_classes = len(self.class_names)
        found: Dict[int, torch.Tensor] = {}
        for features, labels in self.data_loader:
            for i in range(len(features)):
                lbl = labels[i]
                cls = int(torch.argmax(lbl).item()) if lbl.dim() > 0 and lbl.shape[0] > 1 else int(lbl.item())
                if cls not in found:
                    found[cls] = features[i].unsqueeze(0)
            if len(found) == n_classes:
                break
        return found

    def _encode_bindings(self, x: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Run the binding encoder for one instance.
        Returns (bindings [1, K, D], top_mz_values [K]).
        """
        x = x.to(self.device)
        if x.dim() == 3:
            x = x.squeeze(1) if x.shape[1] == 1 else x.squeeze(2)
        B, C = x.shape
        x_log = torch.log1p(x.clamp(min=0.0))

        if self._is_plus:
            top_k = self.model.top_k
            if top_k and top_k < C:
                vals, top_idx = torch.topk(x_log, top_k, dim=1)
                x_in = vals
            else:
                top_idx = torch.arange(C, device=self.device).unsqueeze(0).expand(B, -1)
                x_in = x_log
            curr_mz = top_idx.float()
            bindings, _, _ = self.model.encoder(x_in, top_idx, curr_mz, x_full=x_log)
        else:
            top_k = self.model.top_k
            if top_k and top_k < C:
                _, top_idx = torch.topk(x_log, top_k, dim=1)
                x_sparse = torch.gather(x_log, 1, top_idx)
                roles = self.model.binding_encoder._make_roles(top_idx)
                fillers = self.model.binding_encoder.filler_encoder(x_sparse.unsqueeze(-1))
                if self.model.binding_encoder.binding_mode == "outer_product":
                    outer = torch.einsum("bcd,bce->bcde", roles, fillers)
                    bindings = self.model.binding_encoder.outer_proj(outer.flatten(-2))
                else:
                    bindings = roles * fillers
            else:
                top_idx = torch.arange(C, device=self.device).unsqueeze(0).expand(B, -1)
                bindings, _, _ = self.model.binding_encoder(x_log)

        top_idx_np = top_idx[0].cpu().numpy()
        if self.feature_names is not None:
            try:
                mz_vals = np.array([float(self.feature_names[i]) for i in top_idx_np])
            except (IndexError, ValueError):
                mz_vals = top_idx_np.astype(float)
        else:
            mz_vals = top_idx_np.astype(float)

        return bindings, mz_vals

    def _propagate_intermediate(self, bindings: torch.Tensor) -> torch.Tensor:
        """Run all reasoning layers except the last."""
        if self._is_plus:
            layers = list(self.model.layers)
            for layer in layers[:-1]:
                b = layer["norm1"](bindings + layer["attn"](bindings))
                bindings = layer["norm2"](b + layer["ffn"](b))
            last_attn = layers[-1]["attn"]
        else:
            layers = list(self.model.reasoning_layers)
            for layer in layers[:-1]:
                bindings = layer(bindings)
            last_attn = layers[-1].rel_attn
        return bindings, last_attn

    def _readout_weights(self, bindings: torch.Tensor) -> np.ndarray:
        """Extract per-peak readout attention weights. Returns [K] array."""
        readout = self.model.readout
        B = bindings.shape[0]
        D = bindings.shape[-1]
        q = readout.task_query.expand(B, -1, -1)
        scores = torch.bmm(q, bindings.transpose(1, 2)) / (D**0.5)
        return torch.softmax(scores, dim=-1).squeeze(1).mean(dim=0).detach().cpu().numpy()

    def _plot_signature(
        self,
        attn_matrix: np.ndarray,
        readout_weights: np.ndarray,
        mz_vals: np.ndarray,
        class_name: str,
        output_path: Path,
    ):
        K = attn_matrix.shape[0]
        step = max(1, K // 20)
        ticks = list(range(0, K, step))
        tick_labels = [f"{mz_vals[t]:.1f}" for t in ticks]

        fig = plt.figure(figsize=(15, 6))

        # --- Relational attention heatmap ---
        ax_heat = fig.add_axes([0.07, 0.14, 0.58, 0.78])
        im = ax_heat.imshow(attn_matrix, aspect="auto", cmap="viridis", interpolation="nearest")
        plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04, label="Attention weight")
        ax_heat.set_xticks(ticks)
        ax_heat.set_xticklabels(tick_labels, rotation=90, fontsize=6)
        ax_heat.set_yticks(ticks)
        ax_heat.set_yticklabels(tick_labels, fontsize=6)
        ax_heat.set_xlabel("Key peak (m/z)", fontsize=9)
        ax_heat.set_ylabel("Query peak (m/z)", fontsize=9)
        ax_heat.set_title(
            f"Relational Signature  —  class: {class_name}", fontsize=11, fontweight="bold"
        )

        # --- Task-query readout (marginal bar) ---
        ax_bar = fig.add_axes([0.72, 0.14, 0.25, 0.78])
        ax_bar.barh(range(K), readout_weights, color="steelblue", alpha=0.85)
        ax_bar.set_yticks(ticks)
        ax_bar.set_yticklabels(tick_labels, fontsize=6)
        ax_bar.yaxis.tick_right()
        ax_bar.yaxis.set_label_position("right")
        ax_bar.set_ylim(K - 1, 0)  # match heatmap y-direction
        ax_bar.set_xlabel("Readout weight", fontsize=9)
        ax_bar.set_title("Task readout", fontsize=9)

        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def run(self, output_dir: Path) -> List[str]:
        """
        Generate one relational-signature figure per class.
        Returns list of saved file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        instances = self._get_one_per_class()
        saved = []

        with torch.no_grad():
            for cls_idx, x in sorted(instances.items()):
                class_name = (
                    self.class_names[cls_idx]
                    if cls_idx < len(self.class_names)
                    else str(cls_idx)
                )
                bindings, mz_vals = self._encode_bindings(x)
                bindings, last_attn_module = self._propagate_intermediate(bindings)

                # [K, K] — average over batch (1) and heads
                attn_matrix = (
                    _attn_scores_from_module(last_attn_module, bindings)
                    .mean(dim=(0, 1))
                    .cpu()
                    .numpy()
                )
                readout_weights = self._readout_weights(bindings)

                safe_name = class_name.lower().replace(" ", "_").replace("/", "-")
                out_path = output_dir / f"rbn_relational_signature_{safe_name}.png"
                self._plot_signature(attn_matrix, readout_weights, mz_vals, class_name, out_path)
                logger.info(f"Saved RBN relational signature for '{class_name}' -> {out_path}")
                saved.append(str(out_path))

        return saved


def run_rbn_explanation(
    model: nn.Module,
    data_loader,
    class_names: List[str],
    device: str,
    feature_names=None,
    output_dir: Optional[Path] = None,
) -> List[str]:
    """
    Entry point for RBN/RBNPlus-native XAI.

    Generates one class-conditional relational-signature figure per class and
    saves them to *output_dir* (defaults to ``outputs/xai``).

    Returns the list of saved file paths.
    """
    if output_dir is None:
        output_dir = Path("outputs/xai")
    explainer = RBNExplainer(model, data_loader, class_names, device, feature_names)
    return explainer.run(output_dir)
