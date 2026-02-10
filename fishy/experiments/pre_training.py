# -*- coding: utf-8 -*-
"""
Pre-training strategies for models on mass spectrometry data.

This module implements various pre-training tasks such as masked spectra modeling,
next spectra prediction, peak detection, denoising autoencoding, peak parameter
regression, segment reordering, and contrastive invariance learning. These tasks
are designed to leverage unlabeled or semi-labeled mass spectrometry data to
learn useful representations for downstream tasks.
"""

from dataclasses import dataclass
import logging
import random
from typing import List, Tuple, Union, Optional, TypeVar, Callable, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from fishy.models.deep import Transformer

# Type aliases
T_Tensor = TypeVar("T_Tensor", bound=torch.Tensor)
TensorPair = Tuple[T_Tensor, T_Tensor]
Device = Union[str, torch.device]


@dataclass
class PreTrainingConfig:
    """
    Configuration for pre-training tasks.

    Attributes:
        num_epochs (int): Number of epochs to train for each task.
        file_path (str): Path to save the model checkpoints.
        n_features (int): Number of input features (spectrum length).
        chunk_size (int): Size of the contiguous mask for MSM.
        device (Device): Device to run training on.
        learning_rate (float): Learning rate for the optimizer.
        noise_enabled (bool): Enable noise during augmentation for CTIL.
        shift_enabled (bool): Enable shift during augmentation for CTIL.
        scale_enabled (bool): Enable scale during augmentation for CTIL.
        crop_enabled (bool): Enable crop during augmentation for CTIL.
        flip_enabled (bool): Enable flip during augmentation for CTIL.
        permutation_enabled (bool): Enable permutation during augmentation for CTIL.
        crop_size (float): Portion of spectrum to keep when cropping.
    """
    num_epochs: int = 100
    file_path: str = "transformer_checkpoint.pth"
    n_features: int = 2080
    chunk_size: int = 50
    device: Device = "cuda" if torch.cuda.is_available() else "cpu"
    learning_rate: float = 1e-4
    noise_enabled: bool = False
    shift_enabled: bool = False
    scale_enabled: bool = False
    crop_enabled: bool = False
    flip_enabled: bool = False
    permutation_enabled: bool = False
    crop_size: float = 0.8


def mask_spectra_side(input_spectra: T_Tensor, side: str = "left") -> T_Tensor:
    """
    Masks either the left or right side of the input spectra.

    Args:
        input_spectra (T_Tensor): The input spectra tensor to mask.
        side (str): The side to mask, either 'left' or 'right'.

    Returns:
        T_Tensor: The masked spectra tensor.
    """
    if side not in ["left", "right"]:
        raise ValueError("Side must be either 'left' or 'right'")
    split_index = input_spectra.shape[0] // 2
    masked_spectra = input_spectra.clone()
    if side == "left":
        masked_spectra[:split_index] = 0
    else:
        masked_spectra[split_index:] = 0
    return masked_spectra


class PreTrainer:
    """
    Handles the execution of various self-supervised pre-training tasks.

    Args:
        model (nn.Module): The model to be pre-trained.
        config (PreTrainingConfig): Configuration parameters for the tasks.
        optimizer (Optional[torch.optim.Optimizer]): Optimizer to use. If None, AdamW is initialized.
        logger (Optional[logging.Logger]): Logger instance.
    """
    def __init__(
        self,
        model: nn.Module,
        config: PreTrainingConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.logger = logger if logger else logging.getLogger(__name__)
        self.optimizer = (
            optimizer
            if optimizer
            else AdamW(model.parameters(), lr=config.learning_rate)
        )

    def _perform_epoch_loop(
        self,
        task_name: str,
        train_loader: DataLoader,
        train_step_fn: Callable[[Any], float],
        val_loader: Optional[DataLoader] = None,
        val_step_fn: Optional[Callable[[Any], float]] = None,
        checkpoint_suffix: str = "",
    ) -> Tuple[nn.Module, float]:
        """Generic epoch loop for a pre-training task."""
        final_avg_train_loss = 0.0
        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_train_loss = 0.0
            for batch_data in tqdm(
                train_loader, desc=f"{task_name} Epoch {epoch+1} [Train]", leave=False
            ):
                loss = train_step_fn(batch_data)
                total_train_loss += loss
            avg_train_loss = total_train_loss / len(train_loader)
            final_avg_train_loss = avg_train_loss
            log_msg = f"{task_name} Epoch [{epoch+1}/{self.config.num_epochs}], Train Loss: {avg_train_loss:.4f}"

            if val_loader and val_step_fn:
                self.model.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for batch_data_val in val_loader:
                        val_metric = val_step_fn(batch_data_val)
                        total_val_loss += val_metric
                avg_val_metric = total_val_loss / len(val_loader)
                log_msg += f", Val Loss: {avg_val_metric:.4f}"
            self.logger.info(log_msg)

        model_save_path = f"{self.config.file_path}{checkpoint_suffix}.pth"
        torch.save(self.model.state_dict(), model_save_path)
        self.logger.info(f"{task_name} pre-training finished. Model saved to {model_save_path}")
        return self.model, final_avg_train_loss

    def pre_train_masked_spectra(self, train_loader: DataLoader) -> nn.Module:
        """
        Pre-trains the model using Masked Spectra Modelling (MSM).
        The model attempts to reconstruct contiguous chunks of masked spectral data.
        """
        criterion = nn.MSELoss()
        self.logger.info("Starting Masked Spectra Modelling pre-training...")

        original_fc = None
        if hasattr(self.model, "fc_out") and isinstance(self.model.fc_out, nn.Linear):
            self.logger.info("MSM: Adapting model's final layer for reconstruction.")
            original_fc = self.model.fc_out
            in_features = self.model.fc_out.in_features
            self.model.fc_out = nn.Linear(in_features, self.config.n_features).to(self.config.device)
            self.optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)

        def train_step(batch_data):
            x, _ = batch_data
            x = x.to(self.config.device)
            batch_size, n_features = x.shape
            if n_features > self.config.chunk_size:
                start_indices = torch.randint(0, n_features - self.config.chunk_size + 1, (batch_size,), device=self.config.device)
            else:
                start_indices = torch.zeros((batch_size,), dtype=torch.long, device=self.config.device)

            mask = torch.zeros_like(x, dtype=torch.bool)
            for i in range(batch_size):
                start = start_indices[i]
                mask[i, start:start+self.config.chunk_size] = True

            masked_x = x.clone()
            masked_x[mask] = 0
            self.optimizer.zero_grad()
            outputs = self.model(masked_x)
            loss = criterion(outputs[mask], x[mask]) if torch.any(mask) else torch.tensor(0.0, device=self.config.device)
            loss.backward()
            self.optimizer.step()
            return loss.item()

        result_model = self._perform_epoch_loop("MSM", train_loader, train_step, checkpoint_suffix="_msm")[0]
        if original_fc is not None:
            self.model.fc_out = original_fc
            self.optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
        return result_model

    def pre_train_next_spectra(self, train_loader: DataLoader, val_loader: DataLoader) -> nn.Module:
        """
        Pre-trains the model using Next Spectra Prediction (NSP).
        Learns to distinguish between related (anchor-positive) and unrelated (anchor-negative) masked views.
        """
        self.logger.info("Starting Next Spectra Prediction (NSP) pre-training...")
        train_pairs = self._generate_contrastive_pairs_nsp(train_loader)
        val_pairs = self._generate_contrastive_pairs_nsp(val_loader)
        if not train_pairs: return self.model

        criterion = nn.BCEWithLogitsLoss()

        def _run_single_nsp_epoch(pair_list, is_training):
            self.model.train(is_training)
            epoch_loss = 0.0
            for (left, right), label_list in pair_list:
                l_dev, r_dev = left.to(self.config.device), right.to(self.config.device)
                label = torch.tensor(label_list, dtype=torch.float, device=self.config.device)
                if is_training: self.optimizer.zero_grad()
                output = self.model(l_dev.unsqueeze(0), r_dev.unsqueeze(0))
                loss = criterion(output, label.unsqueeze(0))
                if is_training:
                    loss.backward()
                    self.optimizer.step()
                epoch_loss += loss.item()
            return epoch_loss / len(pair_list)

        for epoch in range(self.config.num_epochs):
            t_loss = _run_single_nsp_epoch(train_pairs, True)
            v_loss = _run_single_nsp_epoch(val_pairs, False) if val_pairs else 0.0
            self.logger.info(f"NSP Epoch [{epoch+1}/{self.config.num_epochs}], TL: {t_loss:.4f}, VL: {v_loss:.4f}")

        torch.save(self.model.state_dict(), f"{self.config.file_path}_nsp.pth")
        return self.model

    def _detect_peaks(self, spectra: T_Tensor, peak_threshold: float, window_size: int) -> T_Tensor:
        """Detects peaks using a sliding window approach."""
        batch_size, n_features = spectra.shape
        peak_labels = torch.zeros_like(spectra, dtype=torch.bool, device=spectra.device)
        for i in range(batch_size):
            s = spectra[i]
            padded = F.pad(s.unsqueeze(0).unsqueeze(0), (window_size // 2, window_size // 2), mode="replicate").squeeze()
            max_i = torch.max(s)
            if max_i == 0: continue
            for j in range(n_features):
                if s[j] == torch.max(padded[j : j + window_size]) and s[j] > peak_threshold * max_i:
                    peak_labels[i, j] = True
        return peak_labels

    def pre_train_peak_prediction(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, peak_threshold: float = 0.1, window_size: int = 5) -> nn.Module:
        """
        Pre-trains the model for Peak Prediction.
        Point-wise binary classification task to identify spectral peaks.
        """
        self.logger.info("Starting Peak Prediction pre-training...")
        def train_step(batch):
            s, _ = batch
            s = s.to(self.config.device)
            targets = self._detect_peaks(s, peak_threshold, window_size).float()
            self.optimizer.zero_grad()
            preds = self.model(s, s)
            loss = F.binary_cross_entropy_with_logits(preds, targets)
            loss.backward()
            self.optimizer.step()
            return loss.item()
        return self._perform_epoch_loop("PeakPred", train_loader, train_step, val_loader, train_step, "_peak_pred")[0]

    def pre_train_denoising_autoencoder(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, noise_std_dev: float = 0.1, mask_point_prob: float = 0.05) -> nn.Module:
        """
        Pre-trains the model using Spectrum Denoising Autoencoding (SDA).
        Model learns to reconstruct clean spectra from inputs corrupted by noise and point masking.
        """
        self.logger.info("Starting SDA pre-training...")
        criterion = nn.MSELoss()
        def step(batch):
            s, _ = batch
            clean = s.to(self.config.device)
            noisy = torch.clamp(clean + torch.randn_like(clean) * noise_std_dev, 0, 1)
            noisy[torch.rand_like(noisy) < mask_point_prob] = 0
            self.optimizer.zero_grad()
            preds = self.model(noisy, noisy)
            loss = criterion(preds, clean)
            loss.backward()
            self.optimizer.step()
            return loss.item()
        return self._perform_epoch_loop("SDA", train_loader, step, val_loader, step, "_sda")[0]

    def pre_train_contrastive_invariance(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, temperature: float = 0.1, embedding_dim: int = 128) -> float:
        """
        Pre-trains the model using Contrastive Transformation Invariance Learning (CTIL).
        Uses NT-Xent loss to maximize similarity between differently augmented views of the same spectrum.
        """
        self.logger.info("Starting CTIL pre-training...")
        def step(batch):
            s, _ = batch
            s = s.to(self.config.device)
            v1 = torch.clamp(s + torch.randn_like(s) * 0.05, 0, 1) # Simple noise
            v2 = s * (torch.rand(s.shape[0], 1, device=s.device) * 0.6 + 0.7) # Simple scale
            self.optimizer.zero_grad()
            z1, z2 = self.model(v1), self.model(v2)
            if isinstance(z1, tuple): z1, z2 = z1[0], z2[0]
            loss = self._nt_xent_loss(z1, z2, temperature)
            loss.backward()
            self.optimizer.step()
            return loss.item()
        return self._perform_epoch_loop("CTIL", train_loader, step, val_loader, step, "_ctil")[1]

    def _nt_xent_loss(self, z_i, z_j, temperature=0.1):
        """NT-Xent loss calculation."""
        batch_size = z_i.shape[0]
        z = F.normalize(torch.cat([z_i, z_j], dim=0), p=2, dim=1)
        sim = torch.matmul(z, z.T) / temperature
        labels = (torch.arange(2 * batch_size, device=z_i.device) + batch_size) % (2 * batch_size)
        sim.masked_fill_(torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device), float("-inf"))
        return F.cross_entropy(sim, labels)

    def _generate_contrastive_pairs_nsp(self, data_loader):
        """Internal helper for NSP pair generation."""
        pairs = []
        for x, _ in data_loader:
            for i in range(x.shape[0]):
                if random.random() < 0.5:
                    pairs.append(((mask_spectra_side(x[i], "right"), mask_spectra_side(x[i], "left")), [1.0, 0.0]))
                else:
                    j = random.randint(0, x.shape[0]-1)
                    pairs.append(((mask_spectra_side(x[i], "right"), mask_spectra_side(x[j], "left")), [0.0, 1.0]))
        return pairs