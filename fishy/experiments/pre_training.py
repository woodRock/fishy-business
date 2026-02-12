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
import time
from typing import List, Tuple, Union, Optional, TypeVar, Callable, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from fishy.models.deep.transformer import Transformer
from fishy._core.utils import get_device, RunContext
from fishy._core.factory import create_model
from fishy._core.config import TrainingConfig
from fishy._core.config_loader import load_config

# Type aliases
T_Tensor = TypeVar("T_Tensor", bound=torch.Tensor)
TensorPair = Tuple[T_Tensor, T_Tensor]
Device = Union[str, torch.device]


@dataclass
class PreTrainingConfig:
    """
    Configuration for pre-training tasks.

    Examples:
        >>> config = PreTrainingConfig(num_epochs=5, n_features=100)
        >>> config.num_epochs == 5
        True
        >>> config.n_features == 100
        True

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
    device: Device = get_device()
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

    Examples:
        >>> import torch
        >>> x = torch.ones(4)
        >>> mask_spectra_side(x, "left").tolist()
        [0.0, 0.0, 1.0, 1.0]
        >>> mask_spectra_side(x, "right").tolist()
        [1.0, 1.0, 0.0, 0.0]

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

    Examples:
        >>> import torch.nn as nn
        >>> model = nn.Linear(10, 10)
        >>> config = PreTrainingConfig(n_features=10, device='cpu')
        >>> trainer = PreTrainer(model, config)
        >>> isinstance(trainer.model, nn.Module)
        True

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
        self.logger.info(
            f"{task_name} pre-training finished. Model saved to {model_save_path}"
        )
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
            self.model.fc_out = nn.Linear(in_features, self.config.n_features).to(
                self.config.device
            )
            self.optimizer = AdamW(
                self.model.parameters(), lr=self.config.learning_rate
            )

        def train_step(batch_data):
            x, _ = batch_data
            x = x.to(self.config.device)
            batch_size, n_features = x.shape
            if n_features > self.config.chunk_size:
                start_indices = torch.randint(
                    0,
                    n_features - self.config.chunk_size + 1,
                    (batch_size,),
                    device=self.config.device,
                )
            else:
                start_indices = torch.zeros(
                    (batch_size,), dtype=torch.long, device=self.config.device
                )

            mask = torch.zeros_like(x, dtype=torch.bool)
            for i in range(batch_size):
                start = start_indices[i]
                mask[i, start : start + self.config.chunk_size] = True

            masked_x = x.clone()
            masked_x[mask] = 0
            self.optimizer.zero_grad()
            outputs = self.model(masked_x)
            loss = (
                criterion(outputs[mask], x[mask])
                if torch.any(mask)
                else torch.tensor(0.0, device=self.config.device)
            )
            loss.backward()
            self.optimizer.step()
            return loss.item()

        result_model = self._perform_epoch_loop(
            "MSM", train_loader, train_step, checkpoint_suffix="_msm"
        )[0]
        if original_fc is not None:
            self.model.fc_out = original_fc
            self.optimizer = AdamW(
                self.model.parameters(), lr=self.config.learning_rate
            )
        return result_model

    def pre_train_next_spectra(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> nn.Module:
        """
        Pre-trains the model using Next Spectra Prediction (NSP).
        Learns to distinguish between related (anchor-positive) and unrelated (anchor-negative) masked views.
        """
        self.logger.info("Starting Next Spectra Prediction (NSP) pre-training...")
        train_pairs = self._generate_contrastive_pairs_nsp(train_loader)
        val_pairs = self._generate_contrastive_pairs_nsp(val_loader)
        if not train_pairs:
            return self.model

        criterion = nn.BCEWithLogitsLoss()

        def _run_single_nsp_epoch(pair_list, is_training):
            self.model.train(is_training)
            epoch_loss = 0.0
            for (left, right), label_list in pair_list:
                l_dev, r_dev = left.to(self.config.device), right.to(self.config.device)
                label = torch.tensor(
                    label_list, dtype=torch.float, device=self.config.device
                )
                if is_training:
                    self.optimizer.zero_grad()
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
            self.logger.info(
                f"NSP Epoch [{epoch+1}/{self.config.num_epochs}], TL: {t_loss:.4f}, VL: {v_loss:.4f}"
            )

        torch.save(self.model.state_dict(), f"{self.config.file_path}_nsp.pth")
        return self.model

    def _detect_peaks(
        self, spectra: T_Tensor, peak_threshold: float, window_size: int
    ) -> T_Tensor:
        """Detects peaks using a sliding window approach."""
        batch_size, n_features = spectra.shape
        peak_labels = torch.zeros_like(spectra, dtype=torch.bool, device=spectra.device)
        for i in range(batch_size):
            s = spectra[i]
            padded = F.pad(
                s.unsqueeze(0).unsqueeze(0),
                (window_size // 2, window_size // 2),
                mode="replicate",
            ).squeeze()
            max_i = torch.max(s)
            if max_i == 0:
                continue
            for j in range(n_features):
                if (
                    s[j] == torch.max(padded[j : j + window_size])
                    and s[j] > peak_threshold * max_i
                ):
                    peak_labels[i, j] = True
        return peak_labels

    def pre_train_peak_prediction(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        peak_threshold: float = 0.1,
        window_size: int = 5,
    ) -> nn.Module:
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

        return self._perform_epoch_loop(
            "PeakPred", train_loader, train_step, val_loader, train_step, "_peak_pred"
        )[0]

    def pre_train_denoising_autoencoder(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        noise_std_dev: float = 0.1,
        mask_point_prob: float = 0.05,
    ) -> nn.Module:
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

        return self._perform_epoch_loop(
            "SDA", train_loader, step, val_loader, step, "_sda"
        )[0]

    def pre_train_contrastive_invariance(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        temperature: float = 0.1,
        embedding_dim: int = 128,
    ) -> float:
        """
        Pre-trains the model using Contrastive Transformation Invariance Learning (CTIL).
        Uses NT-Xent loss to maximize similarity between differently augmented views of the same spectrum.
        """
        self.logger.info("Starting CTIL pre-training...")

        def step(batch):
            s, _ = batch
            s = s.to(self.config.device)
            v1 = torch.clamp(s + torch.randn_like(s) * 0.05, 0, 1)  # Simple noise
            v2 = s * (
                torch.rand(s.shape[0], 1, device=s.device) * 0.6 + 0.7
            )  # Simple scale
            self.optimizer.zero_grad()
            z1, z2 = self.model(v1), self.model(v2)
            if isinstance(z1, tuple):
                z1, z2 = z1[0], z2[0]
            loss = self._nt_xent_loss(z1, z2, temperature)
            loss.backward()
            self.optimizer.step()
            return loss.item()

        return self._perform_epoch_loop(
            "CTIL", train_loader, step, val_loader, step, "_ctil"
        )[1]

    def _nt_xent_loss(self, z_i, z_j, temperature=0.1):
        """NT-Xent loss calculation."""
        batch_size = z_i.shape[0]
        z = F.normalize(torch.cat([z_i, z_j], dim=0), p=2, dim=1)
        sim = torch.matmul(z, z.T) / temperature
        labels = (torch.arange(2 * batch_size, device=z_i.device) + batch_size) % (
            2 * batch_size
        )
        sim.masked_fill_(
            torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device),
            float("-inf"),
        )
        return F.cross_entropy(sim, labels)

    def _generate_contrastive_pairs_nsp(self, data_loader):
        """Internal helper for NSP pair generation."""
        pairs = []
        for x, _ in data_loader:
            for i in range(x.shape[0]):
                if random.random() < 0.5:
                    pairs.append(
                        (
                            (
                                mask_spectra_side(x[i], "right"),
                                mask_spectra_side(x[i], "left"),
                            ),
                            [1.0, 0.0],
                        )
                    )
                else:
                    j = random.randint(0, x.shape[0] - 1)
                    pairs.append(
                        (
                            (
                                mask_spectra_side(x[i], "right"),
                                mask_spectra_side(x[j], "left"),
                            ),
                            [0.0, 1.0],
                        )
                    )
        return pairs


class PreTrainingOrchestrator:
    """
    Handles the orchestration of multiple self-supervised pre-training tasks.

    Uses external configuration (`pre_training.yaml`) to define tasks and their
    hyperparameters. Supports weight chaining between sequential tasks.

    Examples:
        >>> from fishy._core.config import TrainingConfig
        >>> from fishy._core.utils import RunContext
        >>> cfg = TrainingConfig()
        >>> ctx = RunContext("ds", "method", "model") # doctest: +ELLIPSIS
        INFO ... Initialized RunContext: model on ds...
        >>> orch = PreTrainingOrchestrator(cfg, torch.device('cpu'), 10, ctx)
        >>> orch.input_dim == 10
        True

    Attributes:
        config (TrainingConfig): Global training configuration.
        device (torch.device): Computation device.
        input_dim (int): Dimensionality of input spectra.
        ctx (RunContext): Context for logging and checkpointing.
        logger (logging.Logger): Logger instance.
        task_configs (List[Dict]): List of task definitions from config.
    """

    def __init__(
        self,
        config: TrainingConfig,
        device: torch.device,
        input_dim: int,
        ctx: RunContext,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initializes the PreTrainingOrchestrator.

        Args:
            config (TrainingConfig): Configuration object.
            device (torch.device): Computing device.
            input_dim (int): Input feature dimension.
            ctx (RunContext): Experiment context.
            logger (Optional[logging.Logger], optional): Custom logger. Defaults to None.
        """
        self.config = config
        self.device = device
        self.input_dim = input_dim
        self.ctx = ctx
        self.logger = logger if logger else logging.getLogger(__name__)

        # Load tasks from configuration
        self.task_configs = load_config("pre_training")["tasks"]

    def run_all(
        self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None
    ) -> Optional[nn.Module]:
        """
        Runs all enabled pre-training tasks sequentially.

        Weights are chained from one task to the next if layers match.

        Args:
            train_loader (DataLoader): Loader for the training data.
            val_loader (Optional[DataLoader], optional): Loader for validation. Defaults to None.

        Returns:
            Optional[nn.Module]: The model after all pre-training tasks, or None if none enabled.
        """
        enabled_tasks = [
            task
            for task in self.task_configs
            if getattr(self.config, task["name"], False)
        ]

        if not enabled_tasks:
            self.logger.info("No pre-training tasks enabled.")
            return None

        self.logger.info(
            f"Enabled pre-training tasks: {', '.join(t['name'] for t in enabled_tasks)}"
        )

        pre_train_cfg = PreTrainingConfig(
            num_epochs=self.config.epochs,
            file_path=self.config.file_path,
            device=self.device,
            n_features=self.input_dim,
        )

        model_after_last_task: Optional[nn.Module] = None

        for task in enabled_tasks:
            flag = task["name"]
            self.logger.info(f"Starting pre-training task: {flag}")

            # Determine output dimension
            if task["output_dim_type"] == "n_features":
                output_dim = self.input_dim
            else:
                output_dim = task["output_dim"]

            current_model = create_model(self.config, self.input_dim, output_dim).to(
                self.device
            )

            if model_after_last_task:
                self._handle_weight_chaining(current_model, model_after_last_task)

            pre_trainer = PreTrainer(
                model=current_model,
                config=pre_train_cfg,
                optimizer=torch.optim.AdamW(
                    current_model.parameters(), lr=self.config.learning_rate
                ),
            )

            call_args = [train_loader]
            if task["requires_val"]:
                if val_loader is None:
                    self.logger.warning(
                        f"Validation loader for {flag} not found, passing None."
                    )
                call_args.append(val_loader)

            start_time = time.time()
            trained_model = getattr(pre_trainer, task["method"])(
                *call_args, **task["kwargs"]
            )
            self.logger.info(f"{flag} training time: {time.time() - start_time:.2f}s")

            # Save pre-trained checkpoint
            checkpoint_path = self.ctx.get_checkpoint_path(f"pretrained_{flag}.pth")
            torch.save(trained_model.state_dict(), checkpoint_path)
            self.logger.info(
                f"Pre-trained weights for {flag} saved to {checkpoint_path}"
            )

            model_after_last_task = trained_model

        return model_after_last_task

    def _handle_weight_chaining(
        self, current_model: nn.Module, prev_model: nn.Module
    ) -> None:
        """
        Copies compatible weights from the previous model to the current one.

        Args:
            current_model (nn.Module): The model to load weights into.
            prev_model (nn.Module): The model to copy weights from.
        """
        self.logger.info(f"Attempting weight chaining for {self.config.model}")
        try:
            prev_state_dict = prev_model.state_dict()
            current_model_dict = current_model.state_dict()

            load_state_dict = {
                k: v
                for k, v in prev_state_dict.items()
                if k in current_model_dict and v.shape == current_model_dict[k].shape
            }

            missing_keys, unexpected_keys = current_model.load_state_dict(
                load_state_dict, strict=False
            )

            if missing_keys:
                self.logger.debug(f"Chaining: Missing keys: {missing_keys}")
            if unexpected_keys:
                self.logger.debug(f"Chaining: Unexpected keys: {unexpected_keys}")

            self.logger.info("Weight chaining: successfully loaded compatible weights.")
        except Exception as e:
            self.logger.warning(
                f"Weight chaining failed: {e}. Model will train from scratch."
            )

    def adapt_for_finetuning(
        self, model: nn.Module, pre_trained_model: nn.Module
    ) -> None:
        """
        Adapates a pre-trained model for fine-tuning by loading compatible weights.

        Args:
            model (nn.Module): The target model for fine-tuning.
            pre_trained_model (nn.Module): The model containing pre-trained weights.
        """
        self.logger.info("Adapting pre-trained model for fine-tuning...")
        self._handle_weight_chaining(model, pre_trained_model)
