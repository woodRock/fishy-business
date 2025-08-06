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

# from torch.nn import CrossEntropyLoss # Used directly
from torch.optim import AdamW
from torch.utils.data import (
    DataLoader,
)  # TensorDataset not used directly in this snippet
from tqdm import tqdm

from models import Transformer  # Assuming this is your base model for some tasks

# Type aliases
T_Tensor = TypeVar("T_Tensor", bound=torch.Tensor)  # Renamed for clarity
TensorPair = Tuple[T_Tensor, T_Tensor]
Device = Union[str, torch.device]


@dataclass
class PreTrainingConfig:
    """Configuration for pre-training tasks."""

    num_epochs: int = 100
    file_path: str = "transformer_checkpoint.pth"  # Default path
    n_features: int = 2080
    chunk_size: int = 50  # For Masked Spectra Modelling
    device: Device = "cuda" if torch.cuda.is_available() else "cpu"
    learning_rate: float = 1e-4  # Added for default optimizer


def mask_spectra_side(input_spectra: T_Tensor, side: str = "left") -> T_Tensor:
    """Masks either the left or right side of the input spectra.

    Args:
        input_spectra (T_Tensor): The input spectra tensor to mask.
        side (str): The side to mask, either 'left' or 'right'.

    Returns:
        T_Tensor: The masked spectra tensor.
    """
    if side not in ["left", "right"]:
        raise ValueError("Side must be either 'left' or 'right'")
    split_index = (
        input_spectra.shape[0] // 2
    )  # Assumes input_spectra is a single spectrum
    masked_spectra = input_spectra.clone()
    if side == "left":
        masked_spectra[:split_index] = 0
    else:
        masked_spectra[split_index:] = 0
    return masked_spectra


class PreTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: PreTrainingConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initializes the PreTrainer with a model, configuration, and optional optimizer and logger.

        Args:
            model (nn.Module): The model to be pre-trained.
            config (PreTrainingConfig): Configuration for pre-training tasks.
            optimizer (Optional[torch.optim.Optimizer]): Optimizer for training. If None, AdamW is used.
            logger (Optional[logging.Logger]): Logger for logging messages. If None, a default logger is created.
        """
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
        train_step_fn: Callable[[Any], float],  # Takes batch, returns loss
        val_loader: Optional[DataLoader] = None,
        val_step_fn: Optional[
            Callable[[Any], float]
        ] = None,  # Takes batch, returns loss/metric
        checkpoint_suffix: str = "",
    ) -> nn.Module:
        """Generic epoch loop for a pre-training task.

        Args:
            task_name (str): Name of the pre-training task for logging.
            train_loader (DataLoader): DataLoader for training data.
            train_step_fn (Callable): Function to perform a training step, taking a batch and returning loss.
            val_loader (Optional[DataLoader]): DataLoader for validation data. If None, no validation is performed.
            val_step_fn (Optional[Callable]): Function to perform a validation step, taking a batch and returning loss/metric.
            checkpoint_suffix (str): Suffix to append to the model save path.

        Returns:
            nn.Module: The trained model after the epoch loop.
        """
        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_train_loss = 0.0
            for batch_data in tqdm(
                train_loader, desc=f"{task_name} Epoch {epoch+1} [Train]", leave=False
            ):
                loss = train_step_fn(batch_data)
                total_train_loss += loss
            avg_train_loss = total_train_loss / len(train_loader)
            log_msg = f"{task_name} Epoch [{epoch+1}/{self.config.num_epochs}], Train Loss: {avg_train_loss:.4f}"

            if val_loader and val_step_fn:
                self.model.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for batch_data_val in val_loader:
                        val_metric = val_step_fn(
                            batch_data_val
                        )  # Can be loss or other metric
                        total_val_loss += val_metric
                avg_val_metric = total_val_loss / len(
                    val_loader
                )  # Assuming it's loss for now
                log_msg += f", Val Loss: {avg_val_metric:.4f}"  # Adjust if val_step_fn returns other metrics
            self.logger.info(log_msg)

        model_save_path = f"{self.config.file_path}{checkpoint_suffix}.pth"
        torch.save(self.model.state_dict(), model_save_path)
        self.logger.info(
            f"{task_name} pre-training finished. Model saved to {model_save_path}"
        )
        return self.model

    def pre_train_masked_spectra(self, train_loader: DataLoader) -> nn.Module:
        """Pre-trains the model using Masked Spectra Modelling (MSM).

        Args:
            train_loader (DataLoader): DataLoader for training data.

        Returns:
            nn.Module: The trained model after pre-training.
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
        else:
            self.logger.warning(
                "MSM: Model does not have 'fc_out: nn.Linear'. Assuming output is already configured for reconstruction."
            )

        def train_step(batch_data):
            x, _ = batch_data
            x = x.to(self.config.device)
            batch_size, n_features = x.shape

            # For each item in the batch, select a random chunk to mask
            if n_features > self.config.chunk_size:
                start_indices = torch.randint(0, n_features - self.config.chunk_size + 1, (batch_size,), device=self.config.device)
            else:
                start_indices = torch.zeros((batch_size,), dtype=torch.long, device=self.config.device)

            mask = torch.zeros_like(x, dtype=torch.bool)
            for i in range(batch_size):
                start = start_indices[i]
                end = start + self.config.chunk_size
                mask[i, start:end] = True

            masked_x = x.clone()
            masked_x[mask] = 0

            self.optimizer.zero_grad()
            outputs = self.model(masked_x)
            
            # The loss is calculated only on the masked region
            if torch.any(mask):
                loss = criterion(outputs[mask], x[mask])
            else:
                loss = torch.tensor(0.0, device=self.config.device)

            loss.backward()
            self.optimizer.step()
            return loss.item()

        result_model = self._perform_epoch_loop(
            "MSM", train_loader, train_step, checkpoint_suffix="_msm"
        )

        if original_fc is not None:
            self.logger.info("MSM: Restoring model's original final layer.")
            self.model.fc_out = original_fc
            self.optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)

        return result_model

    def _generate_contrastive_pairs_nsp(
        self, data_loader: DataLoader
    ) -> List[Tuple[TensorPair, List[float]]]:
        """Generates contrastive pairs for Next Spectra Prediction (NSP).

        Args:
            data_loader (DataLoader): DataLoader for the dataset.

        Returns:
            List[Tuple[TensorPair, List[float]]]: A list of tuples containing pairs of masked spectra and their labels.
        """
        # Note: Materializing all pairs can be memory intensive for large datasets.
        pairs = []
        all_x_batches = [x_batch for x_batch, _ in data_loader]
        if not all_x_batches:
            return []
        all_x = torch.cat(all_x_batches, dim=0)  # Concatenate all data first
        num_samples = all_x.shape[0]
        if num_samples == 0:
            return []

        for i in range(num_samples):
            # Use a single spectrum as input to mask_spectra_side
            if random.random() < 0.5 and i < num_samples - 1:  # Positive pair
                left = mask_spectra_side(all_x[i], "right")
                right = mask_spectra_side(all_x[i], "left")
                pairs.append(((left, right), [1.0, 0.0]))  # Use float for labels
            elif num_samples > 1:  # Negative pair only if more than one sample
                j = random.choice([k for k in range(num_samples) if k != i])
                left = mask_spectra_side(all_x[i], "right")
                right = mask_spectra_side(all_x[j], "left")
                pairs.append(((left, right), [0.0, 1.0]))
            # If only one sample, cannot make a negative pair this way
        return pairs

    def pre_train_next_spectra(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> nn.Module:
        """Pre-trains the model using Next Spectra Prediction (NSP).

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.

        Returns:
            nn.Module: The trained model after pre-training.
        """
        self.logger.info("Starting Next Spectra Prediction (NSP) pre-training...")
        # Model output for NSP is typically (batch, 2) for binary classification
        # Ensure self.model's output layer is adapted for this, e.g. nn.Linear(..., 2)
        # This adaptation logic might need to be more explicit here or before calling.

        train_pairs = self._generate_contrastive_pairs_nsp(train_loader)
        val_pairs = self._generate_contrastive_pairs_nsp(val_loader)

        # Convert pairs to DataLoaders for batching if not already batched by _generate_contrastive_pairs_nsp
        # For simplicity, assuming _run_epoch_nsp iterates through the list of pairs.
        # A more robust NSP would use a custom Dataset/DataLoader that yields these pairs.
        if not train_pairs:
            self.logger.warning("NSP: No training pairs generated. Skipping training.")
            return self.model

        criterion = nn.BCEWithLogitsLoss()

        def _run_single_nsp_epoch(pair_list, is_training):
            if is_training:
                self.model.train()
            else:
                self.model.eval()

            epoch_loss = 0.0
            if not pair_list:
                return 0.0

            # Simple iteration for now, no batching of pairs here.
            # For larger pair_lists, batching would be needed.
            for (left, right), label_list in pair_list:
                left_dev, right_dev = left.to(self.config.device), right.to(
                    self.config.device
                )
                label = torch.tensor(
                    label_list, dtype=torch.float, device=self.config.device
                )

                if is_training:
                    self.optimizer.zero_grad()
                # Model expects (batch, n_features), so unsqueeze
                output = self.model(
                    left_dev.unsqueeze(0), right_dev.unsqueeze(0)
                )  # Batch of 1 pair
                loss = criterion(output, label.unsqueeze(0))
                if is_training:
                    loss.backward()
                    self.optimizer.step()
                epoch_loss += loss.item()
            return epoch_loss / len(pair_list)

        for epoch in range(self.config.num_epochs):
            train_loss = _run_single_nsp_epoch(train_pairs, is_training=True)
            val_loss = 0.0
            if val_pairs:
                with torch.no_grad():
                    val_loss = _run_single_nsp_epoch(val_pairs, is_training=False)
            self.logger.info(
                f"NSP Epoch [{epoch+1}/{self.config.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

        model_save_path = f"{self.config.file_path}_nsp.pth"
        torch.save(self.model.state_dict(), model_save_path)
        self.logger.info(f"NSP pre-training finished. Model saved to {model_save_path}")
        return self.model

    def _detect_peaks(
        self, spectra: T_Tensor, peak_threshold: float, window_size: int
    ) -> T_Tensor:
        """Detects peaks in the spectra using a sliding window approach.

        Args:
            spectra (T_Tensor): The input spectra tensor of shape (batch_size, n_features).
            peak_threshold (float): Threshold for peak detection relative to the maximum intensity.
            window_size (int): Size of the sliding window to consider for peak detection.

        Returns:
            T_Tensor: A boolean tensor of the same shape as spectra, where True indicates a peak.
        """
        batch_size, n_features = spectra.shape
        peak_labels = torch.zeros_like(spectra, dtype=torch.bool, device=spectra.device)
        for i in range(batch_size):
            spectrum_i = spectra[i]
            padded = F.pad(
                spectrum_i.unsqueeze(0).unsqueeze(0),
                (window_size // 2, window_size // 2),
                mode="replicate",
            ).squeeze()
            max_intensity_spectrum = torch.max(spectrum_i)
            if max_intensity_spectrum == 0:
                continue
            for j in range(n_features):
                window = padded[j : j + window_size]
                center_val = spectrum_i[j]
                if (center_val == torch.max(window)) and (
                    center_val > peak_threshold * max_intensity_spectrum
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
        """Pre-trains the model for Peak Prediction.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (Optional[DataLoader]): DataLoader for validation data.
            peak_threshold (float): Threshold for peak detection relative to the maximum intensity.
            window_size (int): Size of the sliding window to consider for peak detection.

        Returns:
            nn.Module: The trained model after pre-training.
        """
        self.logger.info("Starting Peak Prediction pre-training...")
        # Model output for this task is (batch, n_features) with logits for binary classification per point.

        def train_step(batch_data):
            """Performs a single training step for peak prediction.

            Args:
                batch_data (Tuple[T_Tensor, Any]): A batch of data containing spectra and possibly other data.

            Returns:
                float: The loss for the training step.
            """
            spectra, _ = batch_data
            spectra = spectra.to(self.config.device)
            peak_targets = self._detect_peaks(
                spectra, peak_threshold, window_size
            ).float()  # BCEWithLogitsLoss expects float targets

            self.optimizer.zero_grad()
            predictions = self.model(spectra, spectra)  # Or self.model(spectra)
            loss = F.binary_cross_entropy_with_logits(predictions, peak_targets)
            loss.backward()
            self.optimizer.step()
            return loss.item()

        def val_step(batch_data):
            """Performs a single validation step for peak prediction.

            Args:
                batch_data (Tuple[T_Tensor, Any]): A batch of data containing spectra and possibly other data.

            Returns:
                float: The loss for the validation step.
            """
            spectra, _ = batch_data
            spectra = spectra.to(self.config.device)
            peak_targets = self._detect_peaks(
                spectra, peak_threshold, window_size
            ).float()
            predictions = self.model(spectra, spectra)  # Or self.model(spectra)
            loss = F.binary_cross_entropy_with_logits(predictions, peak_targets)
            return loss.item()

        return self._perform_epoch_loop(
            "PeakPred",
            train_loader,
            train_step,
            val_loader,
            val_step,
            checkpoint_suffix="_peak_pred",
        )

    def _add_gaussian_noise(self, spectra: T_Tensor, std_dev: float = 0.1) -> T_Tensor:
        """Adds Gaussian noise to the spectra.

        Args:
            spectra (T_Tensor): The input spectra tensor of shape (batch_size, n_features).
            std_dev (float): Standard deviation of the Gaussian noise to be added.

        Returns:
            T_Tensor: The noisy spectra tensor with the same shape as input.
        """
        return torch.clamp(spectra + torch.randn_like(spectra) * std_dev, 0, 1)

    def _random_mask_points(
        self, spectra: T_Tensor, mask_prob: float = 0.05
    ) -> T_Tensor:
        """Randomly masks points in the spectra with a given probability.

        Args:
            spectra (T_Tensor): The input spectra tensor of shape (batch_size, n_features).
            mask_prob (float): Probability of masking each point in the spectra.

        Returns:
            T_Tensor: The spectra tensor with some points masked (set to 0).
        """
        noisy_spectra = spectra.clone()
        noisy_spectra[torch.rand_like(spectra) < mask_prob] = 0
        return noisy_spectra

    def pre_train_denoising_autoencoder(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        noise_std_dev: float = 0.1,
        mask_point_prob: float = 0.05,
    ) -> nn.Module:
        """Pre-trains the model using Spectrum Denoising Autoencoding (SDA).

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (Optional[DataLoader]): DataLoader for validation data.
            noise_std_dev (float): Standard deviation of Gaussian noise to be added.
            mask_point_prob (float): Probability of masking points in the spectra.

        Returns:
            nn.Module: The trained model after pre-training.
        """
        self.logger.info(
            "Starting Spectrum Denoising Autoencoding (SDA) pre-training..."
        )
        criterion = nn.MSELoss()
        # Ensure model's output layer matches n_features for reconstruction.

        def train_step(batch_data):
            """Performs a single training step for denoising autoencoding.

            Args:
                batch_data (Tuple[T_Tensor, Any]): A batch of data containing spectra and possibly other data.

            Returns:
                float: The loss for the training step.
            """
            spectra, _ = batch_data
            clean_spectra = spectra.to(self.config.device)
            noisy_spectra = self._add_gaussian_noise(
                clean_spectra.clone(), std_dev=noise_std_dev
            )
            noisy_spectra = self._random_mask_points(
                noisy_spectra, mask_prob=mask_point_prob
            )

            self.optimizer.zero_grad()
            predictions = self.model(
                noisy_spectra, noisy_spectra
            )  # Or self.model(noisy_spectra)
            loss = criterion(predictions, clean_spectra)
            loss.backward()
            self.optimizer.step()
            return loss.item()

        def val_step(batch_data):
            """Performs a single validation step for denoising autoencoding.

            Args:
                batch_data (Tuple[T_Tensor, Any]): A batch of data containing spectra and possibly other data.

            Returns:
                float: The loss for the validation step.
            """
            spectra, _ = batch_data
            clean_spectra = spectra.to(self.config.device)
            noisy_spectra = self._add_gaussian_noise(
                clean_spectra.clone(), std_dev=noise_std_dev
            )
            noisy_spectra = self._random_mask_points(
                noisy_spectra, mask_prob=mask_point_prob
            )
            predictions = self.model(
                noisy_spectra, noisy_spectra
            )  # Or self.model(noisy_spectra)
            return criterion(predictions, clean_spectra).item()

        return self._perform_epoch_loop(
            "SDA",
            train_loader,
            train_step,
            val_loader,
            val_step,
            checkpoint_suffix="_sda",
        )

    def pre_train_peak_parameter_regression(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        peak_detection_threshold: float = 0.1,
        peak_window_size: int = 5,
        mask_value: float = 0.0,
    ) -> nn.Module:
        """Pre-trains the model using Peak Parameter Regression (PPR).

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (Optional[DataLoader]): DataLoader for validation data.
            peak_detection_threshold (float): Threshold for peak detection relative to the maximum intensity.
            peak_window_size (int): Size of the sliding window to consider for peak detection.
            mask_value (float): Value to replace detected peaks in the spectra during training.

        Returns:
            nn.Module: The trained model after pre-training.
        """
        self.logger.info("Starting Peak Parameter Regression (PPR) pre-training...")
        criterion = nn.MSELoss()
        # Ensure model's output matches n_features.

        def train_step(batch_data):
            spectra, _ = batch_data
            spectra = spectra.to(self.config.device)
            peak_indices = self._detect_peaks(
                spectra, peak_detection_threshold, peak_window_size
            )
            if not peak_indices.any():
                return 0.0

            actual_peak_intensities = spectra[peak_indices]
            spectra_masked_peaks = spectra.clone()
            spectra_masked_peaks[peak_indices] = mask_value

            self.optimizer.zero_grad()
            predictions_full = self.model(
                spectra_masked_peaks, spectra_masked_peaks
            )  # Or self.model(spectra_masked_peaks)
            predicted_peak_intensities = predictions_full[peak_indices]

            if actual_peak_intensities.numel() == 0:
                return 0.0
            loss = criterion(predicted_peak_intensities, actual_peak_intensities)
            loss.backward()
            self.optimizer.step()
            # Return loss weighted by number of peaks to correctly average later
            return (
                loss.item() * actual_peak_intensities.numel(),
                actual_peak_intensities.numel(),
            )

        # Custom loop needed if train_step returns (weighted_loss, count)
        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_train_loss, total_train_peaks = 0.0, 0
            for batch in tqdm(
                train_loader, desc=f"PPR Epoch {epoch+1} [Train]", leave=False
            ):
                weighted_loss, num_peaks = train_step(batch)
                total_train_loss += weighted_loss
                total_train_peaks += num_peaks
            avg_train_loss = (
                total_train_loss / total_train_peaks if total_train_peaks > 0 else 0.0
            )
            log_msg = f"PPR Epoch [{epoch+1}/{self.config.num_epochs}], Train Loss: {avg_train_loss:.4f}"

            if val_loader:
                self.model.eval()
                total_val_loss, total_val_peaks = 0.0, 0
                with torch.no_grad():
                    for batch_val in val_loader:
                        # Simplified val_step for PPR, assuming it mirrors train_step's return needs
                        spectra_v, _ = batch_val
                        spectra_v = spectra_v.to(self.config.device)
                        peak_indices_v = self._detect_peaks(
                            spectra_v, peak_detection_threshold, peak_window_size
                        )
                        if not peak_indices_v.any():
                            continue
                        actual_intensities_v = spectra_v[peak_indices_v]
                        masked_spectra_v = spectra_v.clone()
                        masked_spectra_v[peak_indices_v] = mask_value
                        preds_full_v = self.model(masked_spectra_v, masked_spectra_v)
                        pred_intensities_v = preds_full_v[peak_indices_v]
                        if actual_intensities_v.numel() == 0:
                            continue
                        val_loss_item = criterion(
                            pred_intensities_v, actual_intensities_v
                        ).item()
                        total_val_loss += val_loss_item * actual_intensities_v.numel()
                        total_val_peaks += actual_intensities_v.numel()
                avg_val_loss = (
                    total_val_loss / total_val_peaks if total_val_peaks > 0 else 0.0
                )
                log_msg += f", Val Loss: {avg_val_loss:.4f}"
            self.logger.info(log_msg)

        model_save_path = f"{self.config.file_path}_ppr.pth"
        torch.save(self.model.state_dict(), model_save_path)
        self.logger.info(f"PPR pre-training finished. Model saved to {model_save_path}")
        return self.model

    def _segment_and_shuffle_spectra(
        self, spectra: T_Tensor, num_segments: int
    ) -> Tuple[T_Tensor, T_Tensor]:
        """Segments and shuffles the spectra for segment reordering task.

        Args:
            spectra (T_Tensor): The input spectra tensor of shape (batch_size, n_features).
            num_segments (int): Number of segments to divide each spectrum into.

        Returns:
            Tuple[T_Tensor, T_Tensor]: A tuple containing the shuffled spectra and the target
        """
        batch_size, n_features = spectra.shape
        if n_features % num_segments != 0:
            self.logger.warning(
                f"Features {n_features} not divisible by {num_segments}. Truncating/Padding needed or error."
            )
            # For this example, let's assume it's divisible or handled by data prep
            segment_len = n_features // num_segments
            spectra = spectra[
                :, : num_segments * segment_len
            ]  # Ensure divisibility by truncating
            n_features = spectra.shape[1]
        else:
            segment_len = n_features // num_segments

        segmented = spectra.reshape(batch_size, num_segments, segment_len)
        shuffled_list, target_indices_list = [], []
        for i in range(batch_size):
            permutation = torch.randperm(num_segments, device=spectra.device)
            shuffled_list.append(segmented[i][permutation].flatten())
            target_indices_list.append(
                permutation
            )  # Original index of segment now at this position
        return torch.stack(shuffled_list), torch.stack(target_indices_list)

    def pre_train_spectrum_segment_reordering(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_segments: int = 4,
    ) -> nn.Module:
        """Pre-trains the model using Spectrum Segment Reordering (SSR).

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (Optional[DataLoader]): DataLoader for validation data.
            num_segments (int): Number of segments to divide each spectrum into.

        Returns:
            nn.Module: The trained model after pre-training.
        """
        self.logger.info("Starting Spectrum Segment Reordering (SSR) pre-training...")
        criterion = nn.CrossEntropyLoss()
        # Model's final layer must output (num_segments * num_segments) for this task.
        # Store original fc to restore later, if applicable and it's a simple nn.Linear
        original_fc_config = None
        if hasattr(self.model, "fc") and isinstance(self.model.fc, nn.Linear):
            original_fc_config = (self.model.fc.in_features, self.model.fc.out_features)
            if self.model.fc.out_features != num_segments * num_segments:
                self.logger.info(
                    f"SSR: Adapting model.fc for {num_segments*num_segments} outputs."
                )
                self.model.fc = nn.Linear(
                    original_fc_config[0], num_segments * num_segments
                ).to(self.config.device)
                self.optimizer = AdamW(
                    self.model.parameters(), lr=self.config.learning_rate
                )
        else:
            self.logger.warning(
                "SSR: Model does not have 'fc: nn.Linear'. Ensure output is configured correctly."
            )

        def ssr_step(batch_data, is_training):
            """Performs a single training/validation step for segment reordering.

            Args:
                batch_data (Tuple[T_Tensor, Any]): A batch of data containing spectra and possibly other data.
                is_training (bool): Whether this step is for training or validation.

            Returns:
                Tuple[float, int, int]: Loss for the step, number of correct predictions,
                                        and total elements processed.
            """
            spectra, _ = batch_data
            spectra = spectra.to(self.config.device)
            batch_size_eff = spectra.shape[0]
            shuffled_spectra, target_indices = self._segment_and_shuffle_spectra(
                spectra, num_segments
            )

            if is_training:
                self.optimizer.zero_grad()
            predictions = self.model(
                shuffled_spectra, shuffled_spectra
            )  # Or self.model(shuffled_spectra)
            pred_reshaped = predictions.view(batch_size_eff, num_segments, num_segments)

            loss = criterion(
                pred_reshaped.reshape(-1, num_segments), target_indices.reshape(-1)
            )
            if is_training:
                loss.backward()
                self.optimizer.step()

            # Calculate accuracy for logging (optional, can make step fn complex)
            _, pred_labels = torch.max(pred_reshaped, 2)
            correct = (pred_labels == target_indices).sum().item()
            total_elements = target_indices.numel()
            return loss.item(), correct, total_elements

        # Custom loop for SSR to handle accuracy reporting
        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_train_loss, correct_train, total_train = 0.0, 0, 0
            for batch in tqdm(
                train_loader, desc=f"SSR Epoch {epoch+1} [Train]", leave=False
            ):
                loss_item, correct_item, total_item = ssr_step(batch, is_training=True)
                total_train_loss += loss_item * (
                    total_item / num_segments
                )  # Avg loss per sample
                correct_train += correct_item
                total_train += total_item
            avg_train_loss = (
                total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
            )
            train_acc = correct_train / total_train * 100 if total_train > 0 else 0
            log_msg = f"SSR Epoch [{epoch+1}/{self.config.num_epochs}], Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%"

            if val_loader:
                self.model.eval()
                total_val_loss, correct_val, total_val = 0.0, 0, 0
                with torch.no_grad():
                    for batch_val in val_loader:
                        loss_item_v, correct_item_v, total_item_v = ssr_step(
                            batch_val, is_training=False
                        )
                        total_val_loss += loss_item_v * (total_item_v / num_segments)
                        correct_val += correct_item_v
                        total_val += total_item_v
                avg_val_loss = (
                    total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
                )
                val_acc = correct_val / total_val * 100 if total_val > 0 else 0
                log_msg += f", Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%"
            self.logger.info(log_msg)

        if original_fc_config:  # Restore original fc
            self.logger.info(
                f"SSR: Restoring model.fc to output {original_fc_config[1]} features."
            )
            self.model.fc = nn.Linear(original_fc_config[0], original_fc_config[1]).to(
                self.config.device
            )
            self.optimizer = AdamW(
                self.model.parameters(), lr=self.config.learning_rate
            )

        model_save_path = f"{self.config.file_path}_ssr.pth"
        torch.save(self.model.state_dict(), model_save_path)
        self.logger.info(f"SSR pre-training finished. Model saved to {model_save_path}")
        return self.model

    def _intensity_scaling(
        self, spectra: T_Tensor, scale_min: float = 0.7, scale_max: float = 1.3
    ) -> T_Tensor:
        """Scales the intensity of the spectra randomly within a range.

        Args:
            spectra (T_Tensor): The input spectra tensor of shape (batch_size, n_features).
            scale_min (float): Minimum scaling factor.
            scale_max (float): Maximum scaling factor.

        Returns:
            T_Tensor: The scaled spectra tensor with the same shape as input.
        """
        scales = (
            torch.rand(spectra.shape[0], 1, device=spectra.device)
            * (scale_max - scale_min)
            + scale_min
        )
        return torch.clamp(spectra * scales, 0, 1)

    def _nt_xent_loss(
        self, z_i: T_Tensor, z_j: T_Tensor, temperature: float = 0.1
    ) -> T_Tensor:
        """Computes the normalized temperature-scaled cross-entropy loss for contrastive learning.

        Args:
            z_i (T_Tensor): Embeddings from the first view of shape (batch_size, embedding_dim).
            z_j (T_Tensor): Embeddings from the second view of shape (batch_size, embedding_dim).
            temperature (float): Temperature parameter for scaling the similarity.

        Returns:
            T_Tensor: The computed loss value.
        """
        batch_size = z_i.shape[0]
        z = F.normalize(torch.cat([z_i, z_j], dim=0), p=2, dim=1)
        sim_matrix = torch.matmul(z, z.T) / temperature
        labels = (torch.arange(2 * batch_size, device=z_i.device) + batch_size) % (
            2 * batch_size
        )
        sim_matrix_masked = sim_matrix.masked_fill(
            torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device),
            float("-inf"),
        )
        return F.cross_entropy(sim_matrix_masked, labels)

    def pre_train_contrastive_invariance(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        temperature: float = 0.1,
        embedding_dim: int = 128,
    ) -> nn.Module:
        """Pre-trains the model using Contrastive Transformation Invariance Learning (CTIL).

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (Optional[DataLoader]): DataLoader for validation data.
            temperature (float): Temperature parameter for NT-Xent loss.
            embedding_dim (int): Desired output dimension of the model's embeddings.

        Returns:
            nn.Module: The trained model after pre-training.
        """
        self.logger.info(
            "Starting Contrastive Transformation Invariance Learning (CTIL) pre-training..."
        )
        # This task requires model to output embeddings of `embedding_dim`.
        # Adaptation logic for `self.model.fc` or using `model.get_embedding()` + `model.projection_head()`
        original_fc_config, using_proj_head = None, False
        if hasattr(self.model, "get_embedding") and hasattr(
            self.model, "projection_head"
        ):
            self.logger.info(
                "CTIL: Using model.get_embedding() and model.projection_head()."
            )
            if (
                isinstance(self.model.projection_head, nn.Linear)
                and self.model.projection_head.out_features != embedding_dim
            ):
                self.logger.info(
                    f"CTIL: Adapting projection_head for {embedding_dim} outputs."
                )
                proj_in_feat = self.model.projection_head.in_features
                self.model.projection_head = nn.Linear(proj_in_feat, embedding_dim).to(
                    self.config.device
                )
                self.optimizer = AdamW(
                    self.model.parameters(), lr=self.config.learning_rate
                )
            using_proj_head = True
        elif hasattr(self.model, "fc") and isinstance(self.model.fc, nn.Linear):
            original_fc_config = (self.model.fc.in_features, self.model.fc.out_features)
            if self.model.fc.out_features != embedding_dim:
                self.logger.info(
                    f"CTIL: Adapting model.fc for {embedding_dim} outputs."
                )
                self.model.fc = nn.Linear(original_fc_config[0], embedding_dim).to(
                    self.config.device
                )
                self.optimizer = AdamW(
                    self.model.parameters(), lr=self.config.learning_rate
                )
        else:
            self.logger.error(
                "CTIL: Model cannot be adapted for specified embedding_dim. Ensure 'get_embedding'/'projection_head' or 'fc' layer."
            )

        def ctil_step(batch_data):
            """Performs a single training step for contrastive invariance learning.

            Args:
                batch_data (Tuple[T_Tensor, Any]): A batch of data containing spectra and possibly other data.

            Returns:
                float: The loss for the training step.
            """
            spectra_anchor, _ = batch_data
            spectra_anchor = spectra_anchor.to(self.config.device)
            view1 = self._add_gaussian_noise(
                self._intensity_scaling(spectra_anchor.clone()), std_dev=0.05
            )
            view2 = self._add_gaussian_noise(
                self._intensity_scaling(spectra_anchor.clone()), std_dev=0.05
            )

            self.optimizer.zero_grad()
            if (
                using_proj_head
            ):  # Assumes get_embedding returns features for projection_head
                emb1 = self.model.projection_head(self.model.get_embedding(view1))
                emb2 = self.model.projection_head(self.model.get_embedding(view2))
            else:  # Assumes model's main output (after fc adaptation) is the embedding
                emb1 = self.model(view1, view1)  # Or self.model(view1)
                emb2 = self.model(view2, view2)  # Or self.model(view2)

            loss = self._nt_xent_loss(emb1, emb2, temperature)
            loss.backward()
            self.optimizer.step()
            return loss.item()

        self._perform_epoch_loop(
            "CTIL",
            train_loader,
            ctil_step,
            val_loader,
            ctil_step,
            checkpoint_suffix="_ctil",
        )
        """ After training, we may want to restore the original model.fc if it was adapted.
        
        Args: 
            using_proj_head (bool): Whether the model used a projection head.
            original_fc_config (Optional[Tuple[int, int]]): Original configuration of the fc layer if adapted.

        Returns:
            nn.Module: The model with the original fc restored if applicable.
        """

        if (
            not using_proj_head and original_fc_config
        ):  # Restore original fc if it was adapted
            self.logger.info(
                f"CTIL: Restoring model.fc to output {original_fc_config[1]} features."
            )
            self.model.fc = nn.Linear(original_fc_config[0], original_fc_config[1]).to(
                self.config.device
            )
            self.optimizer = AdamW(
                self.model.parameters(), lr=self.config.learning_rate
            )
        return self.model
