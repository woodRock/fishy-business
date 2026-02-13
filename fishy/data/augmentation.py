# -*- coding: utf-8 -*-
"""
Data augmentation module for the deep learning pipeline.
"""

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from fishy.data.datasets import CustomDataset
from fishy._core.utils import get_device


@dataclass
class AugmentationConfig:
    """
    Configuration for data augmentation.

    Examples:
        >>> config = AugmentationConfig(enabled=True, num_augmentations=10)
        >>> config.enabled
        True
        >>> config.num_augmentations
        10
    """

    enabled: bool = False
    num_augmentations: int = 5
    noise_enabled: bool = True
    shift_enabled: bool = False
    scale_enabled: bool = False
    crop_enabled: bool = False
    flip_enabled: bool = False
    permutation_enabled: bool = False
    noise_level: float = 0.1
    shift_range: float = 0.1
    scale_range: float = 0.1
    crop_size: float = 0.8


class DataAugmenter:
    """Applies various data augmentation techniques to a DataLoader."""

    def __init__(self, config: AugmentationConfig) -> None:
        self.config = config

    def _random_crop(self, X_batch: torch.Tensor) -> torch.Tensor:
        n_samples, n_features = X_batch.shape
        cropped_batch = torch.zeros_like(X_batch, device=X_batch.device)
        for i in range(n_samples):
            spectrum = X_batch[i]
            crop_len = int(n_features * self.config.crop_size)
            if crop_len == 0:
                cropped_batch[i] = spectrum
                continue
            start = torch.randint(0, max(1, n_features - crop_len + 1), (1,)).item()
            end = start + crop_len
            cropped_spectrum = spectrum[start:end]
            padding_left = start
            padding_right = n_features - end
            cropped_batch[i] = F.pad(
                cropped_spectrum, (padding_left, padding_right), "constant", 0
            )
        return cropped_batch

    def _random_flip(self, X_batch: torch.Tensor) -> torch.Tensor:
        flipped_batch = X_batch.clone()
        if torch.rand(1).item() < 0.5:
            flipped_batch = torch.flip(flipped_batch, dims=[1])
        return flipped_batch

    def _random_permutation(self, X_batch: torch.Tensor) -> torch.Tensor:
        permuted_batch = X_batch.clone()
        n_samples, n_features = X_batch.shape
        for i in range(n_samples):
            idx = torch.randperm(n_features, device=X_batch.device)
            permuted_batch[i] = permuted_batch[i, idx]
        return permuted_batch

    def _apply_augmentations_to_batch(self, X_batch: torch.Tensor) -> torch.Tensor:
        X_augmented_batch = X_batch.clone()
        n_samples, n_features = X_augmented_batch.shape

        if self.config.noise_enabled:
            X_augmented_batch += torch.normal(
                0,
                self.config.noise_level,
                X_augmented_batch.shape,
                device=X_batch.device,
            )

        if self.config.shift_enabled:
            for k in range(n_samples):
                shift = int(
                    n_features
                    * torch.empty(1)
                    .uniform_(-self.config.shift_range, self.config.shift_range)
                    .item()
                )
                X_augmented_batch[k] = torch.roll(
                    X_augmented_batch[k], shifts=shift, dims=0
                )

        if self.config.scale_enabled:
            for k in range(n_samples):
                scale = (
                    torch.empty(1)
                    .uniform_(1 - self.config.scale_range, 1 + self.config.scale_range)
                    .item()
                )
                X_augmented_batch[k] *= scale

        if self.config.crop_enabled:
            X_augmented_batch = self._random_crop(X_augmented_batch)
        if self.config.flip_enabled:
            X_augmented_batch = self._random_flip(X_augmented_batch)
        if self.config.permutation_enabled:
            X_augmented_batch = self._random_permutation(X_augmented_batch)

        return X_augmented_batch

    def augment(self, dataloader: DataLoader) -> DataLoader:
        if not self.config.enabled or self.config.num_augmentations == 0:
            return dataloader

        device = get_device()
        all_samples, all_labels = [], []
        for samples, labels in dataloader:
            all_samples.append(samples)
            all_labels.append(labels)

        if not all_samples:
            return dataloader

        all_samples_tensor = torch.cat(all_samples, dim=0).to(device)
        all_labels_tensor = torch.cat(all_labels, dim=0)

        augmented_samples_list = [all_samples_tensor]
        for _ in range(self.config.num_augmentations):
            augmented_samples_list.append(
                self._apply_augmentations_to_batch(all_samples_tensor)
            )

        combined_samples = torch.cat(augmented_samples_list, dim=0)

        # Ensure labels are 2D for consistent repeating
        if all_labels_tensor.ndim == 1:
            all_labels_tensor = all_labels_tensor.view(-1, 1)

        combined_labels = all_labels_tensor.repeat(
            self.config.num_augmentations + 1, 1
        ).to(device)

        permutation = torch.randperm(combined_samples.size(0), device=device)
        combined_samples = combined_samples[permutation].cpu().numpy()
        combined_labels = combined_labels[permutation].cpu().numpy()

        new_dataset = CustomDataset(combined_samples, combined_labels)
        return DataLoader(
            new_dataset,
            batch_size=dataloader.batch_size,
            shuffle=True,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
        )
