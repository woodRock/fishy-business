# -*- coding: utf-8 -*-
import pytest
import torch
import torch.nn as nn
import numpy as np
import os
from fishy.data.augmentation import DataAugmenter, AugmentationConfig
from fishy.experiments.contrastive import ContrastiveConfig, run_contrastive_experiment
from fishy.experiments.pre_training import PreTrainingOrchestrator
from fishy._core.config import TrainingConfig
from fishy._core.utils import RunContext
from fishy import get_data_path

from fishy.models.contrastive.byol import BYOLModel
from fishy.models.contrastive.barlow_twins import BarlowTwinsModel
from fishy.models.contrastive.moco import MoCoModel
from fishy.models.contrastive.simsiam import SimSiamModel
from fishy.models.contrastive.simclr import SimCLRModel


def test_spectral_augmentation():
    from torch.utils.data import DataLoader, TensorDataset

    x = torch.randn(16, 100)
    y = torch.randint(0, 2, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)

    cfg = AugmentationConfig(
        enabled=True, num_augmentations=1, noise_enabled=True, noise_level=0.5
    )
    augmentor = DataAugmenter(cfg)
    aug_loader = augmentor.augment(loader)

    aug_samples = []
    for batch_x, _ in aug_loader:
        aug_samples.append(batch_x)
    aug_x_all = torch.cat(aug_samples, dim=0)

    assert len(aug_x_all) == 16 * 2
    assert torch.sum(torch.abs(aug_x_all.mean(dim=0) - x.mean(dim=0))) > 1e-4


def test_contrastive_models_direct():
    input_dim = 100
    backbone = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU())

    # Test BYOL
    byol = BYOLModel(backbone, 64, 64)
    x1, x2 = torch.randn(2, input_dim), torch.randn(2, input_dim)
    out = byol(x1, x2)
    assert isinstance(out, tuple)

    # Test Barlow Twins
    bt = BarlowTwinsModel(backbone, 64, 64)
    out = bt(x1, x2)
    assert isinstance(out, tuple)

    # Test MoCo
    moco = MoCoModel(backbone, 64, 64)
    out = moco(x1, x2)
    assert isinstance(out, tuple)

    # Test SimSiam
    ss = SimSiamModel(backbone, 64, 64)
    out = ss(x1, x2)
    assert isinstance(out, tuple)

    # Test SimCLR
    sc = SimCLRModel(backbone, embedding_dim=64, projection_dim=64, dropout=0.1)
    out = sc(x1, x2)
    assert isinstance(out, tuple)


def test_pretraining_orchestrator(tmp_path):
    data_file = get_data_path()
    if not os.path.exists(data_file):
        pytest.skip("Data file not found")

    cfg = TrainingConfig(
        model="transformer",
        dataset="species",
        file_path=data_file,
        masked_spectra_modelling=True,
        epochs=1,
        batch_size=8,
    )
    ctx = RunContext("test", "test", "test")
    orchestrator = PreTrainingOrchestrator(cfg, input_dim=2080, device="cpu", ctx=ctx)
    from torch.utils.data import DataLoader, TensorDataset

    loader = DataLoader(
        TensorDataset(torch.randn(16, 2080), torch.randint(0, 2, (16,))), batch_size=8
    )
    model = orchestrator.run_all(loader)
    assert isinstance(model, torch.nn.Module)
