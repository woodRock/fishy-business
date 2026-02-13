# -*- coding: utf-8 -*-
import pytest
import os
import pandas as pd
import torch
from fishy._core.config import TrainingConfig, ExperimentConfig
from fishy.experiments.unified_trainer import UnifiedTrainer
from fishy.experiments.classic_training import run_sklearn_experiment
from fishy.experiments.deep_training import run_training_pipeline

from unittest.mock import patch


def test_unified_trainer_single(tmp_path):
    # Use a small config to test the dispatch logic
    data_file = "data/REIMS.xlsx"
    if not os.path.exists(data_file):
        pytest.skip("Data file not found")

    cfg = TrainingConfig(
        model="transformer",
        dataset="species",
        file_path=data_file,
        epochs=1,
        k_folds=2,
        wandb_log=False,
    )
    # Patch all possible device getters
    with (
        patch(
            "fishy.experiments.unified_trainer.get_device",
            return_value=torch.device("cpu"),
        ),
        patch(
            "fishy.experiments.deep_training.get_device",
            return_value=torch.device("cpu"),
        ),
        patch("fishy.engine.trainer.get_device", return_value=torch.device("cpu")),
    ):
        trainer = UnifiedTrainer(cfg)
        results = trainer.run()
    assert isinstance(results, dict)
    assert "val_balanced_accuracy" in results


def test_sklearn_experiment_flow(tmp_path):
    data_file = "data/REIMS.xlsx"
    if not os.path.exists(data_file):
        pytest.skip("Data file not found")

    cfg = TrainingConfig(model="rf", dataset="species", file_path=data_file, k_folds=2)
    stats = run_sklearn_experiment(cfg, "rf", "species", file_path=data_file)
    assert "val_balanced_accuracy" in stats
    assert "folds" in stats


def test_deep_training_pipeline_flow(tmp_path):
    data_file = "data/REIMS.xlsx"
    if not os.path.exists(data_file):
        pytest.skip("Data file not found")

    cfg = TrainingConfig(
        model="dense", dataset="species", file_path=data_file, epochs=1, k_folds=2
    )
    with (
        patch(
            "fishy.experiments.deep_training.get_device",
            return_value=torch.device("cpu"),
        ),
        patch("fishy.engine.trainer.get_device", return_value=torch.device("cpu")),
    ):
        stats = run_training_pipeline(cfg)
    assert "val_balanced_accuracy" in stats
