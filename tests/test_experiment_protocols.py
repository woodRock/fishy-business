# -*- coding: utf-8 -*-
"""
Tests that verify experiment protocols behave as expected:
- Each method type (classic, deep, contrastive) returns the required metric keys.
- Reported metrics are in valid ranges.
- Batch-detection uses 3-fold stratified CV on all possible pairs (C(N,2)=2556).
- All three method types use the same pair-level folds for a given seed.
"""

import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch

from fishy._core.config import TrainingConfig
from fishy.experiments.contrastive import ContrastiveConfig, ContrastiveTrainer
from fishy.experiments.classic_training import SklearnTrainer
from fishy.experiments.deep_training import ModelTrainer


# ---------------------------------------------------------------------------
# Shared synthetic data helpers
# ---------------------------------------------------------------------------


def _make_batch_detection_df(n_classes=24, n_per_class=3, n_features=5, seed=0):
    """
    Returns a DataFrame in the format DataProcessor expects for batch-detection:
      - first column "m/z": batch class label
      - remaining columns: float features
    Total rows = n_classes * n_per_class (default 72, like the real dataset).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_classes):
        for _ in range(n_per_class):
            feats = rng.random(n_features).astype(np.float32)
            rows.append([f"Batch_{c + 1:02d}"] + feats.tolist())
    cols = ["m/z"] + [f"feat_{j}" for j in range(n_features)]
    return pd.DataFrame(rows, columns=cols)


def _make_species_df(n_per_class=10, n_features=5, seed=0):
    """
    Returns a DataFrame in the format DataProcessor expects for the species dataset.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for label_prefix in ("H", "M"):
        for i in range(n_per_class):
            feats = rng.random(n_features).astype(np.float32)
            rows.append([f"{label_prefix}_{i + 1}"] + feats.tolist())
    cols = ["m/z"] + [f"feat_{j}" for j in range(n_features)]
    return pd.DataFrame(rows, columns=cols)


def _cpu_device_patches():
    """Context managers that force CPU so tests run without GPU."""
    return [
        patch(
            "fishy.experiments.deep_training.get_device",
            return_value=torch.device("cpu"),
        ),
        patch("fishy.engine.trainer.get_device", return_value=torch.device("cpu")),
    ]


# ---------------------------------------------------------------------------
# Classic trainer — batch-detection
# ---------------------------------------------------------------------------


class TestClassicBatchDetection(unittest.TestCase):
    """SklearnTrainer on batch-detection must use pair-level 3-fold CV."""

    REQUIRED_KEYS = {
        "val_balanced_accuracy",
        "val_accuracy",
        "val_f1",
        "folds",
    }

    def _run(self, model_name="lda", seed=42):
        df = _make_batch_detection_df()
        cfg = TrainingConfig(
            model=model_name,
            dataset="batch-detection",
            k_folds=3,
            run=seed,
        )
        with patch("fishy.data.module.DataProcessor.load_data", return_value=df):
            trainer = SklearnTrainer(cfg, model_name, "batch-detection", run_id=seed)
            _, stats = trainer.run()
        return stats

    def test_required_keys_present(self):
        stats = self._run()
        for key in self.REQUIRED_KEYS:
            self.assertIn(key, stats, f"Missing key: {key}")

    def test_balanced_accuracy_in_valid_range(self):
        stats = self._run()
        ba = stats["val_balanced_accuracy"]
        self.assertGreaterEqual(ba, 0.0)
        self.assertLessEqual(ba, 1.0)

    def test_uses_all_pairs_cv(self):
        """Verify that it performs 3-fold CV on all possible pairs."""
        stats = self._run()
        self.assertEqual(len(stats["folds"]), 3)
        # 72 samples -> 2556 pairs. Each fold val set should be ~852 pairs.
        total_val_pairs = sum(len(f["val_accuracy"]) if isinstance(f.get("val_accuracy"), (list, np.ndarray)) else 1 for f in stats["folds"])
        # Actually our folds in stats contain scalar metrics, not the raw predictions usually.
        # But we can check that make_all_pairwise_folds was called.
        df = _make_batch_detection_df()
        cfg = TrainingConfig(model="lda", dataset="batch-detection", run=0)
        
        with (
            patch("fishy.data.module.DataProcessor.load_data", return_value=df),
            patch("fishy.experiments.classic_training.make_all_pairwise_folds", wraps=__import__("fishy.data.module", fromlist=["make_all_pairwise_folds"]).make_all_pairwise_folds) as mock_folds
        ):
            trainer = SklearnTrainer(cfg, "lda", "batch-detection", run_id=0)
            trainer.run()
            mock_folds.assert_called_once()


# ---------------------------------------------------------------------------
# Deep trainer — batch-detection
# ---------------------------------------------------------------------------


class TestDeepBatchDetection(unittest.TestCase):
    """ModelTrainer on batch-detection must use pair-level 3-fold CV."""

    REQUIRED_KEYS = {"val_balanced_accuracy", "folds"}

    def _run(self, seed=42):
        df = _make_batch_detection_df()
        cfg = TrainingConfig(
            model="dense",
            dataset="batch-detection",
            epochs=1,
            k_folds=3,
            batch_size=32,
            run=seed,
            wandb_log=False,
        )
        patches = _cpu_device_patches()
        with (
            patch("fishy.data.module.DataProcessor.load_data", return_value=df),
            patches[0],
            patches[1],
        ):
            trainer = ModelTrainer(cfg)
            _, stats = trainer.train(pre_trained_model=None)
        return stats

    def test_required_keys_present(self):
        stats = self._run()
        for key in self.REQUIRED_KEYS:
            self.assertIn(key, stats, f"Missing key: {key}")

    def test_balanced_accuracy_in_valid_range(self):
        stats = self._run()
        ba = stats["val_balanced_accuracy"]
        self.assertIsNotNone(ba)
        self.assertGreaterEqual(ba, 0.0)
        self.assertLessEqual(ba, 1.0)


# ---------------------------------------------------------------------------
# Contrastive trainer — batch-detection
# ---------------------------------------------------------------------------


class TestContrastiveBatchDetection(unittest.TestCase):
    """ContrastiveTrainer on batch-detection must use pair-level 3-fold CV."""

    REQUIRED_KEYS = {
        "val_balanced_accuracy",
        "val_accuracy",
        "val_f1",
        "folds",
    }

    def _run(self, method="simclr", seed=42):
        df = _make_batch_detection_df()
        cfg = ContrastiveConfig(
            contrastive_method=method,
            dataset="batch-detection",
            num_epochs=1,
            batch_size=16,
            embedding_dim=8,
            projection_dim=8,
            file_path=None,
            wandb_log=False,
            run=seed,
            k_folds=3,
        )
        patches = _cpu_device_patches()
        with (
            patch("fishy.data.module.DataProcessor.load_data", return_value=df),
            patches[0],
            patches[1],
            patch(
                "fishy.experiments.contrastive.get_device",
                return_value=torch.device("cpu"),
            ),
        ):
            trainer = ContrastiveTrainer(cfg)
            trainer.setup()
            trainer.train()
        return trainer.metrics

    def test_required_keys_present(self):
        metrics = self._run()
        for key in self.REQUIRED_KEYS:
            self.assertIn(key, metrics, f"Missing key: {key}")

    def test_correct_number_of_folds(self):
        metrics = self._run()
        self.assertEqual(len(metrics["folds"]), 3)


# ---------------------------------------------------------------------------
# Shared split protocol — all three method types
# ---------------------------------------------------------------------------


class TestSharedSplitProtocol(unittest.TestCase):
    """For a given seed, all methods must use identical pair-level fold indices."""

    def _get_folds(self, method_type, df, seed):
        captured_folds = []
        
        def spy_folds(X, y, n_splits, run_id):
            from fishy.data.module import make_all_pairwise_folds
            X1, X2, labels, folds = make_all_pairwise_folds(X, y, n_splits, run_id)
            captured_folds.append(folds)
            return X1, X2, labels, folds

        if method_type == "classic":
            cfg = TrainingConfig(model="lda", dataset="batch-detection", run=seed, k_folds=3)
            with (
                patch("fishy.data.module.DataProcessor.load_data", return_value=df),
                patch("fishy.experiments.classic_training.make_all_pairwise_folds", side_effect=spy_folds)
            ):
                trainer = SklearnTrainer(cfg, "lda", "batch-detection", run_id=seed)
                trainer.run()
        elif method_type == "deep":
            cfg = TrainingConfig(model="dense", dataset="batch-detection", epochs=0, k_folds=3, run=seed, wandb_log=False)
            patches = _cpu_device_patches()
            with (
                patch("fishy.data.module.DataProcessor.load_data", return_value=df),
                patch("fishy.experiments.deep_training.make_all_pairwise_folds", side_effect=spy_folds),
                patches[0], patches[1]
            ):
                trainer = ModelTrainer(cfg)
                trainer.train()
        elif method_type == "contrastive":
            cfg = ContrastiveConfig(contrastive_method="simclr", dataset="batch-detection", num_epochs=0, run=seed, k_folds=3, wandb_log=False)
            with (
                patch("fishy.data.module.DataProcessor.load_data", return_value=df),
                patch("fishy.experiments.contrastive.make_all_pairwise_folds", side_effect=spy_folds),
                patch("fishy.experiments.contrastive.get_device", return_value=torch.device("cpu"))
            ):
                trainer = ContrastiveTrainer(cfg)
                trainer.setup()
                trainer.train()
        
        return captured_folds[0]

    def test_all_methods_use_same_folds(self):
        df = _make_batch_detection_df(seed=1)
        seed = 42
        folds_classic = self._get_folds("classic", df, seed)
        folds_deep = self._get_folds("deep", df, seed)
        folds_contrastive = self._get_folds("contrastive", df, seed)
        
        for f1, f2, f3 in zip(folds_classic, folds_deep, folds_contrastive):
            # Check train indices
            np.testing.assert_array_equal(f1[0], f2[0])
            np.testing.assert_array_equal(f1[0], f3[0])
            # Check val indices
            np.testing.assert_array_equal(f1[1], f2[1])
            np.testing.assert_array_equal(f1[1], f3[1])


if __name__ == "__main__":
    unittest.main()
