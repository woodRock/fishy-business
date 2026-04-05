# -*- coding: utf-8 -*-
"""
Tests that verify experiment protocols behave as expected:
- Each method type (classic, deep, contrastive) returns the required metric keys.
- Reported metrics are in valid ranges.
- The batch-detection path evaluates on a held-out test set (no leakage).
- val_balanced_accuracy mirrors test_balanced_accuracy for batch-detection
  (all methods report the same metric under the same key name).
- All three method types use the same train/test partition for a given seed.
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
      - first column "m/z": batch class label (all samples in a class share the
        same label, so LabelEncoder produces n_classes classes, not n_samples)
      - remaining columns: float features
    Total rows = n_classes * n_per_class (default 72, like the real dataset).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_classes):
        for _ in range(n_per_class):
            feats = rng.random(n_features).astype(np.float32)
            # All samples in class c share the same label string
            rows.append([f"Batch_{c + 1:02d}"] + feats.tolist())
    cols = ["m/z"] + [f"feat_{j}" for j in range(n_features)]
    return pd.DataFrame(rows, columns=cols)


def _make_species_df(n_per_class=10, n_features=5, seed=0):
    """
    Returns a DataFrame in the format DataProcessor expects for the species dataset:
      - first column "m/z": labels starting with "H_" (Hoki) or "M_" (Mackerel)
      - remaining columns: float features
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
        patch("fishy.experiments.deep_training.get_device", return_value=torch.device("cpu")),
        patch("fishy.engine.trainer.get_device", return_value=torch.device("cpu")),
    ]


# ---------------------------------------------------------------------------
# Classic trainer — batch-detection
# ---------------------------------------------------------------------------

class TestClassicBatchDetection(unittest.TestCase):
    """SklearnTrainer on batch-detection must use a held-out test split and
    report the required metric keys with values in [0, 1]."""

    REQUIRED_KEYS = {
        "test_balanced_accuracy",
        "val_balanced_accuracy",
        "test_accuracy",
        "val_accuracy",
        "test_f1",
        "val_f1",
    }

    def _run(self, model_name="lda", seed=42):
        df = _make_batch_detection_df()
        cfg = TrainingConfig(
            model=model_name,
            dataset="batch-detection",
            k_folds=2,
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
        ba = stats["test_balanced_accuracy"]
        self.assertGreaterEqual(ba, 0.0)
        self.assertLessEqual(ba, 1.0)

    def test_val_mirrors_test_balanced_accuracy(self):
        """For batch-detection, val_balanced_accuracy must equal test_balanced_accuracy
        — both sides of a comparison table must read from the same held-out split."""
        stats = self._run()
        self.assertEqual(
            stats["val_balanced_accuracy"],
            stats["test_balanced_accuracy"],
            "val and test balanced accuracy must be equal for batch-detection",
        )

    def test_evaluates_on_held_out_half(self):
        """The test set is ~50% of the data.  The pairwise evaluation is done on
        those held-out samples only — not the full dataset."""
        df = _make_batch_detection_df()
        cfg = TrainingConfig(model="lda", dataset="batch-detection", run=0)
        captures = {}
        real_split = __import__(
            "fishy.data.module", fromlist=["make_pairwise_test_split"]
        ).make_pairwise_test_split

        def spy_split(X, y, run_id, *extra, **kw):
            result = real_split(X, y, run_id, *extra, **kw)
            captures["n_train"] = len(result[0])
            captures["n_test"] = len(result[1])
            captures["n_total"] = len(X)
            return result

        with (
            patch("fishy.data.module.DataProcessor.load_data", return_value=df),
            patch("fishy.experiments.classic_training.make_pairwise_test_split", side_effect=spy_split),
        ):
            trainer = SklearnTrainer(cfg, "lda", "batch-detection", run_id=0)
            trainer.run()

        total = captures["n_total"]
        self.assertAlmostEqual(
            captures["n_test"] / total, 0.5, delta=0.05,
            msg="Held-out test set should be ~50% of the data",
        )
        self.assertEqual(captures["n_train"] + captures["n_test"], total)


# ---------------------------------------------------------------------------
# Classic trainer — non-batch-detection (species)
# ---------------------------------------------------------------------------

class TestClassicSpecies(unittest.TestCase):
    """SklearnTrainer on a normal classification dataset must return per-fold
    metrics and a summary val_balanced_accuracy."""

    def _run(self, model_name="lda", k_folds=3):
        df = _make_species_df()
        cfg = TrainingConfig(model=model_name, dataset="species", k_folds=k_folds, run=0)
        with patch("fishy.data.module.DataProcessor.load_data", return_value=df):
            trainer = SklearnTrainer(cfg, model_name, "species", run_id=0)
            _, stats = trainer.run()
        return stats, k_folds

    def test_required_keys_present(self):
        stats, _ = self._run()
        for key in ("val_balanced_accuracy", "folds"):
            self.assertIn(key, stats)

    def test_correct_number_of_folds(self):
        stats, k_folds = self._run(k_folds=3)
        self.assertEqual(len(stats["folds"]), k_folds)

    def test_balanced_accuracy_in_valid_range(self):
        stats, _ = self._run()
        ba = stats["val_balanced_accuracy"]
        self.assertGreaterEqual(ba, 0.0)
        self.assertLessEqual(ba, 1.0)

    def test_each_fold_has_train_and_val_metrics(self):
        stats, _ = self._run()
        for fold in stats["folds"]:
            self.assertIn("train_balanced_accuracy", fold)
            self.assertIn("val_balanced_accuracy", fold)


# ---------------------------------------------------------------------------
# Deep trainer — batch-detection (pairwise difference-vector binary classification)
# ---------------------------------------------------------------------------

class TestDeepBatchDetection(unittest.TestCase):
    """ModelTrainer on batch-detection must use binary same/different
    classification on difference vectors and report test_balanced_accuracy."""

    REQUIRED_KEYS = {"test_balanced_accuracy", "val_balanced_accuracy"}

    def _run(self, seed=42):
        df = _make_batch_detection_df()
        cfg = TrainingConfig(
            model="dense",
            dataset="batch-detection",
            epochs=1,
            k_folds=2,
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
        ba = stats["test_balanced_accuracy"]
        self.assertIsNotNone(ba)
        self.assertGreaterEqual(ba, 0.0)
        self.assertLessEqual(ba, 1.0)

    def test_val_mirrors_test_balanced_accuracy(self):
        stats = self._run()
        self.assertEqual(
            stats["val_balanced_accuracy"],
            stats["test_balanced_accuracy"],
            "val and test balanced accuracy must be equal for batch-detection",
        )

    def test_binary_same_different_task(self):
        """The pairwise path classifies pairs as same (1) or different (0).
        Predictions must be 0 or 1 — not the original 24 class indices."""
        df = _make_batch_detection_df()
        cfg = TrainingConfig(
            model="dense",
            dataset="batch-detection",
            epochs=1,
            k_folds=2,
            batch_size=32,
            run=0,
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

        preds_entry = stats.get("predictions")
        if preds_entry is not None:
            # DeepEngine stores predictions as {"labels": ..., "preds": array}
            pred_array = (
                preds_entry.get("preds")
                if isinstance(preds_entry, dict)
                else preds_entry
            )
            if pred_array is not None:
                pred_values = np.unique(pred_array)
                self.assertTrue(
                    set(pred_values.tolist()).issubset({0, 1}),
                    f"Expected binary predictions (0/1), got: {pred_values}",
                )


# ---------------------------------------------------------------------------
# Contrastive trainer — batch-detection
# ---------------------------------------------------------------------------

class TestContrastiveBatchDetection(unittest.TestCase):
    """ContrastiveTrainer on batch-detection must hold out a test split,
    evaluate pairwise similarity on it, and report test_balanced_accuracy."""

    REQUIRED_KEYS = {
        "test_balanced_accuracy",
        "val_balanced_accuracy",
        "test_accuracy",
        "val_accuracy",
        "test_f1",
        "val_f1",
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
        )
        patches = _cpu_device_patches()
        with (
            patch("fishy.data.module.DataProcessor.load_data", return_value=df),
            patches[0],
            patches[1],
            patch("fishy.experiments.contrastive.get_device", return_value=torch.device("cpu")),
        ):
            trainer = ContrastiveTrainer(cfg)
            trainer.setup()
            trainer.train()
        return trainer.metrics

    def test_required_keys_present(self):
        metrics = self._run()
        for key in self.REQUIRED_KEYS:
            self.assertIn(key, metrics, f"Missing key: {key}")

    def test_balanced_accuracy_in_valid_range(self):
        metrics = self._run()
        ba = metrics["test_balanced_accuracy"]
        self.assertGreaterEqual(ba, 0.0)
        self.assertLessEqual(ba, 1.0)

    def test_val_mirrors_test_balanced_accuracy(self):
        metrics = self._run()
        self.assertEqual(
            metrics["val_balanced_accuracy"],
            metrics["test_balanced_accuracy"],
        )

    def test_test_set_is_held_out_from_training(self):
        """The contrastive trainer must train on _train_X only and evaluate
        on _test_X only — not the full dataset."""
        df = _make_batch_detection_df()
        cfg = ContrastiveConfig(
            contrastive_method="simclr",
            dataset="batch-detection",
            num_epochs=1,
            batch_size=16,
            embedding_dim=8,
            projection_dim=8,
            wandb_log=False,
            run=0,
        )
        patches = _cpu_device_patches()
        with (
            patch("fishy.data.module.DataProcessor.load_data", return_value=df),
            patches[0],
            patches[1],
            patch("fishy.experiments.contrastive.get_device", return_value=torch.device("cpu")),
        ):
            trainer = ContrastiveTrainer(cfg)
            trainer.setup()

        n_total = len(df)  # 72 after no filtering
        n_train = len(trainer._train_X)
        n_test = len(trainer._test_X)

        self.assertEqual(n_train + n_test, n_total,
                         "Train + test sizes must sum to total samples")
        self.assertAlmostEqual(n_test / n_total, 0.5, delta=0.05,
                               msg="Test split should be ~50% of the data")


# ---------------------------------------------------------------------------
# Shared split protocol — all three method types
# ---------------------------------------------------------------------------

class TestSharedSplitProtocol(unittest.TestCase):
    """For a given seed, classic, deep, and contrastive must all hold out
    the exact same samples in their test sets."""

    def _get_classic_test_samples(self, df, seed):
        cfg = TrainingConfig(model="lda", dataset="batch-detection", run=seed)
        captured = {}
        real_split = __import__(
            "fishy.data.module", fromlist=["make_pairwise_test_split"]
        ).make_pairwise_test_split

        def spy(X, y, run_id, *extra, **kw):
            result = real_split(X, y, run_id, *extra, **kw)
            captured["X_te"] = result[1]
            return result

        with (
            patch("fishy.data.module.DataProcessor.load_data", return_value=df),
            patch("fishy.experiments.classic_training.make_pairwise_test_split", side_effect=spy),
        ):
            trainer = SklearnTrainer(cfg, "lda", "batch-detection", run_id=seed)
            trainer.run()
        return captured["X_te"]

    def _get_deep_test_samples(self, df, seed):
        cfg = TrainingConfig(
            model="dense", dataset="batch-detection",
            epochs=1, k_folds=2, batch_size=32, run=seed, wandb_log=False,
        )
        captured = {}
        real_split = __import__(
            "fishy.data.module", fromlist=["make_pairwise_test_split"]
        ).make_pairwise_test_split

        def spy(X, y, run_id, *extra, **kw):
            result = real_split(X, y, run_id, *extra, **kw)
            captured["X_te"] = result[1]
            return result

        patches = _cpu_device_patches()
        with (
            patch("fishy.data.module.DataProcessor.load_data", return_value=df),
            patch("fishy.experiments.deep_training.make_pairwise_test_split", side_effect=spy),
            patches[0],
            patches[1],
        ):
            trainer = ModelTrainer(cfg)
            trainer.train(pre_trained_model=None)
        return captured["X_te"]

    def _get_contrastive_test_samples(self, df, seed):
        cfg = ContrastiveConfig(
            contrastive_method="simclr",
            dataset="batch-detection",
            num_epochs=1,
            batch_size=16,
            embedding_dim=8,
            projection_dim=8,
            wandb_log=False,
            run=seed,
        )
        patches = _cpu_device_patches()
        with (
            patch("fishy.data.module.DataProcessor.load_data", return_value=df),
            patches[0],
            patches[1],
            patch("fishy.experiments.contrastive.get_device", return_value=torch.device("cpu")),
        ):
            trainer = ContrastiveTrainer(cfg)
            trainer.setup()
        return trainer._test_X

    def test_classic_and_deep_use_same_test_samples(self):
        df = _make_batch_detection_df(seed=1)
        seed = 7
        X_te_classic = self._get_classic_test_samples(df, seed)
        X_te_deep = self._get_deep_test_samples(df, seed)
        np.testing.assert_array_equal(
            X_te_classic, X_te_deep,
            err_msg="Classic and deep methods held out different test samples",
        )

    def test_classic_and_contrastive_use_same_test_samples(self):
        df = _make_batch_detection_df(seed=1)
        seed = 7
        X_te_classic = self._get_classic_test_samples(df, seed)
        X_te_contrastive = self._get_contrastive_test_samples(df, seed)
        np.testing.assert_array_equal(
            X_te_classic, X_te_contrastive,
            err_msg="Classic and contrastive methods held out different test samples",
        )

    def test_different_seeds_give_different_test_sets(self):
        """Sanity check: different seeds must produce different partitions."""
        df = _make_batch_detection_df(seed=1)
        X_te_a = self._get_classic_test_samples(df, seed=0)
        X_te_b = self._get_classic_test_samples(df, seed=99)
        self.assertFalse(
            np.array_equal(X_te_a, X_te_b),
            "Different seeds should produce different test splits",
        )


if __name__ == "__main__":
    unittest.main()
