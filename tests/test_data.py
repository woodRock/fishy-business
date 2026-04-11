# -*- coding: utf-8 -*-
import unittest
from unittest.mock import patch
import torch
import numpy as np
import pandas as pd
from fishy.data.datasets import (
    BaseDataset,
    SiameseDataset,
    BalancedBatchSampler,
)
from fishy.data.module import DataProcessor, DataModule, make_pairwise_test_split


class TestDatasets(unittest.TestCase):
    def setUp(self):
        self.samples = np.random.rand(10, 20).astype(np.float32)
        self.labels = np.random.randint(0, 2, (10,)).astype(np.float32)

    def test_base_dataset(self):
        ds = BaseDataset(self.samples, self.labels)
        self.assertEqual(len(ds), 10)
        s, l = ds[0]
        self.assertEqual(s.shape, (20,))
        self.assertIsInstance(s, torch.Tensor)

    def test_siamese_dataset(self):
        ds = SiameseDataset(self.samples, self.labels)
        # 10 samples -> C(10, 2) = 45 pairs
        self.assertEqual(len(ds), 45)
        x1, x2, label, y1, y2 = ds[0]
        self.assertEqual(x1.shape, (20,))
        self.assertEqual(label.shape, (1,))

    def test_balanced_batch_sampler(self):
        pair_labels = np.array([1, 1, 0, 0, 1, 0])
        sampler = BalancedBatchSampler(pair_labels, batch_size=4)
        # num_batches = min(3, 3) // (4 // 2) = 3 // 2 = 1
        self.assertEqual(len(sampler), 1)
        indices = next(iter(sampler))
        self.assertEqual(len(indices), 4)

    def test_random_projection(self):
        # Sparse samples (each row has at least one signal, but mostly zeros)
        samples = np.zeros((5, 10), dtype=np.float32)
        samples[0, 0] = 1.0
        samples[1, 5] = 2.0
        samples[2, 2] = 0.5
        samples[3, 9] = 1.2
        samples[4, 1] = 0.1
        labels = np.zeros((5, 1))

        # Without projection
        ds_no_proj = BaseDataset(samples, labels, random_projection=False)
        # Check that some values are still zero (BaseDataset does normalize, but 0 remains 0)
        self.assertTrue(torch.any(ds_no_proj.samples == 0))

        # With projection
        ds_proj = BaseDataset(samples, labels, random_projection=True, seed=42)
        # Projection should turn zeros into non-zeros (stochastically, almost certainly)
        # We check that it's no longer the same as the original and has fewer/no zeros
        self.assertFalse(torch.allclose(ds_no_proj.samples, ds_proj.samples))
        # In a random projection of a sparse vector, it's highly likely to be fully dense
        self.assertTrue(torch.all(ds_proj.samples != 0))

    def test_quantization(self):
        # Sparse samples
        samples = np.random.randn(5, 10).astype(np.float32)
        labels = np.zeros((5, 1))

        # With projection and quantization
        ds_quant = BaseDataset(
            samples, labels, random_projection=True, quantize=True, seed=42
        )

        # QJL Sign quantization should result in only -1, 1, or 0 values
        # (and 0 only if the projection result was exactly 0, which is rare)
        unique_vals = torch.unique(ds_quant.samples)
        for val in unique_vals:
            self.assertIn(val.item(), [-1.0, 1.0, 0.0])

        # Verify that we actually have some -1 and 1
        self.assertTrue(torch.any(ds_quant.samples == 1.0))
        self.assertTrue(torch.any(ds_quant.samples == -1.0))

    def test_optional_normalization(self):
        # Raw samples with different TICs
        samples = np.array([[1.0, 2.0], [10.0, 20.0]], dtype=np.float32)
        labels = np.zeros((2, 1))

        # Without normalization
        ds_no_norm = BaseDataset(samples, labels, normalize=False)
        self.assertTrue(torch.allclose(ds_no_norm.samples, torch.tensor(samples)))

        # With normalization
        ds_norm = BaseDataset(samples, labels, normalize=True)
        # L2 norm of [1, 10] is sqrt(101) ~ 10.05
        # L2 norm of [2, 20] is sqrt(404) ~ 20.1
        self.assertFalse(torch.allclose(ds_norm.samples, torch.tensor(samples)))
        # Check that columns are normalized
        norms = torch.norm(ds_norm.samples, p=2, dim=0)
        for n in norms:
            self.assertAlmostEqual(n.item(), 1.0, places=5)


class TestDataModule(unittest.TestCase):
    def test_data_processor_init(self):
        proc = DataProcessor("species")
        self.assertEqual(proc.dataset_name, "species")
        self.assertIsInstance(proc.config, dict)

    @patch("fishy.data.module.DataProcessor.load_data")
    def test_data_module_setup(self, mock_load_data):
        # Create a dummy dataframe that fits expectations
        df = pd.DataFrame(
            {
                "m/z": ["H_1", "M_1", "H_2"],
                "feat1": [1.0, 2.0, 3.0],
                "feat2": [4.0, 5.0, 6.0],
            }
        )
        mock_load_data.return_value = df

        dm = DataModule(
            dataset_name="species",
            file_path="dummy.xlsx",
            batch_size=2,
            is_pre_train=True,
        )
        dm.setup()

        self.assertEqual(dm.batch_size, 2)
        self.assertIsNotNone(dm.train_loader)
        self.assertEqual(dm.get_input_dim(), 2)
        self.assertEqual(dm.get_num_classes(), 2)

    @patch("fishy.data.module.DataProcessor.load_data")
    def test_get_filtered_dataframe(self, mock_load_data):
        df = pd.DataFrame(
            {
                "m/z": ["H_1", "M_1", "QC_1"],  # QC should be filtered out
                "feat1": [1.0, 2.0, 3.0],
            }
        )
        mock_load_data.return_value = df
        dm = DataModule(dataset_name="species", file_path="dummy.xlsx")
        dm.setup()
        filtered = dm.get_filtered_dataframe()
        self.assertEqual(len(filtered), 2)
        self.assertIn("Class Name", filtered.columns)
        self.assertEqual(filtered["Class Name"].iloc[0], "Hoki")


class TestMakePairwiseTestSplit(unittest.TestCase):
    def _make_data(self, n_classes=24, n_per_class=3):
        """Simulate batch-detection: 24 classes × 3 samples = 72 samples."""
        n = n_classes * n_per_class
        X = np.random.rand(n, 10).astype(np.float32)
        # one-hot labels
        y_idx = np.repeat(np.arange(n_classes), n_per_class)
        y = np.eye(n_classes, dtype=np.float32)[y_idx]
        return X, y

    def test_sizes_sum_to_total(self):
        X, y = self._make_data()
        X_tr, X_te, y_tr, y_te = make_pairwise_test_split(X, y, run_id=0)
        self.assertEqual(len(X_tr) + len(X_te), len(X))

    def test_test_size_is_half(self):
        X, y = self._make_data()
        X_tr, X_te, y_tr, y_te = make_pairwise_test_split(X, y, run_id=0)
        self.assertAlmostEqual(len(X_te) / len(X), 0.5, delta=0.02)

    def test_same_run_gives_same_split(self):
        X, y = self._make_data()
        X_tr1, X_te1, _, _ = make_pairwise_test_split(X, y, run_id=42)
        X_tr2, X_te2, _, _ = make_pairwise_test_split(X, y, run_id=42)
        np.testing.assert_array_equal(X_tr1, X_tr2)
        np.testing.assert_array_equal(X_te1, X_te2)

    def test_different_runs_give_different_splits(self):
        X, y = self._make_data()
        X_tr1, _, _, _ = make_pairwise_test_split(X, y, run_id=0)
        X_tr2, _, _, _ = make_pairwise_test_split(X, y, run_id=99)
        self.assertFalse(np.array_equal(X_tr1, X_tr2))

    def test_positive_pairs_exist_in_test_set(self):
        """Test set must have ≥2 samples per class for at least some classes."""
        X, y = self._make_data()
        _, X_te, _, y_te = make_pairwise_test_split(X, y, run_id=0)
        y_te_idx = np.argmax(y_te, axis=1)
        counts = np.bincount(y_te_idx)
        self.assertTrue(
            np.any(counts >= 2),
            "Test set has no class with ≥2 samples — pairwise eval would be degenerate",
        )

    def test_no_sample_appears_in_both_splits(self):
        X, y = self._make_data()
        X_tr, X_te, _, _ = make_pairwise_test_split(X, y, run_id=7)
        # Each row of X is unique; no row should appear in both splits
        for row in X_te:
            matches = np.all(X_tr == row, axis=1)
            self.assertFalse(matches.any(), "Test sample leaked into train set")

    def test_classic_and_deep_use_same_test_split(self):
        """Classic passes y as class indices + onehot extra; deep/contrastive passes
        y as onehot directly. Both must produce the same X train/test partition so
        all three method types evaluate on identical held-out samples."""
        n_classes = 24
        n_per_class = 3
        run_id = 42

        np.random.seed(0)
        X = np.random.rand(n_classes * n_per_class, 10).astype(np.float32)
        y_idx = np.repeat(np.arange(n_classes), n_per_class)
        y_onehot = np.eye(n_classes, dtype=np.float32)[y_idx]

        # Classic call signature: (X, y_indices, run_id, y_onehot_extra)
        X_tr_cls, X_te_cls, _, _, _, _ = make_pairwise_test_split(
            X, y_idx, run_id, y_onehot
        )
        # Deep / contrastive call signature: (X, y_onehot, run_id)
        X_tr_deep, X_te_deep, _, _ = make_pairwise_test_split(X, y_onehot, run_id)

        np.testing.assert_array_equal(
            X_tr_cls,
            X_tr_deep,
            err_msg="Classic and deep train splits differ — methods are NOT on the same data",
        )
        np.testing.assert_array_equal(
            X_te_cls,
            X_te_deep,
            err_msg="Classic and deep test splits differ — methods are NOT on the same data",
        )


if __name__ == "__main__":
    unittest.main()
