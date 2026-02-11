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
    DatasetType,
)
from fishy.data.module import DataProcessor, DataModule


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

    def test_dataset_type_from_string(self):
        self.assertEqual(DatasetType.from_string("species"), DatasetType.SPECIES)
        with self.assertRaises(ValueError):
            DatasetType.from_string("invalid")


class TestDataModule(unittest.TestCase):
    def test_data_processor_init(self):
        proc = DataProcessor(DatasetType.SPECIES)
        self.assertEqual(proc.dataset_type, DatasetType.SPECIES)
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
        # features are: feat1, feat2. total 2 features.
        self.assertEqual(dm.get_input_dim(), 2)
        self.assertEqual(dm.get_num_classes(), 2)


if __name__ == "__main__":
    unittest.main()
