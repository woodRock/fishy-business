# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
import json
from pathlib import Path
from fishy._core.config import TrainingConfig, ExperimentConfig
from fishy._core.utils import set_seed, get_device, NumpyEncoder, RunContext
from fishy._core.config_loader import load_config
from fishy._core.factory import get_model_class, create_model


class TestConfig(unittest.TestCase):
    def test_training_config_defaults(self):
        config = TrainingConfig()
        self.assertEqual(config.model, "transformer")
        self.assertEqual(config.batch_size, 64)

    def test_experiment_config(self):
        config = ExperimentConfig(name="test", datasets=["ds1"], models=["m1"])
        self.assertEqual(config.name, "test")
        self.assertIn("ds1", config.datasets)


class TestUtils(unittest.TestCase):
    def test_set_seed(self):
        set_seed(42)
        val1 = np.random.rand()
        set_seed(42)
        val2 = np.random.rand()
        self.assertEqual(val1, val2)

    def test_get_device(self):
        device = get_device()
        self.assertIsInstance(device, torch.device)

    def test_numpy_encoder(self):
        data = {
            "arr": np.array([1, 2]),
            "int": np.int64(10),
            "float": np.float32(0.5),
            "path": Path("/tmp"),
        }
        encoded = json.dumps(data, cls=NumpyEncoder)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["arr"], [1, 2])
        self.assertEqual(decoded["int"], 10)
        self.assertEqual(decoded["float"], 0.5)
        self.assertEqual(decoded["path"], "/tmp")

    @patch("pathlib.Path.mkdir")
    @patch("fishy._core.utils.logging.FileHandler")
    @patch("fishy._core.utils.logging.StreamHandler")
    def test_run_context_init(self, mock_stream, mock_file, mock_mkdir):
        # Configure mocks to behave like real handlers
        mock_file.return_value.level = 20  # INFO
        mock_stream.return_value.level = 20
        ctx = RunContext(
            dataset="test_ds",
            method="test_method",
            model_name="test_model",
            base_output_dir="/tmp/outputs",
        )
        self.assertTrue(
            str(ctx.run_dir).startswith("/tmp/outputs/test_ds/test_method/test_model_")
        )
        self.assertEqual(mock_mkdir.call_count, 5)


class TestConfigLoader(unittest.TestCase):
    def test_load_config_success(self):
        # Assuming datasets.yaml exists
        config = load_config("datasets")
        self.assertIsInstance(config, dict)
        self.assertIn("species", config)

    def test_load_config_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_config("non_existent_config_file_123")


class TestFactory(unittest.TestCase):
    def test_get_model_class(self):
        cls = get_model_class("fishy.models.deep.transformer.Transformer")
        self.assertEqual(cls.__name__, "Transformer")

    def test_create_model(self):
        config = TrainingConfig(
            model="transformer", hidden_dimension=64, num_layers=1, num_heads=2
        )
        model = create_model(config, input_dim=128, output_dim=10)
        self.assertIsInstance(model, torch.nn.Module)


if __name__ == "__main__":
    unittest.main()
