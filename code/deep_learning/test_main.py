import unittest
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn
from deep_learning.main import (
    TrainingConfig,
    ModelTrainer,
    create_model,
    MODEL_REGISTRY,
)


class TestDeepLearning(unittest.TestCase):

    def setUp(self):
        """Set up a default configuration for tests."""
        self.args = {
            "file_path": "dummy.xlsx",
            "dataset": "species",
            "model": "transformer",
            "run": 0,
            "output": "logs/test_results",
            "data_augmentation": False,
            "masked_spectra_modelling": True,  # Enable a pre-training task
            "next_spectra_prediction": False,
            "next_peak_prediction": False,
            "spectrum_denoising_autoencoding": False,
            "peak_parameter_regression": False,
            "spectrum_segment_reordering": False,
            "contrastive_transformation_invariance_learning": False,
            "early_stopping": 10,
            "dropout": 0.1,
            "label_smoothing": 0.1,
            "epochs": 1,
            "learning_rate": 1e-4,
            "batch_size": 2,
            "hidden_dimension": 16,
            "num_layers": 1,
            "num_heads": 2,
            "num_augmentations": 5,
            "noise_level": 0.1,
            "shift_enabled": False,
            "scale_enabled": False,
        }
        self.config = TrainingConfig(**self.args)

    @patch("deep_learning.main.ModelTrainer._setup_logging")
    def test_trainer_initialization(self, mock_setup_logging):
        """Test that the ModelTrainer initializes correctly."""
        trainer = ModelTrainer(self.config)
        self.assertEqual(trainer.config, self.config)
        self.assertEqual(trainer.n_classes, 2)
        self.assertEqual(trainer.n_features, 2080)

    def test_create_model_valid(self):
        """Test creating a valid model from the registry."""
        model = create_model(self.config, input_dim=2080, output_dim=2)
        self.assertIsInstance(model, nn.Module)
        self.assertIsInstance(model, MODEL_REGISTRY[self.config.model])

    def test_create_model_invalid(self):
        """Test that creating an invalid model raises a ValueError."""
        self.config.model = "invalid_model"
        with self.assertRaises(ValueError):
            create_model(self.config, input_dim=2080, output_dim=2)

    @patch("deep_learning.main.create_data_module")
    @patch("deep_learning.main.PreTrainer")
    @patch("deep_learning.main.ModelTrainer._setup_logging")
    def test_pre_train(
        self, mock_setup_logging, mock_pre_trainer, mock_create_data_module
    ):
        """Test the pre-training phase."""
        # Setup mocks
        mock_data_module = MagicMock()
        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        mock_data_module.setup.return_value = (mock_train_loader, mock_val_loader)
        mock_create_data_module.return_value = mock_data_module

        mock_pre_trainer_instance = MagicMock()
        mock_pre_trainer.return_value = mock_pre_trainer_instance
        mock_pre_trainer_instance.pre_train_masked_spectra.return_value = MagicMock(
            spec=nn.Module
        )

        # Initialize trainer and run pre-training
        trainer = ModelTrainer(self.config)
        trainer.data_module = mock_data_module
        pre_trained_model = trainer.pre_train()

        # Assertions
        mock_create_data_module.assert_not_called()  # data_module is set manually
        mock_pre_trainer.assert_called_once()
        mock_pre_trainer_instance.pre_train_masked_spectra.assert_called_once()
        self.assertIsInstance(pre_trained_model, nn.Module)

    @patch("deep_learning.main.create_data_module")
    @patch("deep_learning.main.train_model")
    @patch("deep_learning.main.ModelTrainer._setup_logging")
    def test_train(self, mock_setup_logging, mock_train_model, mock_create_data_module):
        """Test the fine-tuning phase."""
        # Setup mocks
        mock_data_module = MagicMock()
        mock_train_loader = MagicMock()
        mock_data_module.get_train_dataloader.return_value = mock_train_loader
        mock_create_data_module.return_value = mock_data_module

        mock_trained_model = MagicMock(spec=nn.Module)
        mock_train_model.return_value = (mock_trained_model, {})

        # Initialize trainer and run fine-tuning
        trainer = ModelTrainer(self.config)
        trainer.data_module = mock_data_module
        final_model = trainer.train()

        # Assertions
        mock_create_data_module.assert_not_called()  # data_module is set manually
        mock_train_model.assert_called_once()
        self.assertIsInstance(final_model, nn.Module)


if __name__ == "__main__":
    unittest.main()
