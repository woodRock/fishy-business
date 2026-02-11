# -*- coding: utf-8 -*-
import unittest
from unittest.mock import patch, MagicMock
from fishy.experiments.unified_trainer import UnifiedTrainer, run_unified_training
from fishy._core.config import TrainingConfig


class TestExperiments(unittest.TestCase):
    @patch("fishy.experiments.unified_trainer.UnifiedTrainer._run_single")
    def test_run_unified_training_single(self, mock_run_single):
        mock_run_single.return_value = {"acc": 0.9}
        cfg = TrainingConfig(model="transformer", dataset="species")
        res = run_unified_training(cfg)
        self.assertEqual(res["acc"], 0.9)
        mock_run_single.assert_called_once()

    @patch("fishy.experiments.unified_trainer.UnifiedTrainer._run_batch")
    def test_unified_trainer_batch_dispatch(self, mock_run_batch):
        from fishy._core.config import ExperimentConfig

        mock_run_batch.return_value = MagicMock()  # pd.DataFrame
        exp_cfg = ExperimentConfig(name="batch")
        trainer = UnifiedTrainer(exp_cfg)
        trainer.run()
        mock_run_batch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
