# -*- coding: utf-8 -*-
import pytest
import unittest.mock as mock
from fishy.experiments.unified_trainer import UnifiedTrainer
from fishy._core.config import ExperimentConfig, TrainingConfig
import pandas as pd

def test_unified_trainer_batch_error_handling():
    """Validates that UnifiedTrainer continues even if one model fails."""
    # Setup config with two models, one will fail
    exp_cfg = ExperimentConfig(
        name="test_error_handling",
        datasets=["species"],
        models=["failing_model", "working_model"],
        num_runs=1
    )
    
    trainer = UnifiedTrainer(exp_cfg)
    
    # Mock _run_single to fail for 'failing_model' and succeed for 'working_model'
    def side_effect(config):
        if config.model == "failing_model":
            raise RuntimeError("CUDA out of memory (simulated)")
        return {
            "accuracy": 0.8,
            "balanced_accuracy": 0.75,
            "val_loss": 0.1,
            "train_loss": 0.05
        }

    with mock.patch.object(UnifiedTrainer, '_run_single', side_effect=side_effect):
        # We also need to mock summarize_results and display_statistical_summary to avoid side effects
        with mock.patch('fishy.experiments.unified_trainer.summarize_results') as mock_sum, \
             mock.patch('fishy.experiments.unified_trainer.display_statistical_summary'), \
             mock.patch('fishy.cli.main.detect_method', return_value="deep"):
            
            # Create a dummy dataframe for summarize_results to return
            mock_sum.return_value = pd.DataFrame([{"model": "working_model", "accuracy": 0.8}])
            
            # This should not raise an exception
            results = trainer._run_batch()
            
            # Verify results only contain the working model
            # Note: results_summary in _run_batch will only have entries for successful models
            # We check if summarize_results was called with the correct data
            call_args = mock_sum.call_args[0][0]
            assert "species|||working_model" in call_args
            assert "species|||failing_model" not in call_args
            assert len(call_args["species|||working_model"]) == 1

def test_unified_trainer_run_single_cleanup():
    """Ensures CUDA cleanup is called even if _run_single fails."""
    train_cfg = TrainingConfig(model="transformer", dataset="species")
    trainer = UnifiedTrainer(train_cfg)
    
    with mock.patch("torch.cuda.is_available", return_value=True), \
         mock.patch("torch.cuda.empty_cache") as mock_empty_cache, \
         mock.patch("gc.collect") as mock_gc:
        
        # Force an error in one of the dispatch methods
        with mock.patch.object(UnifiedTrainer, "_dispatch_deep", side_effect=ValueError("Test Error")):
            with pytest.raises(ValueError, match="Test Error"):
                trainer._run_single(train_cfg)
            
            # Check cleanup was still called
            assert mock_gc.called
            assert mock_empty_cache.called
