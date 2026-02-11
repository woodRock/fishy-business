# -*- coding: utf-8 -*-
"""
Orchestrator for running experiments from configuration files.
"""

import logging
from typing import Dict, Any
from fishy._core.config import TrainingConfig, ExperimentConfig
from fishy.experiments.unified_trainer import run_unified_training
from fishy._core.utils import set_seed

logger = logging.getLogger(__name__)

def run_experiment_from_config(config_path: str):
    """
    Runs a batch of experiments as defined in a YAML config.
    """
    exp_cfg = ExperimentConfig.from_yaml(config_path)
    logger.info(f"Starting experiment: {exp_cfg.name}")

    results_summary = {}
    from fishy.cli.main import DEFAULT_DATA_PATH

    for dataset in exp_cfg.datasets:
        for model in exp_cfg.models:
            logger.info(f"Batch: Model {model} on Dataset {dataset}")
            
            model_results = []
            for run_id in range(exp_cfg.num_runs):
                # Base config
                seed = (run_id + 1) * 123
                set_seed(seed)
                
                config = TrainingConfig(
                    model=model,
                    dataset=dataset,
                    run=seed,
                    file_path=DEFAULT_DATA_PATH,
                    benchmark=exp_cfg.benchmark,
                    figures=exp_cfg.figures,
                    wandb_log=exp_cfg.wandb_log
                )
                
                # Apply overrides
                for k, v in exp_cfg.overrides.items():
                    if hasattr(config, k):
                        setattr(config, k, v)
                
                # Run unified training (auto-detects method)
                from fishy.cli.main import detect_method
                config.method = detect_method(model)
                
                res = run_unified_training(config)
                model_results.append(res)
            
            results_summary[f"{dataset}_{model}"] = model_results

    return results_summary
