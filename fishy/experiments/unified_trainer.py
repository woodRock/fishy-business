# -*- coding: utf-8 -*-
"""
Unified trainer orchestrator for all model types (Deep, Classic, Contrastive, Evolutionary).
Supports both single-run and batch-orchestrated experiments.
"""

import logging
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union
from dataclasses import asdict
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from fishy._core.config import TrainingConfig, ExperimentConfig
from fishy._core.utils import RunContext, get_device, set_seed
from fishy.data.module import create_data_module
from fishy.analysis.benchmark import run_benchmark
from fishy._core.config_loader import load_config

logger = logging.getLogger(__name__)


class UnifiedTrainer:
    """
    Consolidated trainer that dispatches to specific training engines.
    Can run single experiments or batch suites defined by ExperimentConfig.
    """

    def __init__(self, config: Union[TrainingConfig, ExperimentConfig]):
        self.config = config
        self.wandb_run = None

    def run(self) -> Union[Dict[str, Any], pd.DataFrame]:
        """Entry point for all training and orchestration tasks."""
        if isinstance(self.config, ExperimentConfig):
            return self._run_batch()
        else:
            return self._run_single(self.config)

    def _run_batch(self) -> pd.DataFrame:
        """Runs a batch of experiments as defined in ExperimentConfig."""
        exp_cfg = self.config
        logger.info(f"Starting batch experiment: {exp_cfg.name}")

        results_summary = {}
        # Avoid circular import
        from fishy.cli.main import DEFAULT_DATA_PATH, detect_method

        for dataset in exp_cfg.datasets:
            for model in exp_cfg.models:
                logger.info(f"Batch: Model {model} on Dataset {dataset}")

                model_results = []
                for run_id in range(exp_cfg.num_runs):
                    seed = (run_id + 1) * 123
                    set_seed(seed)

                    # Create TrainingConfig from ExperimentConfig defaults + overrides
                    train_cfg = TrainingConfig(
                        model=model,
                        dataset=dataset,
                        run=seed,
                        file_path=DEFAULT_DATA_PATH,
                        benchmark=exp_cfg.benchmark,
                        figures=exp_cfg.figures,
                        statistical=exp_cfg.statistical,
                        wandb_log=exp_cfg.wandb_log,
                    )

                    # Apply overrides from ExperimentConfig
                    for k, v in exp_cfg.overrides.items():
                        if hasattr(train_cfg, k):
                            setattr(train_cfg, k, v)

                    train_cfg.method = detect_method(model)

                    res = self._run_single(train_cfg)
                    model_results.append(res)

                results_summary[f"{dataset}|||{model}"] = model_results

        # Statistical analysis for the batch
        if exp_cfg.statistical:
            from fishy.analysis.statistical import summarize_results

            logger.info("Performing batch statistical significance analysis...")
            summary_df = summarize_results(results_summary)

            ctx = RunContext(
                dataset="all", method="experiment", model_name=exp_cfg.name
            )
            ctx.save_dataframe(summary_df, "statistical_analysis.csv")

            print("\n--- BATCH STATISTICAL SIGNIFICANCE SUMMARY ---")
            print(summary_df.to_string(index=False))
            print("----------------------------------------------\n")
            return summary_df

        return pd.DataFrame()

    def _run_single(self, config: TrainingConfig) -> Dict[str, Any]:
        """Runs a single training experiment."""
        self.logger = logger

        # Setup W&B
        wandb_run = None
        if config.wandb_log:
            wandb_run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=asdict(config),
                reinit=True,
                group=f"{config.dataset}_{config.method}",
                job_type="training",
            )

        ctx = RunContext(
            dataset=config.dataset,
            method=config.method,
            model_name=config.model,
            wandb_run=wandb_run,
        )
        ctx.save_config(config)
        device = get_device()

        ctx.logger.info(
            f"Starting {config.method} training for {config.model} on {config.dataset}"
        )

        start_time = time.time()
        results = {}

        try:
            if config.transfer:
                results = self._dispatch_transfer(config, wandb_run)
            elif config.method == "deep":
                results = self._dispatch_deep(config, wandb_run)
            elif config.method in ["classic", "evolutionary", "probabilistic"]:
                results = self._dispatch_sklearn(config, wandb_run)
            elif config.method == "contrastive":
                results = self._dispatch_contrastive(config, wandb_run)
            else:
                raise ValueError(f"Unknown training method: {config.method}")

            training_time = time.time() - start_time
            results["total_training_time_s"] = training_time

            # Post-training analysis
            if config.benchmark:
                self._do_benchmark(config, ctx, device, training_time)

            if config.figures:
                self._generate_figures(config, ctx, results)

            if config.xai:
                self._run_xai(ctx)

        finally:
            if wandb_run:
                wandb_run.finish()

        return results

    def _dispatch_transfer(self, config: TrainingConfig, wandb_run: Any):
        from fishy.experiments.transfer import run_sequential_transfer_learning

        return run_sequential_transfer_learning(
            model_name=config.model,
            transfer_datasets=config.transfer_datasets,
            target_dataset=config.target_dataset,
            num_epochs_transfer=config.epochs_transfer,
            num_epochs_finetune=config.epochs_finetune,
            learning_rate=config.learning_rate if config.learning_rate else 1e-3,
            file_path=config.file_path,
            wandb_log=config.wandb_log,
            wandb_project=config.wandb_project,
            wandb_entity=config.wandb_entity,
            run=config.run,
            wandb_run=wandb_run,
        )

    def _dispatch_deep(self, config: TrainingConfig, wandb_run: Any):
        from fishy.experiments.deep_training import run_training_pipeline

        return run_training_pipeline(config, wandb_run=wandb_run)

    def _dispatch_sklearn(self, config: TrainingConfig, wandb_run: Any):
        from fishy.experiments.classic_training import run_sklearn_experiment

        return run_sklearn_experiment(
            config,
            config.model,
            config.dataset,
            config.run,
            config.file_path,
            wandb_run=wandb_run,
        )

    def _dispatch_contrastive(self, config: TrainingConfig, wandb_run: Any):
        from fishy.experiments.contrastive import (
            run_contrastive_experiment,
            ContrastiveConfig,
        )

        c_cfg = ContrastiveConfig(
            contrastive_method=config.model,
            dataset=config.dataset,
            num_epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            file_path=config.file_path,
            wandb_log=config.wandb_log,
            wandb_project=config.wandb_project,
            wandb_entity=config.wandb_entity,
        )
        return run_contrastive_experiment(c_cfg, wandb_run=wandb_run)

    def _do_benchmark(
        self,
        config: TrainingConfig,
        ctx: RunContext,
        device: torch.device,
        training_time: float,
    ):
        data_module = create_data_module(
            dataset_name=config.dataset, file_path=config.file_path
        )
        data_module.setup()
        input_dim = data_module.get_input_dim()
        run_benchmark(
            model=None,
            input_dim=input_dim,
            device=device,
            ctx=ctx,
            training_time=training_time,
        )

    def _generate_figures(
        self, config: TrainingConfig, ctx: RunContext, results: Dict[str, Any]
    ):
        # Figure generation logic (same as before)
        if "epoch_metrics" in results:
            metrics = results["epoch_metrics"]
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            if "train_losses" in metrics and metrics["train_losses"]:
                plt.plot(metrics["train_losses"], label="Train Loss")
            if "val_losses" in metrics and metrics["val_losses"]:
                plt.plot(metrics["val_losses"], label="Val Loss")
            plt.title("Loss vs. Step")
            plt.legend()

            plt.subplot(1, 2, 2)
            accs = []
            if "val_metrics" in metrics:
                accs = [m.get("balanced_accuracy", 0) for m in metrics["val_metrics"]]
            elif "val_balanced_accuracy" in metrics:
                accs = metrics["val_balanced_accuracy"]
            if accs:
                plt.plot(accs, label="Val Acc")
            plt.title("Accuracy vs. Step")
            plt.legend()
            ctx.save_figure(plt, "training_curves.png")
            plt.close()

        preds_dict = results.get("best_val_predictions", results.get("predictions"))
        if preds_dict:
            y_true, y_pred, y_probs = (
                preds_dict.get("labels"),
                preds_dict.get("preds"),
                preds_dict.get("probs"),
            )
            if y_true is not None and y_pred is not None:
                cm = confusion_matrix(y_true, y_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                ctx.save_figure(plt, "confusion_matrix.png")
                plt.close()

    def _run_xai(self, ctx: RunContext):
        pass


def run_unified_training(
    config: Union[TrainingConfig, ExperimentConfig],
) -> Union[Dict[str, Any], pd.DataFrame]:
    """
    Universal entry point for all training tasks.

    Examples:
        >>> from fishy._core.config import TrainingConfig
        >>> cfg = TrainingConfig(model="transformer", dataset="species")
        >>> isinstance(cfg, TrainingConfig)
        True

    Args:
        config (Union[TrainingConfig, ExperimentConfig]): Configuration object.

    Returns:
        Union[Dict[str, Any], pd.DataFrame]: Results of the training run(s).
    """
    trainer = UnifiedTrainer(config)
    return trainer.run()


def run_all_benchmarks(quick: bool = False, wandb_log: bool = False) -> pd.DataFrame:
    """Helper to run a full benchmark suite using UnifiedTrainer and ExperimentConfig."""
    models_cfg = load_config("models")
    datasets_cfg = load_config("datasets")

    classic = list(models_cfg["classic_models"].keys())
    deep = list(models_cfg["deep_models"].keys())
    evo = list(models_cfg["evolutionary_models"].keys())

    datasets = [
        d for d in ["species", "part", "oil", "cross-species"] if d in datasets_cfg
    ]

    if quick:
        num_runs, datasets = 2, ["species"]
        models = ["opls-da", "transformer", "ga"]
    else:
        num_runs = 30
        models = classic + deep + evo

    exp_cfg = ExperimentConfig(
        name="full_benchmark_suite",
        num_runs=num_runs,
        datasets=datasets,
        models=models,
        wandb_log=wandb_log,
        statistical=True,
    )

    return run_unified_training(exp_cfg)
