# -*- coding: utf-8 -*-
"""
Unified trainer orchestrator for all model types (Deep, Classic, Contrastive, Evolutionary).
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
from fishy._core.utils import RunContext, get_device, set_seed, console
from fishy.data.module import create_data_module
from fishy.analysis.benchmark import run_benchmark
from fishy._core.config_loader import load_config
from fishy.analysis.statistical import summarize_results, display_statistical_summary
from rich.table import Table
from rich.panel import Panel

logger = logging.getLogger(__name__)


class UnifiedTrainer:
    """Consolidated trainer that dispatches to specific training engines."""

    def __init__(self, config: Union[TrainingConfig, ExperimentConfig]):
        self.config = config

    def run(self) -> Union[Dict[str, Any], pd.DataFrame]:
        if isinstance(self.config, ExperimentConfig):
            return self._run_batch()

        # Auto-detect method if it's the default "deep" but the model is actually classic/evo/etc.
        # This allows for a cleaner 3-line API in notebooks.
        if hasattr(self.config, "method") and self.config.method == "deep":
            from fishy._core.config_loader import (
                detect_method,
            )  # noqa: F401 (canonical location)

            detected = detect_method(self.config.model)
            if detected != "deep":
                self.config.method = detected

        return self._run_single(self.config)

    def _run_batch(self) -> pd.DataFrame:
        exp_cfg = self.config
        results_summary = {}
        from fishy.cli.main import DEFAULT_DATA_PATH
        from fishy._core.config_loader import detect_method
        from fishy._core.utils import console

        status_manager = (
            console.status(f"[bold blue]Executing Batch: {exp_cfg.name}...")
            if exp_cfg.num_runs > 1
            else None
        )

        try:
            if status_manager:
                status_manager.start()
                console._status = status_manager

            for dataset in exp_cfg.datasets:
                for model in exp_cfg.models:
                    if status_manager:
                        status_manager.update(
                            f"[bold blue]Batch: [bold]{model}[/] on [bold]{dataset}[/]"
                        )
                    model_results = []
                    for run_id in range(exp_cfg.num_runs):
                        try:
                            seed = (run_id + 1) * 123
                            set_seed(seed)
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
                            for k, v in exp_cfg.overrides.items():
                                if hasattr(train_cfg, k):
                                    setattr(train_cfg, k, v)
                            train_cfg.method = detect_method(model)

                            run_results = self._run_single(train_cfg)
                            # Strip heavy objects before storing in the summary list
                            run_results.pop("model", None)
                            run_results.pop("data_module", None)
                            model_results.append(run_results)
                        except Exception as e:
                            logger.error(
                                f"❌ Failed run {run_id} for {model} on {dataset}: {e}"
                            )
                            console.print(
                                f"[bold red]Error in {model}/{dataset} run {run_id}:[/] {e}"
                            )
                            # Continue to next run or model
                            continue

                    if model_results:
                        results_summary[f"{dataset}|||{model}"] = model_results
        finally:
            if status_manager:
                status_manager.stop()
                console._status = None

        summary_df = summarize_results(results_summary)
        display_statistical_summary(summary_df, show_significance=exp_cfg.statistical)

        ctx = RunContext(dataset="all", method="experiment", model_name=exp_cfg.name)
        ctx.save_dataframe(summary_df, "statistical_analysis.csv")
        # Save JSON version for dashboard
        import json
        from fishy._core.utils import NumpyEncoder

        with open(ctx.run_dir / "summary.json", "w") as f:
            json.dump(
                summary_df.to_dict(orient="records"), f, indent=4, cls=NumpyEncoder
            )
        return summary_df

    def _run_single(self, config: TrainingConfig) -> Dict[str, Any]:
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

        ctx_model_name = config.model
        if config.method == "contrastive":
            ctx_model_name = f"{config.model}-{config.encoder_type}"

        ctx = RunContext(
            dataset=config.dataset,
            method=config.method,
            model_name=ctx_model_name,
            wandb_run=wandb_run,
        )
        ctx.save_config(config)
        device = get_device()

        start_time = time.time()
        results = {}
        try:
            # Temporarily stop console status if active to allow inner progress bars
            # Rich only allows one Live/Progress/Status at a time.
            from fishy._core.utils import console

            active_status = getattr(console, "_status", None)
            if active_status:
                active_status.stop()

            if config.transfer:
                results = self._dispatch_transfer(config, wandb_run, ctx)
            elif config.method == "deep":
                results = self._dispatch_deep(config, wandb_run, ctx)
            elif config.method in ["classic", "evolutionary", "probabilistic"]:
                results = self._dispatch_sklearn(config, wandb_run, ctx)
            elif config.method == "contrastive":
                results = self._dispatch_contrastive(config, wandb_run, ctx)

            # Restart if it was running
            if active_status:
                active_status.start()

            # Standardize common fields across all methods
            if "class_names" not in results:
                dm = create_data_module(
                    dataset_name=config.dataset,
                    file_path=config.file_path,
                    polar=config.polar,
                )
                dm.setup()
                results["class_names"] = dm.get_class_names()
                if "data_module" not in results:
                    results["data_module"] = dm

            training_time = time.time() - start_time
            results["total_training_time_s"] = training_time
            if config.benchmark:
                self._do_benchmark(config, ctx, device, training_time)
            if config.figures:
                self._generate_figures(config, ctx, results)
        finally:
            if wandb_run:
                wandb_run.finish()

            # Explicit memory management to prevent CUDA OOM
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()

        return results

    def _dispatch_transfer(self, config, wandb_run, ctx):
        from fishy.experiments.transfer import run_sequential_transfer_learning

        return run_sequential_transfer_learning(
            model_name=config.model,
            transfer_datasets=config.transfer_datasets,
            target_dataset=config.target_dataset,
            num_epochs_transfer=config.epochs_transfer,
            num_epochs_finetune=config.epochs_finetune,
            file_path=config.file_path,
            wandb_log=config.wandb_log,
            run=config.run,
            wandb_run=wandb_run,
        )

    def _dispatch_deep(self, config, wandb_run, ctx):
        from fishy.experiments.deep_training import ModelTrainer

        # Return (model, stats) consistent with other trainers
        trainer = ModelTrainer(config, wandb_run=wandb_run, ctx=ctx)
        pre_trained_model = trainer.pre_train()
        model, stats = trainer.train(pre_trained_model)
        stats["model"] = model
        stats["data_module"] = trainer.data_module

        # Clean up trainer reference to allow GC of its internal model if not in stats
        del trainer
        return stats

    def _dispatch_sklearn(self, config, wandb_run, ctx):
        from fishy.experiments.classic_training import SklearnTrainer

        # Standardized return
        trainer = SklearnTrainer(
            config,
            config.model,
            config.dataset,
            config.run,
            config.file_path,
            wandb_run=wandb_run,
            ctx=ctx,
        )
        model, stats = trainer.run()
        stats["model"] = model
        stats["data_module"] = trainer.data_module

        del trainer
        return stats

    def _dispatch_contrastive(self, config, wandb_run, ctx):
        from fishy.experiments.contrastive import (
            run_contrastive_experiment,
            ContrastiveConfig,
            ContrastiveTrainer,
        )

        c_cfg = ContrastiveConfig(
            contrastive_method=config.model,
            dataset=config.dataset,
            num_epochs=config.epochs if config.epochs else 100,
            batch_size=config.batch_size,
            file_path=config.file_path,
            wandb_log=config.wandb_log,
            encoder_type=config.encoder_type,
            run=config.run,
            random_projection=config.random_projection,
            quantize=config.quantize,
            turbo_quant=config.turbo_quant,
            polar=config.polar,
            normalize=config.normalize,
        )
        trainer = ContrastiveTrainer(c_cfg, wandb_run=wandb_run, ctx=ctx)
        try:
            trainer.setup()
            trainer.train()
            # Standardize contrastive results to include model for XAI
            trainer.metrics["model"] = trainer.model
            trainer.metrics["data_module"] = trainer.data_module
            return trainer.metrics
        finally:
            del trainer

    def _do_benchmark(self, config, ctx, device, training_time):
        dm = create_data_module(
            dataset_name=config.dataset,
            file_path=config.file_path,
            random_projection=config.random_projection,
            quantize=config.quantize,
            turbo_quant=config.turbo_quant,
            polar=config.polar,
            normalize=config.normalize,
            run_id=config.run,
        )
        dm.setup()
        run_benchmark(
            model=None,
            input_dim=dm.get_input_dim(),
            device=device,
            ctx=ctx,
            training_time=training_time,
        )

    def _generate_figures(self, config, ctx, results):
        if "epoch_metrics" in results and results["epoch_metrics"] is not None:
            m = results["epoch_metrics"]
        elif "history" in results and results["history"] is not None:
            m = results["history"]
        else:
            m = None

        if m is not None:
            plt.figure(figsize=(12, 5))

            # 1. Loss Curve
            plt.subplot(1, 2, 1)
            train_loss = m.get("train_losses", m.get("loss", []))
            val_loss = m.get("val_losses", [])
            plt.plot(train_loss, label="Train Loss", color="royalblue", lw=2)
            if val_loss:
                plt.plot(val_loss, label="Val Loss", color="darkorange", lw=2)
            plt.title(f"Loss: {config.model} on {config.dataset}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 2. Balanced Accuracy Curve
            plt.subplot(1, 2, 2)
            val_accs = [
                met.get("balanced_accuracy", 0) for met in m.get("val_metrics", [])
            ]
            train_accs = [
                met.get("balanced_accuracy", 0) for met in m.get("train_metrics", [])
            ]

            if not train_accs and "accuracy" in m:
                train_accs = m["accuracy"]

            if train_accs:
                plt.plot(train_accs, label="Train Acc", color="royalblue", lw=2)
            if val_accs:
                plt.plot(val_accs, label="Val Acc", color="darkorange", lw=2)

            plt.title(f"Balanced Accuracy: {config.model}")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.ylim(0, 1.05)
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            ctx.save_figure(plt, "training_curves.png")
            plt.close()

        # Original history logic kept as fallback for specific contrastive results if needed
        elif "history" in results:
            # Support for contrastive history
            h = results["history"]
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(
                h.get("loss", []), label="Contrastive Loss", color="royalblue", lw=2
            )
            plt.title(f"Contrastive Loss: {config.model}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            plt.plot(
                h.get("accuracy", []),
                label="Pair-wise Accuracy",
                color="darkorange",
                lw=2,
            )
            plt.title(f"Pair-wise Accuracy: {config.model}")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.ylim(0, 1.05)
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            ctx.save_figure(plt, "training_curves.png")
            plt.close()


def run_unified_training(config):
    return UnifiedTrainer(config).run()


def run_all_benchmarks(quick=False, num_runs=None, **kwargs):
    models_cfg = load_config("models")
    classic = list(models_cfg["classic_models"].keys())
    deep = list(models_cfg["deep_models"].keys())
    evo = list(models_cfg["evolutionary_models"].keys())

    if quick:
        actual_runs = num_runs if num_runs is not None else 2
        datasets, models = ["species"], ["opls-da", "transformer"]
    else:
        actual_runs = num_runs if num_runs is not None else 30
        datasets, models = ["species", "part", "oil"], classic + deep + evo

    exp_cfg = ExperimentConfig(
        name="full_benchmark",
        num_runs=actual_runs,
        datasets=datasets,
        models=models,
        statistical=True,
        **kwargs,
    )
    return run_unified_training(exp_cfg)
