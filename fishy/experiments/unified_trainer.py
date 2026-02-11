# -*- coding: utf-8 -*-
"""
Unified trainer orchestrator for all model types (Deep, Classic, Contrastive, Evolutionary).
"""

import logging
import time
import torch
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import asdict
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from fishy._core.config import TrainingConfig
from fishy._core.utils import RunContext, get_device
from fishy.data.module import create_data_module
from fishy.analysis.benchmark import run_benchmark

logger = logging.getLogger(__name__)

class UnifiedTrainer:
    """
    Consolidated trainer that dispatches to specific training engines
    based on the model type and method.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.wandb_run = None
        if self.config.wandb_log:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=asdict(self.config),
                reinit=True,
                group=f"{self.config.dataset}_{self.config.method}",
                job_type="training",
            )
        
        self.ctx = RunContext(
            dataset=self.config.dataset,
            method=self.config.method,
            model_name=self.config.model,
            wandb_run=self.wandb_run,
        )
        self.logger = self.ctx.logger
        self.ctx.save_config(config)
        self.device = get_device()

    def run(self) -> Dict[str, Any]:
        """Entry point for all training tasks."""
        self.logger.info(f"Starting {self.config.method} training for {self.config.model} on {self.config.dataset}")
        
        start_time = time.time()
        results = {}

        try:
            if self.config.transfer:
                results = self._run_transfer()
            elif self.config.method == "deep":
                results = self._run_deep()
            elif self.config.method == "classic":
                results = self._run_classic()
            elif self.config.method == "contrastive":
                results = self._run_contrastive()
            elif self.config.method == "evolutionary":
                results = self._run_evolutionary()
            else:
                raise ValueError(f"Unknown training method: {self.config.method}")
            
            training_time = time.time() - start_time
            results["total_training_time_s"] = training_time

            # Post-training analysis
            if self.config.benchmark:
                self._do_benchmark(training_time)
            
            if self.config.figures:
                self._generate_figures(results)
            
            if self.config.xai:
                self._run_xai()

        finally:
            if self.wandb_run:
                self.wandb_run.finish()

        return results

    def _run_transfer(self):
        from fishy.experiments.transfer import run_sequential_transfer_learning
        if not self.config.transfer_datasets or not self.config.target_dataset:
            raise ValueError("Transfer learning requires --transfer-datasets and --target-dataset")
            
        return run_sequential_transfer_learning(
            model_name=self.config.model,
            transfer_datasets=self.config.transfer_datasets,
            target_dataset=self.config.target_dataset,
            num_epochs_transfer=self.config.epochs_transfer,
            num_epochs_finetune=self.config.epochs_finetune,
            learning_rate=self.config.learning_rate if self.config.learning_rate else 1e-3,
            file_path=self.config.file_path,
            wandb_log=self.config.wandb_log,
            wandb_project=self.config.wandb_project,
            wandb_entity=self.config.wandb_entity,
            run=self.config.run,
        )

    def _run_deep(self):
        from fishy.experiments.deep_training import run_training_pipeline
        return run_training_pipeline(self.config)

    def _run_classic(self):
        from fishy.experiments.classic_training import run_classic_experiment
        return run_classic_experiment(
            self.config, self.config.model, self.config.dataset, self.config.run, self.config.file_path
        )

    def _run_contrastive(self):
        from fishy.experiments.contrastive import run_contrastive_experiment, ContrastiveConfig
        # Map TrainingConfig to ContrastiveConfig
        c_cfg = ContrastiveConfig(
            contrastive_method=self.config.model,
            dataset=self.config.dataset,
            num_epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            file_path=self.config.file_path,
            wandb_log=self.config.wandb_log,
        )
        return run_contrastive_experiment(c_cfg)

    def _run_evolutionary(self):
        from fishy.experiments.evolutionary import run_evolutionary_experiment
        return run_evolutionary_experiment(
            model_name=self.config.model,
            dataset=self.config.dataset,
            generations=self.config.epochs if self.config.epochs else 10,
            population=self.config.batch_size if self.config.batch_size else 100,
            run=self.config.run,
            data_file_path=self.config.file_path,
            wandb_log=self.config.wandb_log,
        )

    def _do_benchmark(self, training_time: float):
        # We need an instance of the model and input dim
        data_module = create_data_module(dataset_name=self.config.dataset, file_path=self.config.file_path)
        data_module.setup()
        input_dim = data_module.get_input_dim()
        
        # This is a bit of a hack to get the model without knowing its class directly
        # For now, we try to run the benchmark suite which handles torch vs non-torch
        # We'll pass None as model if we can't easily get it here, and the benchmarker will handle it.
        run_benchmark(model=None, input_dim=input_dim, device=self.device, ctx=self.ctx, training_time=training_time)

    def _generate_figures(self, results: Dict[str, Any]):
        self.logger.info("Generating analysis figures...")
        
        # 1. Training Curves (Loss/Accuracy) - Mostly for Deep/Evolutionary
        if "epoch_metrics" in results:
            metrics = results["epoch_metrics"]
            plt.figure(figsize=(12, 5))
            
            # Loss
            plt.subplot(1, 2, 1)
            if "train_losses" in metrics and metrics["train_losses"]:
                plt.plot(metrics["train_losses"], label="Train Loss")
            if "val_losses" in metrics and metrics["val_losses"]:
                plt.plot(metrics["val_losses"], label="Val Loss")
            plt.title("Loss vs. Epoch/Gen"); plt.xlabel("Step"); plt.ylabel("Loss"); plt.legend()
            
            # Accuracy
            plt.subplot(1, 2, 2)
            # Handle both "val_metrics" (list of dicts) and direct lists
            accs = []
            if "val_metrics" in metrics:
                accs = [m.get("balanced_accuracy", 0) for m in metrics["val_metrics"]]
            elif "val_balanced_accuracy" in metrics:
                accs = metrics["val_balanced_accuracy"]
                
            if accs:
                plt.plot(accs, label="Val Balanced Accuracy")
            plt.title("Accuracy vs. Epoch/Gen"); plt.xlabel("Step"); plt.ylabel("Accuracy"); plt.legend()
            
            self.ctx.save_figure(plt, "training_curves.png")
            plt.close()

        # 2. Evaluation Figures (Confusion Matrix, ROC, PR)
        # Use best_val_predictions if available
        preds_dict = results.get("best_val_predictions")
        if not preds_dict and "predictions" in results:
            preds_dict = results["predictions"]

        if preds_dict:
            y_true = preds_dict.get("labels")
            y_pred = preds_dict.get("preds")
            y_probs = preds_dict.get("probs")
            
            if y_true is not None and y_pred is not None:
                # Confusion Matrix
                cm = confusion_matrix(y_true, y_pred)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.title(f"Confusion Matrix - {self.config.model}")
                plt.xlabel("Predicted"); plt.ylabel("True")
                self.ctx.save_figure(plt, "confusion_matrix.png")
                plt.close()

                # ROC and PR Curves (Needs probabilities)
                if y_probs is not None:
                    from sklearn.metrics import roc_curve, precision_recall_curve, auc
                    from sklearn.preprocessing import label_binarize
                    
                    n_classes = y_probs.shape[1]
                    if n_classes == 2:
                        # Binary case: plot a single curve for the positive class
                        plt.figure(figsize=(10, 8))
                        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
                        roc_auc = auc(fpr, tpr)
                        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
                        plt.plot([0, 1], [0, 1], "k--")
                        plt.title(f"ROC Curve - {self.config.model}")
                        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
                        self.ctx.save_figure(plt, "roc_curve.png")
                        plt.close()

                        plt.figure(figsize=(10, 8))
                        precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1])
                        plt.plot(recall, precision, label="PR curve")
                        plt.title(f"Precision-Recall Curve - {self.config.model}")
                        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend()
                        self.ctx.save_figure(plt, "pr_curve.png")
                        plt.close()

                    elif n_classes > 2:
                        # Multi-class case: One-vs-Rest
                        plt.figure(figsize=(10, 8))
                        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
                        for i in range(n_classes):
                            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
                            plt.plot(fpr, tpr, label=f"Class {i} (AUC = {auc(fpr, tpr):.2f})")
                        plt.plot([0, 1], [0, 1], "k--")
                        plt.title(f"ROC Curve - {self.config.model}")
                        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
                        self.ctx.save_figure(plt, "roc_curve.png")
                        plt.close()

                        # PR Curve
                        plt.figure(figsize=(10, 8))
                        for i in range(n_classes):
                            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
                            plt.plot(recall, precision, label=f"Class {i}")
                        plt.title(f"Precision-Recall Curve - {self.config.model}")
                        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend()
                        self.ctx.save_figure(plt, "pr_curve.png")
                        plt.close()
        
        # If classic model and no probs, we might still have just preds
        elif "val_balanced_accuracy" in results and not preds_dict:
            self.logger.warning("Predictions not found in results. Some figures skipped.")

    def _run_xai(self):
        from fishy.analysis.xai import explain_predictions, ExplainerConfig
        self.logger.info("Running XAI analysis...")
        # Simplified XAI run
        e_cfg = ExplainerConfig(output_dir=self.ctx.figure_dir / "xai")
        # This needs a trained model, which should be saved in checkpoints
        pass

def run_unified_training(config: TrainingConfig) -> Dict[str, Any]:
    trainer = UnifiedTrainer(config)
    return trainer.run()
