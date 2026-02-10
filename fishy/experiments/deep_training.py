# -*- coding: utf-8 -*-
"""
Training orchestrator for the deep learning pipeline.
"""

import logging
import time
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any, Callable
import wandb

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from fishy._core.config import TrainingConfig
from fishy._core.factory import create_model
from fishy._core.utils import RunContext
from fishy.experiments.pre_training import PreTrainer, PreTrainingConfig
from fishy.engine.training_loops import train_model, evaluate_model
from fishy.data.module import create_data_module
from fishy.data.datasets import CustomDataset, SiameseDataset
from fishy.engine.losses import coral_loss, cumulative_link_loss

class ModelTrainer:
    """
    Orchestrates the model training pipeline, from data setup to pre-training and fine-tuning.
    """

    N_CLASSES_PER_DATASET = {
        "species": 2,
        "part": 7,
        "oil": 7,
        "cross-species": 3,
        "cross-species-hard": 15,
        "instance-recognition": 2,
        "instance-recognition-hard": 24,
    }
    PRETRAIN_TASK_DEFINITIONS: List[
        Tuple[str, Callable[["ModelTrainer"], int], str, bool, Dict[str, Any]]
    ] = [
        (
            "masked_spectra_modelling",
            lambda self: self.n_features,
            "pre_train_masked_spectra",
            False,
            {},
        ),
        ("next_spectra_prediction", lambda self: 2, "pre_train_next_spectra", True, {}),
        (
            "next_peak_prediction",
            lambda self: self.n_features,
            "pre_train_peak_prediction",
            False,
            {"peak_threshold": 0.1, "window_size": 5},
        ),
        (
            "spectrum_denoising_autoencoding",
            lambda self: self.n_features,
            "pre_train_denoising_autoencoder",
            False,
            {},
        ),
        (
            "peak_parameter_regression",
            lambda self: self.n_features,
            "pre_train_peak_parameter_regression",
            False,
            {},
        ),
        (
            "spectrum_segment_reordering",
            lambda self: 4 * 4,
            "pre_train_spectrum_segment_reordering",
            False,
            {"num_segments": 4},
        ),
        (
            "contrastive_transformation_invariance_learning",
            lambda self: 128,
            "pre_train_contrastive_invariance",
            False,
            {"embedding_dim": 128},
        ),
    ]

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.wandb_run = None
        if self.config.wandb_log:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=asdict(self.config), # Pass TrainingConfig as W&B config
                reinit=True, # Important for multiple runs in one script
                group=f"{self.config.dataset}_{self.config.model}", # Group runs by dataset and model
                job_type="training"
            )
        self.ctx = RunContext(dataset=config.dataset, method="deep", model_name=config.model, wandb_run=self.wandb_run)
        self.logger = self.ctx.logger
        self.ctx.save_config(config)

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        dataset_name = self.config.dataset
        if self.config.regression and self.config.dataset == "oil":
            dataset_name = "oil_regression"

        if self.config.regression:
            self.n_classes = 1
        else:
            if dataset_name not in self.N_CLASSES_PER_DATASET:
                raise ValueError(f"Invalid dataset: {dataset_name}")
            self.n_classes = self.N_CLASSES_PER_DATASET[dataset_name]
            if "instance-recognition" in dataset_name:
                self.n_classes = 2

        self.data_module = create_data_module(
            file_path=config.file_path,
            dataset_name=dataset_name,
            batch_size=config.batch_size,
            augmentation_config=config,
        )
        self.data_module.setup()
        self.n_features = self.data_module.get_input_dim()

    def pre_train(self) -> Optional[nn.Module]:
        self.logger.info("Evaluating pre-training phase")
        enabled_tasks = [
            task
            for task in self.PRETRAIN_TASK_DEFINITIONS
            if getattr(self.config, task[0], False)
        ]
        if not enabled_tasks:
            self.logger.info("No pre-training tasks enabled.")
            return None

        self.logger.info(
            f"Enabled pre-training tasks: {', '.join(t[0] for t in enabled_tasks)}"
        )
        if self.data_module is None:
            self.logger.error("Pre-training DataModule not set.")
            return None
        train_loader: DataLoader = self.data_module.get_train_dataloader()
        val_loader: Optional[DataLoader] = (
            self.data_module.get_val_dataloader()
            if hasattr(self.data_module, "get_val_dataloader")
            else None
        )

        pre_train_cfg = PreTrainingConfig(
            num_epochs=self.config.epochs,
            file_path=self.config.file_path,
            device=self.device,
            n_features=self.n_features,
        )

        model_after_last_task: Optional[nn.Module] = None
        for flag, out_dim_fn, method, req_val, kwargs in enabled_tasks:
            self.logger.info(f"Starting pre-training task: {flag}")
            output_dim = out_dim_fn(self)

            current_model = create_model(self.config, self.n_features, output_dim).to(
                self.device
            )
            if model_after_last_task:
                self._handle_weight_chaining(current_model, model_after_last_task)

            pre_trainer = PreTrainer(
                model=current_model,
                config=pre_train_cfg,
                optimizer=torch.optim.AdamW(
                    current_model.parameters(), lr=self.config.learning_rate
                ),
            )

            call_args = [train_loader]
            if req_val:
                if val_loader is None:
                    self.logger.warning(
                        f"Validation loader for {flag} not found, passing None."
                    )
                call_args.append(val_loader)

            start_time = time.time()
            trained_model = getattr(pre_trainer, method)(*call_args, **kwargs)
            self.logger.info(f"{flag} training time: {time.time() - start_time:.2f}s")

            # Save pre-trained checkpoint
            checkpoint_path = self.ctx.get_checkpoint_path(f"pretrained_{flag}.pth")
            torch.save(trained_model.state_dict(), checkpoint_path)
            self.logger.info(
                f"Pre-trained weights for {flag} saved to {checkpoint_path}"
            )

            model_after_last_task = trained_model

        self.logger.info("Pre-training completed.")
        return model_after_last_task

    def _handle_weight_chaining(self, current_model: nn.Module, prev_model: nn.Module):
        self.logger.info(
            f"Attempting to load weights from previous model for {self.config.model}"
        )
        try:
            prev_state_dict = prev_model.state_dict()
            current_model_dict = current_model.state_dict()

            load_state_dict = {
                k: v
                for k, v in prev_state_dict.items()
                if k in current_model_dict and v.shape == current_model_dict[k].shape
            }
            missing_keys, unexpected_keys = current_model.load_state_dict(
                load_state_dict, strict=False
            )
            if missing_keys:
                self.logger.warning(f"Chaining: Missing keys: {missing_keys}")
            if unexpected_keys:
                self.logger.warning(f"Chaining: Unexpected keys: {unexpected_keys}")
            self.logger.info("Weight chaining: successfully loaded compatible weights.")
        except Exception as e:
            self.logger.warning(
                f"Weight chaining failed: {e}. Model will train from scratch."
            )

    def _adapt_pretrained_model_for_finetuning(
        self, model: nn.Module, pre_trained_model: nn.Module
    ):
        """
        Adapts a pre-trained model for fine-tuning by loading compatible weights.
        """
        self.logger.info("Adapting pre-trained model for fine-tuning...")
        self._handle_weight_chaining(model, pre_trained_model)

    def analyze_oil_predictions(self, predictions: Dict[str, np.ndarray], fold: int):
        if not predictions:
            self.logger.warning(f"Fold {fold + 1}: No predictions to analyze.")
            return

        true_labels = predictions["labels"]
        pred_labels = predictions["preds"]

        if self.config.regression:
            from sklearn.metrics import r2_score

            mae = np.mean(np.abs(true_labels - pred_labels))
            self.logger.info(f"Fold {fold + 1} Regression MAE: {mae:.4f}")
            r2 = r2_score(true_labels, pred_labels)
            self.logger.info(f"Fold {fold + 1} Regression R2 Score: {r2:.4f}")
        else:
            mae = np.mean(np.abs(true_labels - pred_labels))
            self.logger.info(f"Fold {fold + 1} Ordinal MAE: {mae:.4f}")
            cm = confusion_matrix(true_labels, pred_labels)
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title(f"Fold {fold + 1} Confusion Matrix for Oil Dataset")
            self.ctx.save_figure(plt, f"oil_confusion_matrix_fold_{fold + 1}.png")
            plt.close()

        errors = pred_labels - true_labels
        plt.figure()
        plt.hist(
            errors, bins=np.arange(errors.min(), errors.max() + 2) - 0.5, rwidth=0.8
        )
        plt.xlabel("Prediction Error (Predicted - True)")
        plt.ylabel("Frequency")
        plt.title(f"Fold {fold + 1} Prediction Error Distribution for Oil Dataset")
        self.ctx.save_figure(plt, f"oil_prediction_error_fold_{fold + 1}.png")
        plt.close()

    def train(self, pre_trained_model: Optional[nn.Module] = None) -> nn.Module:
        from sklearn.model_selection import train_test_split

        if self.data_module is None:
            self.logger.error("Fine-tuning DataModule not set.")
            return create_model(self.config, self.n_features, self.n_classes)

        if "instance-recognition" in self.config.dataset:
            self.logger.info(
                "Using Train/Validation/Test split on Siamese pairs for instance-recognition."
            )

            full_dataset_samples = self.data_module.get_dataset().samples.cpu().numpy()
            full_dataset_labels = self.data_module.get_dataset().labels.cpu().numpy()
            full_siamese_dataset = SiameseDataset(
                full_dataset_samples, full_dataset_labels
            )

            pair_indices = np.arange(len(full_siamese_dataset))
            pair_labels_for_stratify = (
                full_siamese_dataset.paired_labels.cpu().numpy().flatten()
            )

            train_val_indices, test_indices = train_test_split(
                pair_indices,
                test_size=0.2,
                random_state=self.config.run,
                stratify=pair_labels_for_stratify,
            )
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=0.25,
                random_state=self.config.run,
                stratify=pair_labels_for_stratify[train_val_indices],
            )

            train_dataset = Subset(full_siamese_dataset, train_indices)
            val_dataset = Subset(full_siamese_dataset, val_indices)
            test_dataset = Subset(full_siamese_dataset, test_indices)

            self.logger.info(
                f"Data split: {len(train_dataset)} train pairs, {len(val_dataset)} validation pairs, {len(test_dataset)} test pairs."
            )

            pin_memory_val = True if self.device.type == "cuda" else False
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=pin_memory_val,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=pin_memory_val,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=pin_memory_val,
            )

            model_to_finetune = create_model(
                self.config, self.n_features, self.n_classes
            ).to(self.device)
            if pre_trained_model:
                self.logger.info("Transferring pre-trained weights for fine-tuning.")
                self._adapt_pretrained_model_for_finetuning(
                    model_to_finetune, pre_trained_model
                )

            if self.config.regression:
                criterion = nn.MSELoss()
            elif self.config.use_coral:
                criterion = coral_loss
            else:
                criterion = nn.CrossEntropyLoss(
                    label_smoothing=self.config.label_smoothing
                )
            optimizer = torch.optim.AdamW(
                model_to_finetune.parameters(), lr=self.config.learning_rate
            )

            trained_model, train_val_metrics = train_model(
                model=model_to_finetune,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=self.config.epochs,
                patience=self.config.early_stopping,
                is_augmented=self.config.data_augmentation,
                device=self.device,
                use_coral=self.config.use_coral,
                num_classes=self.n_classes,
                regression=self.config.regression,
                ctx=self.ctx, # Pass context
            )

            self.logger.info("Evaluating on the test set.")
            test_results = evaluate_model(
                trained_model,
                test_loader,
                criterion,
                self.device,
                self.config.use_coral,
                self.n_classes,
                regression=self.config.regression,
            )

            # Advanced Visualizations for W&B
            if self.ctx.wandb_run:
                # Extract class names if available, otherwise use indices
                class_names = [str(i) for i in range(self.n_classes)]
                if hasattr(self.data_module, 'get_class_names'):
                    class_names = self.data_module.get_class_names()

                y_true = test_results["predictions"]["labels"]
                y_preds = test_results["predictions"]["preds"]
                y_probs = test_results["predictions"]["probs"]

                # 1. Log Summary Charts (Confusion Matrix, ROC, PR)
                if not self.config.regression and y_probs is not None:
                    self.ctx.log_summary_charts(y_true, y_probs, class_names)

                # 2. Log Prediction Table with Spectral Plots
                # Get a sample of spectra from the test loader
                test_spectra, _ = next(iter(test_loader))
                test_spectra_np = test_spectra.cpu().numpy()
                self.ctx.log_prediction_table(
                    spectra=test_spectra_np,
                    preds=y_preds[:len(test_spectra_np)],
                    targets=y_true[:len(test_spectra_np)],
                    probs=y_probs[:len(test_spectra_np)] if y_probs is not None else np.eye(self.n_classes)[y_preds[:len(test_spectra_np)]],
                    class_names=class_names,
                    table_name="test_predictions_samples"
                )

            final_metrics = {
                "train_loss": train_val_metrics.get("train_loss"),
                "train_accuracy": train_val_metrics.get("train_accuracy"),
                "val_loss": train_val_metrics.get("val_loss"),
                "val_accuracy": train_val_metrics.get("val_accuracy"),
                "best_epoch": train_val_metrics.get("epoch"),
                "test_loss": test_results.get("loss"),
                "test_accuracy": test_results.get("metrics", {}).get(
                    "balanced_accuracy"
                ),
            }

            self.ctx.save_results(final_metrics, filename="final_metrics.json")
            torch.save(
                trained_model.state_dict(),
                self.ctx.get_checkpoint_path("final_model.pth"),
            )
            return final_metrics

        else:
            full_dataset_samples = self.data_module.get_dataset().samples.cpu().numpy()
            full_dataset_labels = self.data_module.get_dataset().labels.cpu().numpy()
            self.logger.info(
                "Starting main fine-tuning phase with Stratified K-Fold Cross-Validation"
            )
            k_folds = self.config.k_folds
            cv_splitter = StratifiedKFold(
                n_splits=k_folds, shuffle=True, random_state=self.config.run
            )
            split_args = (full_dataset_samples, np.argmax(full_dataset_labels, axis=1))

            all_fold_metrics = []

            for fold, (train_index, val_index) in enumerate(
                cv_splitter.split(*split_args)
            ):
                self.logger.info(f"--- Starting Fold {fold + 1}/{k_folds} ---")

                X_train, X_val = (
                    full_dataset_samples[train_index],
                    full_dataset_samples[val_index],
                )
                y_train, y_val = (
                    full_dataset_labels[train_index],
                    full_dataset_labels[val_index],
                )

                dataset_name_str = (
                    self.data_module.processor.dataset_type.name.lower().replace(
                        "_", "-"
                    )
                )
                dataset_class = (
                    SiameseDataset
                    if "instance-recognition" in dataset_name_str
                    else CustomDataset
                )

                train_dataset = dataset_class(X_train, y_train)
                val_dataset = dataset_class(X_val, y_val)

                if isinstance(train_dataset, SiameseDataset):
                    num_train_pairs = len(train_dataset)
                    num_train_pos_pairs = np.sum(
                        train_dataset.paired_labels.cpu().numpy() == 1
                    )
                    self.logger.info(
                        f"Train pairs: {num_train_pairs} (Positive: {num_train_pos_pairs})"
                    )
                    if num_train_pairs == 0 or num_train_pos_pairs == 0:
                        self.logger.warning(
                            f"Fold {fold + 1} train set has no pairs or no positive pairs. Skipping fold."
                        )
                        continue
                else:
                    self.logger.info(f"Train samples: {len(train_dataset)}")

                pin_memory_val = True if self.device.type == "cuda" else False
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=pin_memory_val,
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=pin_memory_val,
                )

                model_to_finetune = create_model(
                    self.config, self.n_features, self.n_classes
                ).to(self.device)

                if pre_trained_model:
                    self.logger.info(
                        "Transferring pre-trained weights for fine-tuning."
                    )
                    self._adapt_pretrained_model_for_finetuning(
                        model_to_finetune, pre_trained_model
                    )

                if self.config.regression:
                    criterion = nn.MSELoss()
                elif self.config.use_coral:
                    criterion = coral_loss
                elif self.config.use_cumulative_link:
                    criterion = lambda logits, labels: cumulative_link_loss(
                        logits, labels, self.n_classes
                    )
                else:
                    criterion = nn.CrossEntropyLoss(
                        label_smoothing=self.config.label_smoothing
                    )
                optimizer = torch.optim.AdamW(
                    model_to_finetune.parameters(), lr=self.config.learning_rate
                )

                trained_model_instance, metrics = train_model(
                    model=model_to_finetune,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    num_epochs=self.config.epochs,
                    patience=self.config.early_stopping,
                    is_augmented=self.config.data_augmentation,
                    device=self.device,
                    use_coral=self.config.use_coral,
                    use_cumulative_link=self.config.use_cumulative_link,
                    num_classes=self.n_classes,
                    regression=self.config.regression,
                    ctx=self.ctx, # Pass context
                )
                if "best_val_predictions" in metrics:
                    if self.config.dataset == "oil":
                        self.analyze_oil_predictions(
                            metrics["best_val_predictions"], fold
                        )

                    # Advanced Visualizations for W&B (last fold)
                    if fold == k_folds - 1 and self.ctx.wandb_run:
                        class_names = [str(i) for i in range(self.n_classes)]
                        if hasattr(self.data_module, 'get_class_names'):
                            class_names = self.data_module.get_class_names()

                        y_true = metrics["best_val_predictions"]["labels"]
                        y_preds = metrics["best_val_predictions"]["preds"]
                        y_probs = metrics["best_val_predictions"]["probs"]

                        if not self.config.regression and y_probs is not None:
                            self.ctx.log_summary_charts(y_true, y_probs, class_names)

                        val_spectra, _ = next(iter(val_loader))
                        val_spectra_np = val_spectra.cpu().numpy()
                        self.ctx.log_prediction_table(
                            spectra=val_spectra_np,
                            preds=y_preds[:len(val_spectra_np)],
                            targets=y_true[:len(val_spectra_np)],
                            probs=y_probs[:len(val_spectra_np)] if y_probs is not None else np.eye(self.n_classes)[y_preds[:len(val_spectra_np)]],
                            class_names=class_names,
                            table_name="val_predictions_samples_last_fold"
                        )

                    del metrics["best_val_predictions"]

                all_fold_metrics.append(metrics)

                # Save fold model
                torch.save(
                    trained_model_instance.state_dict(),
                    self.ctx.get_checkpoint_path(f"model_fold_{fold+1}.pth"),
                )

            self.logger.info("Cross-Validation finished.")

            if all_fold_metrics:
                stats = {
                    k: np.mean(
                        [
                            m[k]
                            for m in all_fold_metrics
                            if isinstance(m[k], (int, float))
                        ]
                    )
                    for k in all_fold_metrics[0]
                    if isinstance(all_fold_metrics[0][k], (int, float))
                }
                self.logger.info(f"Average metrics across {k_folds} folds: {stats}")

                suffix = "classification"
                if self.config.regression:
                    suffix = "regression"
                elif self.config.use_coral:
                    suffix = "coral"
                elif self.config.use_cumulative_link:
                    suffix = "cumulative_link"

                file_name = f"aggregated_stats_{suffix}.json"
                self.ctx.save_results(
                    {
                        "config": asdict(self.config),
                        "stats": stats,
                        "folds": all_fold_metrics,
                    },
                    filename=file_name,
                )
                return stats
            else:
                self.logger.warning("No folds completed successfully.")
                return {}

def run_training_pipeline(config: TrainingConfig):
    """Executes the training pipeline for a given configuration."""
    trainer = ModelTrainer(config)
    logger = trainer.logger
    logger.info(f"Training configuration: {config}")
    logger.info(f"Using device: {trainer.device}")
    logger.info("Starting training pipeline")

    try:
        # --- Pre-training Phase ---
        any_pretrain_task_enabled = any(
            getattr(config, task[0])
            for task in ModelTrainer.PRETRAIN_TASK_DEFINITIONS
        )
        pre_trained_model = None
        if any_pretrain_task_enabled:
            pre_trained_model = trainer.pre_train()
        else:
            logger.info("Skipping pre-training phase.")

        # --- Fine-tuning Phase ---
        metrics = trainer.train(pre_trained_model)
        logger.info("Training pipeline completed.")
        return metrics
    finally:
        if trainer.wandb_run:
            trainer.wandb_run.finish()
