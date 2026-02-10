# -*- coding: utf-8 -*-
"""
Unified CLI wrapper for the fishy business module.
"""

import argparse
import logging
import sys
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_PATH = str(PROJECT_ROOT / "data" / "REIMS.xlsx")

from fishy._core.config import TrainingConfig
from fishy.experiments.deep_training import ModelTrainer, run_training_pipeline
from fishy.experiments.classic_training import run_classic_experiment
from fishy.experiments.benchmark import run_benchmark
from fishy.experiments.transfer import run_sequential_transfer_learning
from fishy.experiments.evolutionary import run_gp_experiment
from fishy.experiments.contrastive import run_contrastive_experiment, ContrastiveConfig
from fishy.analysis.xai import explain_predictions, ExplainerConfig
from fishy._core.factory import MODEL_REGISTRY
from fishy.data.module import create_data_module
from fishy.data.datasets import CustomDataset, SiameseDataset
from fishy.data.augmentation import AugmentationConfig

def setup_base_parser():
    parser = argparse.ArgumentParser(
        prog="fishy", 
        description="Unified Spectra Deep Learning CLI Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    return parser, subparsers

def add_train_args(subparsers):
    train_parser = subparsers.add_parser("train", help="Run the model training pipeline")
    
    # Combined list of deep and classic models
    all_models = list(MODEL_REGISTRY.keys()) + ["knn", "dt", "lr", "lda", "nb", "rf", "svm", "opls-da"]
    
    train_parser.add_argument("-fp", "--file-path", type=str, default=DEFAULT_DATA_PATH, help="Path to dataset")
    train_parser.add_argument("-d", "--dataset", type=str, default="species", choices=ModelTrainer.N_CLASSES_PER_DATASET.keys(), help="Dataset name")
    train_parser.add_argument("-m", "--model", type=str, default="transformer", choices=all_models, help="Model type")
    train_parser.add_argument("-r", "--run", type=int, default=0, help="Run identifier")
    train_parser.add_argument("-nr", "--num-runs", type=int, default=1, help="Number of independent runs")
    train_parser.add_argument("-o", "--output", type=str, default="logs/results_base", help="Output log path")
    train_parser.add_argument("-e", "--epochs", type=int, default=100, help="Training epochs")
    train_parser.add_argument("-bs", "--batch-size", type=int, default=64, help="Batch size")
    train_parser.add_argument("-lr", "--learning-rate", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("-es", "--early-stopping", type=int, default=20, help="Early stopping patience")
    train_parser.add_argument("-do", "--dropout", type=float, default=0.2, help="Dropout")
    train_parser.add_argument("-ls", "--label-smoothing", type=float, default=0.1, help="Label smoothing")
    train_parser.add_argument("-kf", "--k-folds", type=int, default=3, help="K-folds")
    train_parser.add_argument("-hd", "--hidden-dimension", type=int, default=128, help="Hidden dimension")
    train_parser.add_argument("-l", "--num-layers", type=int, default=4, help="Number of layers")
    train_parser.add_argument("-nh", "--num-heads", type=int, default=4, help="Number of attention heads")
    train_parser.add_argument("-da", "--data-augmentation", action="store_true", help="Enable augmentation")
    train_parser.add_argument("--num-augmentations", type=int, default=5, help="Number of augmentations")
    train_parser.add_argument("--noise-level", type=float, default=0.05, help="Noise level")
    train_parser.add_argument("--shift-enabled", action="store_true", help="Enable shift")
    train_parser.add_argument("--scale-enabled", action="store_true", help="Enable scale")
    train_parser.add_argument("--use-coral", action="store_true", help="Use CORAL loss")
    train_parser.add_argument("--use-cumulative-link", action="store_true", help="Use Cumulative Link loss")
    train_parser.add_argument("--regression", action="store_true", help="Perform regression")
    train_parser.add_argument("--wandb-log", action="store_true", help="Log to Weights & Biases")
    train_parser.add_argument("--wandb-project", type=str, default="fishy-business", help="W&B project name")
    train_parser.add_argument("--wandb-entity", type=str, default="victoria-university-of-wellington", help="W&B entity name")

    for task_flag, _, _, _, _ in ModelTrainer.PRETRAIN_TASK_DEFINITIONS:
        train_parser.add_argument(f"--{task_flag.replace('_', '-')}", action="store_true", help=f"Enable {task_flag}")

def add_benchmark_args(subparsers):
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark multiple models")
    bench_parser.add_argument("-fp", "--file-path", type=str, default=DEFAULT_DATA_PATH, help="Path to dataset")
    bench_parser.add_argument("models", type=str, nargs="+", help="Models to benchmark")
    bench_parser.add_argument("-w", "--warmup", type=int, default=0, help="Warmup epochs")
    bench_parser.add_argument("-o", "--output", type=str, default="benchmark_results.csv", help="Output CSV path")
    bench_parser.add_argument("--wandb-log", action="store_true", help="Log to Weights & Biases")
    bench_parser.add_argument("--wandb-project", type=str, default="fishy-business", help="W&B project name")
    bench_parser.add_argument("--wandb-entity", type=str, default="victoria-university-of-wellington", help="W&B entity name")

def add_transfer_args(subparsers):
    trans_parser = subparsers.add_parser("transfer", help="Sequential transfer learning")
    trans_parser.add_argument("-fp", "--file-path", type=str, default=DEFAULT_DATA_PATH, help="Path to dataset")
    trans_parser.add_argument("-m", "--model", type=str, required=True, choices=MODEL_REGISTRY.keys(), help="Model type")
    trans_parser.add_argument("-td", "--transfer-datasets", type=str, nargs="+", required=True, help="Datasets for transfer")
    trans_parser.add_argument("-target", "--target-dataset", type=str, required=True, help="Target dataset")
    trans_parser.add_argument("-et", "--epochs-transfer", type=int, default=10, help="Epochs per transfer phase")
    trans_parser.add_argument("-ef", "--epochs-finetune", type=int, default=20, help="Epochs for finetuning")
    trans_parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3, help="Learning rate")
    trans_parser.add_argument("--wandb-log", action="store_true", help="Log to Weights & Biases")
    trans_parser.add_argument("--wandb-project", type=str, default="fishy-business", help="W&B project name")
    trans_parser.add_argument("--wandb-entity", type=str, default="victoria-university-of-wellington", help="W&B entity name")

def add_xai_args(subparsers):
    xai_parser = subparsers.add_parser("xai", help="Explain model predictions (LIME/Grad-CAM)")
    xai_parser.add_argument("-d", "--dataset", type=str, default="part", help="Dataset name")
    xai_parser.add_argument("-m", "--model", type=str, default="transformer", help="Model type")
    xai_parser.add_argument("-i", "--instance", type=str, default="frames", help="Instance name to explain")
    xai_parser.add_argument("-l", "--label", type=float, nargs="+", help="Target label vector")
    xai_parser.add_argument("--method", type=str, default="lime", choices=["lime", "gradcam"], help="XAI method")

def add_evolutionary_args(subparsers):
    evo_parser = subparsers.add_parser("evolutionary", help="Run Genetic Programming experiments")
    evo_parser.add_argument("-fp", "--file-path", type=str, default=DEFAULT_DATA_PATH, help="Path to dataset")
    evo_parser.add_argument("-d", "--dataset", type=str, default="species", help="Dataset name")
    evo_parser.add_argument("-g", "--generations", type=int, default=10, help="Number of generations")
    evo_parser.add_argument("-p", "--population", type=int, default=1023, help="Population size")
    evo_parser.add_argument("-r", "--run", type=int, default=0, help="Run identifier")
    evo_parser.add_argument("--wandb-log", action="store_true", help="Log to Weights & Biases")
    evo_parser.add_argument("--wandb-project", type=str, default="fishy-business", help="W&B project name")
    evo_parser.add_argument("--wandb-entity", type=str, default="victoria-university-of-wellington", help="W&B entity name")

def add_contrastive_args(subparsers):
    cont_parser = subparsers.add_parser("contrastive", help="Run Contrastive Learning experiments")
    cont_parser.add_argument("-fp", "--file-path", type=str, default=DEFAULT_DATA_PATH, help="Path to dataset")
    cont_parser.add_argument("-m", "--method", type=str, default="simclr", help="Contrastive method")
    cont_parser.add_argument("-e", "--encoder", type=str, default="transformer", help="Encoder type")
    cont_parser.add_argument("-epochs", "--epochs", type=int, default=100, help="Number of epochs")
    cont_parser.add_argument("--wandb-log", action="store_true", help="Log to Weights & Biases")
    cont_parser.add_argument("--wandb-project", type=str, default="fishy-business", help="W&B project name")
    cont_parser.add_argument("--wandb-entity", type=str, default="victoria-university-of-wellington", help="W&B entity name")

def handle_train(args):
    classic_models = ["knn", "dt", "lr", "lda", "nb", "rf", "svm", "opls-da"]
    config = TrainingConfig.from_args(args) # Create config first
    if args.model.lower() in classic_models:
        run_classic_experiment(
            config=config, # Pass config
            model_name=args.model,
            dataset_name=args.dataset,
            run_id=args.run,
            file_path=args.file_path
        )
        return

    if "instance-recognition" in args.dataset and args.num_runs > 1:
        all_runs_metrics = []
        # base_config already created above as config

        for i in range(args.num_runs):
            print(f"--- Starting Run {i + 1}/{args.num_runs} ---")
            args_copy = argparse.Namespace(**vars(args))
            args_copy.run = i
            config_run = TrainingConfig.from_args(args_copy)
            metrics = run_training_pipeline(config_run)
            all_runs_metrics.append(metrics)
    else:
        # config already created above
        run_training_pipeline(config)

def handle_xai(args):
    t_cfg = TrainingConfig(
        file_path=DEFAULT_DATA_PATH,
        model=args.model,
        dataset=args.dataset,
        run=0,
        output="tmp/xai",
        data_augmentation=False,
        masked_spectra_modelling=False,
        next_spectra_prediction=False,
        next_peak_prediction=False,
        spectrum_denoising_autoencoding=False,
        peak_parameter_regression=False,
        spectrum_segment_reordering=False,
        contrastive_transformation_invariance_learning=False,
        early_stopping=0,
        dropout=0.0,
        label_smoothing=0.0,
        epochs=1,
        learning_rate=1e-4,
        batch_size=32,
        hidden_dimension=128,
        num_layers=2,
        num_heads=2,
        num_augmentations=0,
        noise_level=0.0,
        shift_enabled=False,
        scale_enabled=False,
        k_folds=1
    )
    e_cfg = ExplainerConfig(output_dir=Path("tmp/figures/xai"))
    explain_predictions(
        dataset_name=args.dataset,
        model_name=args.model,
        training_config=t_cfg,
        explainer_config=e_cfg,
        instance_name=args.instance,
        target_label=args.label if args.label else [1.0, 0.0],
        method=args.method
    )

def main():
    parser, subparsers = setup_base_parser()
    add_train_args(subparsers)
    add_benchmark_args(subparsers)
    add_transfer_args(subparsers)
    add_xai_args(subparsers)
    add_evolutionary_args(subparsers)
    add_contrastive_args(subparsers)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)

    try:
        if args.command == "train":
            handle_train(args)
        elif args.command == "benchmark":
            run_benchmark(
                args.models, 
                warmup_epochs=args.warmup, 
                output_file=args.output, 
                file_path=args.file_path,
                wandb_log=args.wandb_log,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity
            )
        elif args.command == "transfer":
            run_sequential_transfer_learning(
                model_name=args.model,
                transfer_datasets=args.transfer_datasets,
                target_dataset=args.target_dataset,
                num_epochs_transfer=args.epochs_transfer,
                num_epochs_finetune=args.epochs_finetune,
                learning_rate=args.learning_rate,
                file_path=args.file_path,
                wandb_log=args.wandb_log,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity
            )
        elif args.command == "xai":
            handle_xai(args)
        elif args.command == "evolutionary":
            run_gp_experiment(
                dataset=args.dataset,
                generations=args.generations,
                population=args.population,
                run=args.run,
                data_file_path=args.file_path,
                wandb_log=args.wandb_log,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity
            )
        elif args.command == "contrastive":
            c_cfg = ContrastiveConfig(
                contrastive_method=args.method,
                encoder_type=args.encoder,
                num_epochs=args.epochs,
                file_path=args.file_path,
                dataset=args.dataset if hasattr(args, 'dataset') else "species",
                wandb_log=args.wandb_log,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity
            )
            run_contrastive_experiment(c_cfg)
    except Exception as e:
        logging.error(f"Error executing command {args.command}: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()