# -*- coding: utf-8 -*-
"""
Unified CLI wrapper for the fishy business module.
"""

import argparse
import logging
import sys
import argcomplete
from pathlib import Path
from typing import Tuple, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_PATH = str(PROJECT_ROOT / "data" / "REIMS.xlsx")

from fishy._core.config import TrainingConfig
from fishy.experiments.unified_trainer import run_unified_training
from fishy._core.config_loader import load_config

def get_all_models() -> List[str]:
    """Helper to get all registered model names across all methods."""
    cfg = load_config("models")
    all_models = []
    for section in ["deep_models", "classic_models", "evolutionary_models", "contrastive_models"]:
        all_models.extend(list(cfg.get(section, {}).keys()))
    return sorted(list(set(all_models)))

def get_all_datasets() -> List[str]:
    """Helper to get all registered dataset names."""
    cfg = load_config("datasets")
    return sorted(list(cfg.keys()))

def setup_parser() -> argparse.ArgumentParser:
    """Sets up the unified argument parser with context-aware help."""
    # Context detection for help output
    is_transfer = "--transfer" in sys.argv
    is_full = "--all" in sys.argv or "-a" in sys.argv
    
    def get_help(help_str, context_active=True):
        return help_str if (is_full or context_active) else argparse.SUPPRESS

    all_models = get_all_models()
    all_datasets = get_all_datasets()

    parser = argparse.ArgumentParser(
        prog="fishy",
        description="Unified Spectra Deep Learning CLI Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # --- TRAIN COMMAND ---
    train_parser = subparsers.add_parser("train", help="Train a model (Deep, Classic, Evolutionary, Contrastive)")
    
    # Config File support
    train_parser.add_argument("-c", "--config", type=str, help="Path to a YAML configuration file (TrainingConfig or ExperimentConfig)")
    
    # Basic Arguments
    train_parser.add_argument("-m", "--model", type=str, default="transformer", choices=all_models, help="Model architecture")
    train_parser.add_argument("-d", "--dataset", type=str, default="species", choices=all_datasets, help="Dataset name")
    train_parser.add_argument("--all", "-a", action="store_true", help="Show all expert options in --help")
    
    # Analysis & Reporting Flags (Common)
    train_parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarking")
    train_parser.add_argument("--figures", action="store_true", help="Generate training and evaluation figures")
    train_parser.add_argument("--statistical", action="store_true", help="Add statistical significance test to the runs")
    train_parser.add_argument("--wandb-log", action="store_true", help="Enable W&B logging")
    
    # Task specific modes (Common)
    train_parser.add_argument("--transfer", action="store_true", help="Enable sequential transfer learning")
    train_parser.add_argument("--ordinal", nargs="?", const="coral", choices=["coral", "clm"], help="Enable ordinal regression mode (default: coral)")
    train_parser.add_argument("--regression", action="store_true", help="Enable standard regression mode")

    # Hidden Expert/Contextual Flags
    train_parser.add_argument("-fp", "--file-path", type=str, default=DEFAULT_DATA_PATH, help=get_help("Path to dataset", False))
    train_parser.add_argument("--xai", action="store_true", help=get_help("Run Explainable AI analysis", False))
    
    # Transfer Context (Shown automatically if --transfer is in command line)
    train_parser.add_argument("-td", "--transfer-datasets", type=str, nargs="+", choices=all_datasets, help=get_help("Datasets to pre-train on (sequential)", is_transfer))
    train_parser.add_argument("-target", "--target-dataset", type=str, choices=all_datasets, help=get_help("Final dataset to fine-tune on", is_transfer))
    train_parser.add_argument("--epochs-transfer", type=int, default=10, help=get_help("Epochs per transfer phase", is_transfer))
    train_parser.add_argument("--epochs-finetune", type=int, default=20, help=get_help("Epochs for final phase", is_transfer))
    
    # Hyperparameter overrides (Hidden by default, shown with --all)
    hp_group = train_parser.add_argument_group("Hyperparameter Overrides")
    hp_group.add_argument("-e", "--epochs", type=int, help=get_help("Number of epochs/generations", False))
    hp_group.add_argument("-bs", "--batch-size", type=int, help=get_help("Batch size/population size", False))
    hp_group.add_argument("-lr", "--learning-rate", type=float, help=get_help("Learning rate", False))
    hp_group.add_argument("-kf", "--k-folds", type=int, default=3, help=get_help("Number of cross-validation folds", False))
    hp_group.add_argument("-n", "--num-runs", type=int, default=1, help=get_help("Number of independent runs", False))
    hp_group.add_argument("-r", "--run", type=int, default=0, help=get_help("Run identifier/seed", False))
    
    # --- WIZARD COMMAND ---
    subparsers.add_parser("wizard", help="Start the interactive setup wizard")
    
    # --- RUN ALL COMMAND ---
    run_all_parser = subparsers.add_parser("run_all", help="Run the full benchmarking suite")
    run_all_parser.add_argument("-n", "--num-runs", type=int, default=30)
    run_all_parser.add_argument("--quick", action="store_true")
    run_all_parser.add_argument("--wandb-log", action="store_true")

    argcomplete.autocomplete(parser)
    return parser

def detect_method(model_name: str) -> str:
    """Automatically detects the training method based on the model name."""
    cfg = load_config("models")
    model_name = model_name.lower()
    
    if model_name in cfg.get("deep_models", {}):
        return "deep"
    if model_name in cfg.get("classic_models", {}):
        return "classic"
    if model_name in cfg.get("evolutionary_models", {}):
        return "evolutionary"
    if model_name in cfg.get("contrastive_models", {}):
        return "contrastive"
    
    return "deep" # Default fallback

def main() -> None:
    parser = setup_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    try:
        if args.command == "train":
            if args.config:
                # 1. Determine if it's an ExperimentConfig or TrainingConfig
                import yaml
                with open(args.config, "r") as f:
                    config_data = yaml.safe_load(f)
                
                # ExperimentConfig usually has 'models' list
                if isinstance(config_data, dict) and "models" in config_data:
                    from fishy.experiments.config_orchestrator import run_experiment_from_config
                    run_experiment_from_config(args.config)
                else:
                    # 2. Try as a single TrainingConfig
                    config = TrainingConfig.from_yaml(args.config)
                    # Detect method if not in YAML
                    if not hasattr(config, "method") or not config.method:
                        config.method = detect_method(config.model)
                    
                    if config.num_runs > 1 or config.statistical:
                        # Handle multiple runs even for single config
                        results = []
                        for i in range(config.num_runs):
                            config.run = (i + 1) * 123
                            results.append(run_unified_training(config))
                        
                        if config.statistical:
                            from fishy.analysis.statistical import summarize_results
                            # Compare against OPLS-DA if this isn't OPLS-DA
                            if config.model != "opls-da":
                                print(f"Running baseline (opls-da) for statistical comparison...")
                                baseline_results = []
                                for i in range(config.num_runs):
                                    b_config = TrainingConfig(
                                        model="opls-da", dataset=config.dataset, 
                                        run=(i + 1) * 123, method="classic",
                                        file_path=config.file_path
                                    )
                                    baseline_results.append(run_unified_training(b_config))
                                
                                results_map = {
                                    f"{config.dataset}|||{config.model}": results,
                                    f"{config.dataset}|||opls-da": baseline_results
                                }
                            else:
                                results_map = {f"{config.dataset}|||{config.model}": results}
                            
                            summary_df = summarize_results(results_map)
                            print("\n--- STATISTICAL SIGNIFICANCE SUMMARY ---")
                            print(summary_df.to_string(index=False))
                            print("----------------------------------------\n")
                    else:
                        run_unified_training(config)
            else:
                # Standard CLI execution
                method = detect_method(args.model)
                config = TrainingConfig.from_args(args)
                config.method = method
                
                if config.num_runs > 1 or config.statistical:
                    # If statistical is requested but num_runs is 1, default to 5 runs for some significance
                    n_runs = config.num_runs if config.num_runs > 1 else (5 if config.statistical else 1)
                    results = []
                    for i in range(n_runs):
                        config.run = (i + 1) * 123
                        results.append(run_unified_training(config))
                    
                    if config.statistical:
                        from fishy.analysis.statistical import summarize_results
                        if config.model != "opls-da":
                            print(f"Running baseline (opls-da) for statistical comparison...")
                            baseline_results = []
                            for i in range(n_runs):
                                b_config = TrainingConfig(
                                    model="opls-da", dataset=config.dataset, 
                                    run=(i + 1) * 123, method="classic",
                                    file_path=config.file_path
                                )
                                baseline_results.append(run_unified_training(b_config))
                            
                            results_map = {
                                f"{config.dataset}|||{config.model}": results,
                                f"{config.dataset}|||opls-da": baseline_results
                            }
                        else:
                            results_map = {f"{config.dataset}|||{config.model}": results}
                        
                        summary_df = summarize_results(results_map)
                        print("\n--- STATISTICAL SIGNIFICANCE SUMMARY ---")
                        print(summary_df.to_string(index=False))
                        print("----------------------------------------\n")
                else:
                    run_unified_training(config)
            
        elif args.command == "wizard":
            from fishy.cli.wizard import run_wizard
            run_wizard()

        elif args.command == "run_all":
            from fishy.experiments.orchestrator import run_all_experiments
            run_all_experiments(
                num_runs=args.num_runs,
                quick=args.quick,
                wandb_log=args.wandb_log,
                file_path=DEFAULT_DATA_PATH
            )
            
    except Exception as e:
        logging.error(f"Error executing command {args.command}: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
