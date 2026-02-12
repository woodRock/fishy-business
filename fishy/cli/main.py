# -*- coding: utf-8 -*-
"""
Unified CLI wrapper for the fishy business module.
Supports context-sensitive help and expert flag management.
"""

import argparse
import logging
import sys
import argcomplete
from pathlib import Path
from typing import Tuple, List, Dict, Any

DEFAULT_DATA_PATH = None

from fishy._core.config import TrainingConfig
from fishy.experiments.unified_trainer import run_unified_training
from fishy._core.config_loader import load_config
from fishy._core.utils import set_seed, console
from rich.panel import Panel
from rich.table import Table
from fishy.analysis.statistical import summarize_results, display_statistical_summary

def get_all_models() -> List[str]:
    cfg = load_config("models")
    all_models = []
    for section in [
        "deep_models", "classic_models", "evolutionary_models", 
        "contrastive_models", "probabilistic_models"
    ]:
        all_models.extend(list(cfg.get(section, {}).keys()))
    return sorted(list(set(all_models)))

def get_all_datasets() -> List[str]:
    cfg = load_config("datasets")
    return sorted(list(cfg.keys()))

def detect_method(model_name: str) -> str:
    cfg = load_config("models")
    m = model_name.lower()
    if m in cfg.get("deep_models", {}): return "deep"
    if m in cfg.get("classic_models", {}): return "classic"
    if m in cfg.get("evolutionary_models", {}): return "evolutionary"
    if m in cfg.get("contrastive_models", {}): return "contrastive"
    if m in cfg.get("probabilistic_models", {}): return "probabilistic"
    return "deep"

def setup_parser() -> argparse.ArgumentParser:
    all_models = get_all_models()
    all_datasets = get_all_datasets()
    
    # Check for context-sensitive help triggers in raw argv
    show_all = "--all" in sys.argv
    selected_model = None
    for i, arg in enumerate(sys.argv):
        if arg in ["-m", "--model"] and i + 1 < len(sys.argv):
            selected_model = sys.argv[i+1]
            break
    
    is_contrastive = selected_model and detect_method(selected_model) == "contrastive"

    parser = argparse.ArgumentParser(
        prog="fishy",
        description="Unified Spectra Deep Learning CLI Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("-c", "--config", type=str, help="Path to YAML config")
    train_parser.add_argument("-m", "--model", type=str, default="transformer", choices=all_models)
    train_parser.add_argument("-d", "--dataset", type=str, default="species", choices=all_datasets)
    train_parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    train_parser.add_argument("--figures", action="store_true", help="Generate analysis figures")
    train_parser.add_argument("--all", action="store_true", help="Show all expert hyperparameters in --help")

    # CONTEXT SENSITIVE: Only show --encoder if model is contrastive or --all is used
    encoder_help = "Encoder type for contrastive learning (e.g., transformer, cnn)"
    if is_contrastive or show_all:
        train_parser.add_argument("--encoder", type=str, default="dense", help=encoder_help)
    else:
        # Hide it from help but still allow it to be parsed if passed
        train_parser.add_argument("--encoder", type=str, default="dense", help=argparse.SUPPRESS)

    # Expert Hyperparameters Group
    hp_group_label = "Hyperparameters" if show_all else "Common Hyperparameters"
    hp_group = train_parser.add_argument_group(hp_group_label)
    hp_group.add_argument("-e", "--epochs", type=int, help="Number of training epochs")
    hp_group.add_argument("-n", "--num-runs", type=int, default=1, help="Number of independent runs")
    hp_group.add_argument("-fp", "--file-path", type=str, default=DEFAULT_DATA_PATH, help="Path to spectral data file. If omitted, uses the internal package dataset.")

    if show_all:
        hp_group.add_argument("--batch-size", type=int, default=64)
        hp_group.add_argument("--lr", "--learning-rate", type=float, default=1e-4)
        hp_group.add_argument("--hidden-dim", type=int, default=128)
        train_parser.add_argument("--wandb-log", action="store_true")
        train_parser.add_argument("--statistical", action="store_true")

    subparsers.add_parser("wizard", help="Interactive setup")
    
    run_all_parser = subparsers.add_parser("run_all", help="Full benchmark suite")
    run_all_parser.add_argument("-n", "--num-runs", type=int, default=30)
    run_all_parser.add_argument("--quick", action="store_true")
    
    argcomplete.autocomplete(parser)
    return parser

def display_final_summary(results: Dict[str, Any]):
    """Prints a beautiful summary table of results."""
    table = Table(title="[bold green]Training Complete - Results Summary[/]", box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Train", justify="right", style="magenta")
    table.add_column("Val", justify="right", style="magenta")
    
    metrics = [("Accuracy", "accuracy"), ("Balanced Accuracy", "balanced_accuracy"), ("MAE", "mae"), ("MSE", "mse"), ("Precision", "precision"), ("Recall", "recall"), ("F1 Score", "f1")]
    
    # Add Loss to the top if traditional metrics are zero (like in contrastive)
    has_meaningful_metrics = any(results.get(f"val_{k}", results.get(k, 0.0)) > 0 for _, k in metrics)
    if not has_meaningful_metrics and "val_loss" in results:
        metrics.insert(0, ("Contrastive Loss", "loss"))

    for label, key in metrics:
        tr = results.get(f"train_{key}", results.get(key, 0.0))
        val = results.get(f"val_{key}", results.get(key, 0.0))
        if isinstance(tr, (int, float)) and isinstance(val, (int, float)):
            table.add_row(label, f"{tr:.4f}", f"{val:.4f}")
    console.print("\n"); console.print(Panel(table, expand=False, border_style="green"))
    if "total_training_time_s" in results: console.print(f"[dim italic]Elapsed training time: {results['total_training_time_s']:.4f} seconds[/]\n")

def main() -> None:
    parser = setup_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help(); sys.exit(0)

    try:
        if args.command == "train":
            if args.config:
                import yaml
                with open(args.config, "r") as f: config_data = yaml.safe_load(f)
                if isinstance(config_data, dict) and "models" in config_data:
                    from fishy._core.config import ExperimentConfig
                    exp_cfg = ExperimentConfig.from_yaml(args.config); run_unified_training(exp_cfg)
                else:
                    config = TrainingConfig.from_yaml(args.config)
                    if not hasattr(config, "method") or not config.method: config.method = detect_method(config.model)
                    _handle_train_execution(config)
            else:
                config = TrainingConfig.from_args(args)
                config.method = detect_method(args.model)
                _handle_train_execution(config)
        elif args.command == "wizard":
            from fishy.cli.wizard import run_wizard; run_wizard()
        elif args.command == "run_all":
            from fishy.experiments.unified_trainer import run_all_benchmarks
            run_all_benchmarks(quick=args.quick, num_runs=args.num_runs)
    except Exception:
        console.print_exception(show_locals=True); sys.exit(1)

def _handle_train_execution(config: TrainingConfig):
    n_runs = config.num_runs if config.num_runs > 1 else (5 if config.statistical else 1)
    results = []
    import time
    start_all = time.time()
    with console.status(f"[bold green]Running {n_runs} experiments...") as status:
        for i in range(n_runs):
            seed = (i + 1) * 123; config.run = seed; set_seed(seed)
            status.update(f"[bold green]Experiment {i+1}/{n_runs}...")
            results.append(run_unified_training(config))
        
        final_res = {}
        if results:
            for k in results[0].keys():
                vals = [r[k] for r in results if k in r and isinstance(r[k], (int, float))]
                if vals: final_res[k] = sum(vals) / len(vals)

        final_res["total_training_time_s"] = time.time() - start_all
        if not config.statistical: display_final_summary(final_res)
        else:
            status.update("[bold yellow]Running statistical comparison...")
            baseline_results = []
            if config.model != "opls-da":
                for i in range(n_runs):
                    seed = (i + 1) * 123; set_seed(seed)
                    b_cfg = TrainingConfig(model="opls-da", dataset=config.dataset, run=seed, method="classic", file_path=config.file_path)
                    baseline_results.append(run_unified_training(b_cfg))
            
            res_map = {f"{config.dataset}|||{config.model}": results}
            if baseline_results: res_map[f"{config.dataset}|||opls-da"] = baseline_results
            summary_df = summarize_results(res_map)
            display_statistical_summary(summary_df, show_significance=config.model != "opls-da")

if __name__ == "__main__":
    main()
