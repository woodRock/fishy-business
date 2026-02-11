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
from fishy._core.utils import set_seed, console
from rich.panel import Panel
from rich.table import Table

def get_all_models() -> List[str]:
    cfg = load_config("models")
    all_models = []
    for section in ["deep_models", "classic_models", "evolutionary_models", "contrastive_models", "probabilistic_models"]:
        all_models.extend(list(cfg.get(section, {}).keys()))
    return sorted(list(set(all_models)))

def get_all_datasets() -> List[str]:
    cfg = load_config("datasets")
    return sorted(list(cfg.keys()))

def setup_parser() -> argparse.ArgumentParser:
    all_models = get_all_models()
    all_datasets = get_all_datasets()
    parser = argparse.ArgumentParser(prog="fishy", description="Unified Spectra Deep Learning CLI Tool", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("-c", "--config", type=str, help="Path to YAML config")
    train_parser.add_argument("-m", "--model", type=str, default="transformer", choices=all_models)
    train_parser.add_argument("-d", "--dataset", type=str, default="species", choices=all_datasets)
    train_parser.add_argument("--benchmark", action="store_true")
    train_parser.add_argument("--figures", action="store_true")
    train_parser.add_argument("--statistical", action="store_true")
    train_parser.add_argument("--wandb-log", action="store_true")
    hp_group = train_parser.add_argument_group("Hyperparameters")
    hp_group.add_argument("-e", "--epochs", type=int)
    hp_group.add_argument("-n", "--num-runs", type=int, default=1)
    hp_group.add_argument("-fp", "--file-path", type=str, default=DEFAULT_DATA_PATH)
    subparsers.add_parser("wizard", help="Interactive setup")
    run_all_parser = subparsers.add_parser("run_all", help="Full benchmark suite")
    run_all_parser.add_argument("-n", "--num-runs", type=int, default=30)
    run_all_parser.add_argument("--quick", action="store_true")
    argcomplete.autocomplete(parser)
    return parser

def detect_method(model_name: str) -> str:
    cfg = load_config("models")
    m = model_name.lower()
    if m in cfg.get("deep_models", {}): return "deep"
    if m in cfg.get("classic_models", {}): return "classic"
    if m in cfg.get("evolutionary_models", {}): return "evolutionary"
    if m in cfg.get("contrastive_models", {}): return "contrastive"
    if m in cfg.get("probabilistic_models", {}): return "probabilistic"
    return "deep"

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
                    exp_cfg = ExperimentConfig.from_yaml(args.config)
                    run_unified_training(exp_cfg)
                else:
                    config = TrainingConfig.from_yaml(args.config)
                    if not hasattr(config, "method") or not config.method: config.method = detect_method(config.model)
                    _handle_train_execution(config)
            else:
                config = TrainingConfig.from_args(args)
                config.method = detect_method(args.model)
                _handle_train_execution(config)
        elif args.command == "wizard":
            from fishy.cli.wizard import run_wizard
            run_wizard()
        elif args.command == "run_all":
            from fishy.experiments.unified_trainer import run_all_benchmarks
            with console.status("[bold green]Running full benchmark suite..."):
                run_all_benchmarks(quick=args.quick)
    except Exception:
        console.print_exception(show_locals=True)
        sys.exit(1)

def _handle_train_execution(config: TrainingConfig):
    n_runs = config.num_runs if config.num_runs > 1 else (5 if config.statistical else 1)
    results = []
    
    with console.status(f"[bold green]Running {n_runs} experiments...") as status:
        for i in range(n_runs):
            seed = (i + 1) * 123
            config.run = seed
            set_seed(seed)
            status.update(f"[bold green]Running Experiment {i+1}/{n_runs}...")
            results.append(run_unified_training(config))
        
        if config.statistical and config.model != "opls-da":
            status.update("[bold yellow]Running baseline (opls-da) for comparison...")
            baseline_results = []
            for i in range(n_runs):
                seed = (i + 1) * 123
                set_seed(seed)
                b_config = TrainingConfig(model="opls-da", dataset=config.dataset, run=seed, method="classic", file_path=config.file_path)
                baseline_results.append(run_unified_training(b_config))
            
            from fishy.analysis.statistical import summarize_results
            summary_df = summarize_results({f"{config.dataset}|||{config.model}": results, f"{config.dataset}|||opls-da": baseline_results})
            
            table = Table(title="Statistical Significance Summary")
            for col in summary_df.columns: table.add_column(col)
            for _, row in summary_df.iterrows(): table.add_row(*[str(val) for val in row])
            console.print("\n", table)
