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
from fishy._core.config_loader import load_config, detect_method
from fishy._core.utils import set_seed, console, get_device
from rich.panel import Panel
from rich.table import Table
from fishy.analysis.statistical import summarize_results, display_statistical_summary
from fishy.analysis.biomarker import run_biomarker_pipeline


def get_all_models() -> List[str]:
    cfg = load_config("models")
    all_models = []
    for section in [
        "deep_models",
        "classic_models",
        "evolutionary_models",
        "contrastive_models",
        "probabilistic_models",
    ]:
        all_models.extend(list(cfg.get(section, {}).keys()))
    return sorted(list(set(all_models)))


def get_all_datasets() -> List[str]:
    cfg = load_config("datasets")
    return sorted(list(cfg.keys()))


def setup_parser() -> argparse.ArgumentParser:
    all_models = get_all_models()
    all_datasets = get_all_datasets()

    # Check for context-sensitive help triggers in raw argv
    show_all = "--all" in sys.argv
    selected_model = None
    for i, arg in enumerate(sys.argv):
        if arg in ["-m", "--model"] and i + 1 < len(sys.argv):
            selected_model = sys.argv[i + 1]
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
    train_parser.add_argument(
        "-m", "--model", type=str, default="transformer", choices=all_models
    )
    train_parser.add_argument(
        "-d", "--dataset", type=str, default="species", choices=all_datasets
    )
    train_parser.add_argument(
        "--benchmark", action="store_true", help="Run performance benchmarks"
    )
    train_parser.add_argument(
        "--figures", action="store_true", help="Generate analysis figures"
    )
    train_parser.add_argument(
        "--xai", action="store_true", help="Run explainability and biomarker discovery"
    )
    train_parser.add_argument(
        "--all", action="store_true", help="Show all expert hyperparameters in --help"
    )

    # CONTEXT SENSITIVE: Only show --encoder if model is contrastive or --all is used
    encoder_help = "Encoder type for contrastive learning (e.g., transformer, cnn)"
    if is_contrastive or show_all:
        train_parser.add_argument(
            "--encoder", type=str, default="dense", help=encoder_help
        )
    else:
        # Hide it from help but still allow it to be parsed if passed
        train_parser.add_argument(
            "--encoder", type=str, default="dense", help=argparse.SUPPRESS
        )

    # Expert Hyperparameters Group
    hp_group_label = "Hyperparameters" if show_all else "Common Hyperparameters"
    hp_group = train_parser.add_argument_group(hp_group_label)
    hp_group.add_argument("-e", "--epochs", type=int, help="Number of training epochs")
    hp_group.add_argument(
        "-N", "--num-runs", type=int, default=1, help="Number of independent runs"
    )
    hp_group.add_argument(
        "--normalize", action="store_true", help="Apply TIC/L2 normalization to the input spectra"
    )
    hp_group.add_argument(
        "-fp",
        "--file-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to spectral data file. If omitted, uses the internal package dataset.",
    )
    hp_group.add_argument(
        "--seed", type=int, dest="run", default=42, help="Random seed for the run"
    )
    hp_group.add_argument(
        "--wandb-log", action="store_true", help="Log results to Weights & Biases"
    )
    hp_group.add_argument(
        "--statistical",
        action="store_true",
        help="Perform statistical significance tests",
    )

    hp_group.add_argument(
        "-r", "--random-projection", action="store_true", help="Apply random projection to the input data (QJL-style)"
    )
    hp_group.add_argument(
        "-q", "--quantize", action="store_true", help="Apply QJL/TurboQuant quantization (sign-bit) after random projection"
    )
    hp_group.add_argument(
        "-T", "--turbo-quant", action="store_true", help="Apply Two-Stage TurboQuant (Residual-based QJL)"
    )
    hp_group.add_argument(
        "--polar", action="store_true", help="Apply polar quantization (unit-norm) after random projection"
    )
    hp_group.add_argument("--batch-size", type=int, default=64)
    hp_group.add_argument("--lr", "--learning-rate", type=float, default=1e-4)
    hp_group.add_argument("--hidden-dim", type=int, default=128)
    hp_group.add_argument("--num-layers", type=int, default=4)
    hp_group.add_argument("--num-heads", type=int, default=4)
    hp_group.add_argument("--num-kv-heads", type=int, default=2)
    hp_group.add_argument("--dropout", type=float, default=0.1)
    hp_group.add_argument(
        "--top-k", type=int, default=None, help="Process only the top-K peaks"
    )
    hp_group.add_argument(
        "--binding-type", type=str, default=None, choices=["hadamard", "outer_product", "complex", "multiplicative", "additive"]
    )
    hp_group.add_argument(
        "--use-performer", action="store_true", help="Use linear attention for speed"
    )
    hp_group.add_argument(
        "--use-checkpointing", action="store_true", help="Enable gradient checkpointing"
    )

    subparsers.add_parser("wizard", help="Interactive setup")

    subparsers.add_parser(
        "dashboard", help="Launch the interactive Streamlit dashboard"
    )

    download_parser = subparsers.add_parser(
        "download-data", help="Download private REIMS dataset"
    )
    download_parser.add_argument(
        "--token", type=str, help="GitHub Personal Access Token"
    )

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

    metrics = [
        ("Accuracy", "accuracy"),
        ("Balanced Accuracy", "balanced_accuracy"),
        ("MAE", "mae"),
        ("MSE", "mse"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1"),
    ]

    # Add Loss to the top if traditional metrics are zero (like in contrastive)
    has_meaningful_metrics = any(
        results.get(f"val_{k}", results.get(k, 0.0)) > 0 for _, k in metrics
    )
    if not has_meaningful_metrics and "val_loss" in results:
        metrics.insert(0, ("Contrastive Loss", "loss"))

    for label, key in metrics:
        tr = results.get(f"train_{key}", results.get(key, 0.0))
        val = results.get(f"val_{key}", results.get(key, 0.0))
        if isinstance(tr, (int, float)) and isinstance(val, (int, float)):
            table.add_row(label, f"{tr:.4f}", f"{val:.4f}")
    console.print("\n")
    console.print(Panel(table, expand=False, border_style="green"))
    if "total_training_time_s" in results:
        console.print(
            f"[dim italic]Elapsed training time: {results['total_training_time_s']:.4f} seconds[/]\n"
        )


def main() -> None:
    parser = setup_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    try:
        if args.command == "train":
            if args.config:
                import yaml

                with open(args.config, "r") as f:
                    config_data = yaml.safe_load(f)
                if isinstance(config_data, dict) and "models" in config_data:
                    from fishy._core.config import ExperimentConfig

                    exp_cfg = ExperimentConfig.from_yaml(args.config)
                    run_unified_training(exp_cfg)
                else:
                    config = TrainingConfig.from_yaml(args.config)
                    if not hasattr(config, "method") or not config.method:
                        config.method = detect_method(config.model)
                    _handle_train_execution(config)
            else:
                config = TrainingConfig.from_args(args)
                config.method = detect_method(args.model)
                _handle_train_execution(config)
        elif args.command == "wizard":
            from fishy.cli.wizard import run_wizard

            run_wizard()
        elif args.command == "dashboard":
            import subprocess
            from pathlib import Path
            import fishy

            # Locate the dashboard relative to the package or current directory
            dashboard_path = Path(fishy.__file__).parent.parent / "dashboard" / "app.py"

            if not dashboard_path.exists():
                # Fallback for local development if installed in editable mode differently
                dashboard_path = Path("dashboard/app.py")

            if not dashboard_path.exists():
                console.print("[bold red]Error:[/] Could not locate dashboard/app.py")
                sys.exit(1)

            console.print(f"[bold green]Launching dashboard...[/]")
            try:
                subprocess.run(["streamlit", "run", str(dashboard_path)])
            except KeyboardInterrupt:
                console.print("\n[bold blue]Dashboard stopped.[/]")
        elif args.command == "download-data":
            from fishy._core.data_manager import download_dataset

            success = download_dataset(token=args.token)
            if not success:
                sys.exit(1)
        elif args.command == "run_all":
            from fishy.experiments.unified_trainer import run_all_benchmarks

            run_all_benchmarks(quick=args.quick, num_runs=args.num_runs)
    except Exception:
        console.print_exception(show_locals=True)
        sys.exit(1)


def _handle_train_execution(config: TrainingConfig):
    n_runs = (
        config.num_runs if config.num_runs > 1 else (5 if config.statistical else 1)
    )
    results = []
    import time

    start_all = time.time()

    # Only use console.status for multi-run or statistical tasks to avoid LiveError with inner progress bars
    if n_runs > 1 or config.statistical:
        status_manager = console.status(f"[bold green]Running {n_runs} experiments...")
        status_manager.start()
        # Store on console so we can stop it later in unified_trainer
        console._status = status_manager
        try:
            for i in range(n_runs):
                try:
                    seed = (i + 1) * 123
                    config.run = seed
                    set_seed(seed)
                    status_manager.update(f"[bold green]Experiment {i+1}/{n_runs}...")
                    res = run_unified_training(config)
                    # Strip memory-intensive objects
                    res.pop("model", None)
                    res.pop("data_module", None)
                    results.append(res)
                except Exception as e:
                    console.print(f"[bold red]Experiment {i+1} failed:[/] {e}")

                # Forced cleanup between runs
                import gc, torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        finally:
            status_manager.stop()
            console._status = None
    else:
        # Single run: no status here, let the inner trainer handle progress
        seed = config.run if config.run != 0 else 123
        config.run = seed
        set_seed(seed)
        res = run_unified_training(config)

        if config.xai:
            model = res.get("model")
            dm = res.get("data_module")
            if model and dm:
                feature_names = dm.get_train_dataframe().columns[1:]
                console.print(
                    "\n[bold yellow]Starting Automated Biomarker Discovery...[/]"
                )
                report = run_biomarker_pipeline(
                    model=model,
                    data_loader=dm.get_train_dataloader(),
                    feature_names=feature_names,
                    device=get_device(),
                )
                console.print(
                    Panel(
                        report,
                        title="[bold]XAI Pipeline Results[/]",
                        border_style="yellow",
                    )
                )

        # Strip memory-intensive objects
        res.pop("model", None)
        res.pop("data_module", None)
        results.append(res)

        # Forced cleanup
        import gc, torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    final_res = {}
    if results:
        for k in results[0].keys():
            vals = [r[k] for r in results if k in r and isinstance(r[k], (int, float))]
            if vals:
                final_res[k] = sum(vals) / len(vals)

    final_res["total_training_time_s"] = time.time() - start_all
    if not config.statistical:
        display_final_summary(final_res)
    else:
        # Note: We need a status if we want to update it, but let's be safe.
        baseline_results = []
        if config.model != "opls-da":
            for i in range(n_runs):
                seed = (i + 1) * 123
                set_seed(seed)
                b_cfg = TrainingConfig(
                    model="opls-da",
                    dataset=config.dataset,
                    run=seed,
                    method="classic",
                    file_path=config.file_path,
                )
                baseline_results.append(run_unified_training(b_cfg))

        res_map = {f"{config.dataset}|||{config.model}": results}
        if baseline_results:
            res_map[f"{config.dataset}|||opls-da"] = baseline_results
        summary_df = summarize_results(res_map)
        display_statistical_summary(
            summary_df, show_significance=config.model != "opls-da"
        )


if __name__ == "__main__":
    main()
