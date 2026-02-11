# -*- coding: utf-8 -*-
"""
Interactive wizard for setting up experiments with consistent numbered choices.
"""

from typing import List, Dict, Any
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.text import Text

from fishy._core.config_loader import load_config
from fishy._core.config import TrainingConfig
from fishy._core.utils import console

def ask_numbered_choice(title: str, options: List[str], default_idx: int = 0) -> str:
    """Helper to present a list of options as a numbered table and return the selection."""
    table = Table(title=title, box=None, show_header=False)
    for i, opt in enumerate(options, 1):
        table.add_row(f"[bold cyan]{i}[/]", opt)
    console.print(table)
    
    choice_idx = Prompt.ask(
        f"Select an option (1-{len(options)})", 
        choices=[str(i) for i in range(1, len(options) + 1)], 
        default=str(default_idx + 1)
    )
    return options[int(choice_idx) - 1]

def run_wizard():
    console.clear()
    
    # 1. Welcome Header
    header_text = Text(" FISHY BUSINESS ", style="bold white on blue")
    header_text.append("\nExperiment Setup Wizard", style="italic cyan")
    console.print(Panel(header_text, expand=False, border_style="blue"))

    models_cfg = load_config("models")
    datasets_cfg = load_config("datasets")

    # 2. Select Model Category
    categories = {
        "Deep Learning": "deep_models",
        "Classic ML": "classic_models",
        "Evolutionary": "evolutionary_models",
        "Contrastive": "contrastive_models",
        "Probabilistic / Bayesian": "probabilistic_models",
    }
    cat_names = list(categories.keys())
    selected_cat_name = ask_numbered_choice("Model Categories", cat_names)
    section_key = categories[selected_cat_name]

    # 3. Select Specific Model
    available_models = sorted(list(models_cfg[section_key].keys()))
    model = ask_numbered_choice(f"{selected_cat_name} Models", available_models)

    # 4. Select Dataset
    available_datasets = sorted(list(datasets_cfg.keys()))
    dataset = ask_numbered_choice("Available Datasets", available_datasets)

    # 5. Analysis Flags
    console.print("\n[bold underline]Analysis & Logging Options[/]")
    benchmark = Confirm.ask("Enable performance benchmarking?", default=False)
    figures = Confirm.ask("Generate training/eval figures?", default=True)
    wandb_log = Confirm.ask("Log to Weights & Biases?", default=False)

    # 6. Advanced Options
    is_transfer = False
    is_ordinal = False
    is_regression = False
    
    if section_key == "deep_models":
        is_transfer = Confirm.ask("Enable Sequential Transfer Learning?", default=False)
        is_ordinal = Confirm.ask("Enable Ordinal Regression?", default=False)
    elif section_key in ["classic_models", "probabilistic_models"]:
        is_regression = Confirm.ask("Enable Regression Mode?", default=False)

    # 7. Summary and Output
    config = TrainingConfig(
        model=model, dataset=dataset, benchmark=benchmark,
        figures=figures, wandb_log=wandb_log, transfer=is_transfer,
        regression=is_regression,
    )

    summary_table = Table(title="[bold green]Configuration Summary[/]", box=None)
    summary_table.add_column("Property", style="cyan")
    summary_table.add_column("Value", style="magenta")
    summary_table.add_row("Model", model)
    summary_table.add_row("Dataset", dataset)
    summary_table.add_row("Benchmark", "Enabled" if benchmark else "Disabled")
    summary_table.add_row("W&B Log", "Enabled" if wandb_log else "Disabled")
    if is_transfer: summary_table.add_row("Mode", "Transfer Learning")
    if is_regression: summary_table.add_row("Mode", "Regression")
    
    console.print("\n", summary_table)

    output_options = ["CLI Command", "YAML Config File"]
    selected_output = ask_numbered_choice("How would you like to save this setup?", output_options)

    if selected_output == "CLI Command":
        cmd = f"python3 main.py train -m {model} -d {dataset}"
        if benchmark: cmd += " --benchmark"
        if figures: cmd += " --figures"
        if wandb_log: cmd += " --wandb-log"
        if is_transfer: cmd += " --transfer"
        if is_ordinal: cmd += " --ordinal"
        if is_regression: cmd += " --regression"
        
        console.print("\n")
        console.print(Panel(f"[bold green]{cmd}[/]", title="Generated Command", border_style="green"))
    else:
        filename = Prompt.ask("Enter config filename", default="experiment.yaml")
        config.to_yaml(filename)
        console.print(f"\n[bold green]✓[/] Configuration saved to [bold]{filename}[/]")
        console.print(f"Run it using: [cyan]python3 main.py train -c {filename}[/]\n")

if __name__ == "__main__":
    run_wizard()
