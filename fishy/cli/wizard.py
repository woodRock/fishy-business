# -*- coding: utf-8 -*-
"""
Interactive wizard for setting up experiments.
"""

import os
import yaml
from typing import Dict, Any, List
from pathlib import Path
from fishy._core.config_loader import load_config
from fishy._core.config import TrainingConfig

def ask_choice(question: str, options: List[str], default: str = None) -> str:
    print(f"\n{question}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    
    while True:
        choice = input(f"Select option (default {default}): ").strip()
        if not choice and default:
            return default
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            if choice in options:
                return choice
        print("Invalid selection. Try again.")

def ask_bool(question: str, default: bool = False) -> bool:
    d_str = "Y/n" if default else "y/N"
    choice = input(f"\n{question} [{d_str}]: ").strip().lower()
    if not choice:
        return default
    return choice == "y"

def run_wizard():
    print("="*40)
    print(" FISHY BUSINESS EXPERIMENT WIZARD ")
    print("="*40)

    models_cfg = load_config("models")
    datasets_cfg = load_config("datasets")

    # 1. Select Model
    sections = {
        "Deep Learning": "deep_models",
        "Classic ML": "classic_models",
        "Evolutionary": "evolutionary_models",
        "Contrastive": "contrastive_models",
        "Probabilistic / Bayesian": "probabilistic_models"
    }
    section_name = ask_choice("Select Model Category:", list(sections.keys()), "Deep Learning")
    section_key = sections[section_name]
    
    available_models = sorted(list(models_cfg[section_key].keys()))
    model = ask_choice(f"Select {section_name} Model:", available_models, available_models[0])

    # 2. Select Dataset
    available_datasets = sorted(list(datasets_cfg.keys()))
    dataset = ask_choice("Select Dataset:", available_datasets, "species")

    # 3. Analysis Flags
    benchmark = ask_bool("Enable performance benchmarking?")
    figures = ask_bool("Generate analysis figures?")
    wandb_log = ask_bool("Log to Weights & Biases?")

    # 4. Advanced Options
    is_transfer = False
    is_ordinal = False
    is_regression = False
    if section_key == "deep_models":
        is_transfer = ask_bool("Enable Sequential Transfer Learning?")
        is_ordinal = ask_bool("Enable Ordinal Regression?")
    elif section_key in ["classic_models", "probabilistic_models"]:
        is_regression = ask_bool("Enable Regression Mode?")

    # 5. Summary and Output
    print("\n" + "-"*20)
    output_type = ask_choice("How would you like to save this setup?", ["CLI Command", "YAML Config File"], "CLI Command")

    config = TrainingConfig(
        model=model,
        dataset=dataset,
        benchmark=benchmark,
        figures=figures,
        wandb_log=wandb_log,
        transfer=is_transfer,
        regression=is_regression,
    )

    if output_type == "CLI Command":
        cmd = f"python3 main.py train -m {model} -d {dataset}"
        if benchmark: cmd += " --benchmark"
        if figures: cmd += " --figures"
        if wandb_log: cmd += " --wandb-log"
        if is_transfer: cmd += " --transfer"
        if is_ordinal: cmd += " --ordinal"
        if is_regression: cmd += " --regression"
        
        print("\nGenerated Command:")
        print(f"\033[92m{cmd}\033[0m\n")
    else:
        filename = input("\nEnter config filename (default: experiment.yaml): ").strip() or "experiment.yaml"
        config.to_yaml(filename)
        print(f"\nConfiguration saved to \033[92m{filename}\033[0m")
        print(f"Run it using: python3 main.py train -c {filename}\n")

if __name__ == "__main__":
    run_wizard()
