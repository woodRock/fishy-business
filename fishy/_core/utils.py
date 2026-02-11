# -*- coding: utf-8 -*-
"""
Utility module for unified logging, result management, and directory structure.
"""

import logging
import json
import time
import os
import random
from pathlib import Path
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional, Union
import pandas as pd
import numpy as np
import torch
import wandb
import wandb.sdk.wandb_run

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Centralized Rich Console
custom_theme = Theme(
    {
        "info": "dim cyan",
        "warning": "magenta",
        "error": "bold red",
        "success": "bold green",
    }
)
console = Console(theme=custom_theme)


def set_seed(seed: int):
    """Sets the seed for reproducibility across multiple libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Returns the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types and Path objects."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


class RunContext:
    """
    Manages experiment lifecycle, logging, and results.
    Redirects detailed logs to a hidden /logs directory.
    """

    def __init__(
        self,
        dataset: str,
        method: str,
        model_name: str,
        base_output_dir: str = "outputs",
        wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
    ):
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.dataset = dataset
        self.method = method
        self.model_name = model_name
        self.wandb_run = wandb_run

        # Local experiment dir
        self.run_dir = (
            Path(base_output_dir) / dataset / method / f"{model_name}_{self.timestamp}"
        )

        # Global hidden logs dir
        self.global_log_dir = Path("logs")
        self.global_log_dir.mkdir(exist_ok=True)

        self.log_dir = self.run_dir / "logs"
        self.result_dir = self.run_dir / "results"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.figure_dir = self.run_dir / "figures"
        self.benchmark_dir = self.run_dir / "benchmark"

        self._create_dirs()
        self.logger = self._setup_logging()

        self.logger.info(
            f"Initialized RunContext: [bold]{model_name}[/] on [bold]{dataset}[/]"
        )

    def _create_dirs(self):
        for d in [
            self.log_dir,
            self.result_dir,
            self.checkpoint_dir,
            self.figure_dir,
            self.benchmark_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(f"fishy.{self.dataset}.{self.model_name}")
        logger.setLevel(logging.INFO)

        if logger.handlers:
            return logger

        # 1. Rich Console Handler (Clean, beautiful output)
        rich_handler = RichHandler(
            console=console,
            show_time=False,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        rich_handler.setLevel(logging.INFO)
        logger.addHandler(rich_handler)

        # 2. File Handler (Detailed logs in hidden folder)
        log_file = (
            self.global_log_dir
            / f"{self.dataset}_{self.model_name}_{self.timestamp}.log"
        )
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        return logger

    def save_results(self, results: Dict[str, Any], filename: str = "metrics.json"):
        path = self.result_dir / filename
        with open(path, "w") as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder)

        if self.wandb_run:
            log_dict = {}
            for k, v in results.items():
                if k == "stats" and isinstance(v, dict):
                    log_dict.update(v)
                elif isinstance(v, (int, float, str, bool, np.integer, np.floating)):
                    log_dict[k] = v
            if log_dict:
                log_dict = json.loads(json.dumps(log_dict, cls=NumpyEncoder))
                self.wandb_run.log(log_dict, commit=False)
            self.wandb_run.save(str(path), base_path=str(self.run_dir))

    def save_config(self, config: Any, filename: str = "config.json"):
        path = self.run_dir / filename
        config_dict = (
            asdict(config)
            if is_dataclass(config)
            else (
                config.to_dict()
                if hasattr(config, "to_dict")
                else (config if isinstance(config, dict) else {"summary": str(config)})
            )
        )
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=4, cls=NumpyEncoder)
        if self.wandb_run:
            self.wandb_run.config.update(
                json.loads(json.dumps(config_dict, cls=NumpyEncoder))
            )
            self.wandb_run.save(str(path), base_path=str(self.run_dir))

    def save_dataframe(self, df: pd.DataFrame, filename: str):
        path = self.result_dir / filename
        if filename.endswith(".csv"):
            df.to_csv(path, index=False)
        elif filename.endswith(".json"):
            df.to_json(path, indent=4)
        else:
            df.to_pickle(path)
        if self.wandb_run:
            self.wandb_run.save(str(path), base_path=str(self.run_dir))

    def save_figure(self, fig: Any, filename: str):
        path = self.figure_dir / filename
        if hasattr(fig, "savefig"):
            fig.savefig(path)
        else:
            import matplotlib.pyplot as plt

            plt.savefig(path)
        if self.wandb_run:
            self.wandb_run.log(
                {f"figure/{filename}": wandb.Image(str(path))}, commit=False
            )
            self.wandb_run.save(str(path), base_path=str(self.run_dir))

    def get_checkpoint_path(self, filename: str) -> Path:
        return self.checkpoint_dir / filename

    def log_metric(self, step: int, metrics: Dict[str, float]):
        if self.wandb_run:
            self.wandb_run.log(metrics, step=step)

    def log_summary_charts(
        self, y_true: np.ndarray, y_probs: np.ndarray, class_names: list
    ):
        if self.wandb_run:
            self.wandb_run.log(
                {
                    "conf_mat": wandb.plot.confusion_matrix(
                        probs=y_probs, y_true=y_true, class_names=class_names
                    )
                },
                commit=False,
            )
            self.wandb_run.log(
                {"roc": wandb.plot.roc_curve(y_true, y_probs, labels=class_names)},
                commit=False,
            )

    def log_prediction_table(
        self, spectra, preds, targets, probs, class_names, table_name="predictions"
    ):
        if self.wandb_run:
            import matplotlib.pyplot as plt

            columns = [
                "id",
                "spectrum",
                "prediction",
                "target",
                "confidence",
                "is_correct",
            ]
            table = wandb.Table(columns=columns)
            for i in range(min(len(spectra), 100)):
                plt.figure(figsize=(4, 3))
                plt.plot(spectra[i])
                plt.close()
                table.add_data(
                    i,
                    wandb.Image(plt),
                    class_names[int(preds[i])],
                    class_names[int(targets[i])],
                    float(probs[i].max()),
                    bool(preds[i] == targets[i]),
                )
            self.wandb_run.log({table_name: table}, commit=False)
