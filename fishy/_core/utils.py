# -*- coding: utf-8 -*-
"""
Utility module for unified logging, result management, and directory structure.
"""

import logging
import json
import time
import os
from pathlib import Path
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional
import pandas as pd

class RunContext:
    """
    Manages the lifecycle of an experiment run, including directory creation,
    unified logging, and result persistence.
    """
    def __init__(self, experiment_name: str, run_id: int = 0, base_output_dir: str = "outputs"):
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.experiment_name = experiment_name
        self.run_id = run_id
        
        # Structured output directory
        self.run_dir = Path(base_output_dir) / experiment_name / f"run_{run_id}_{self.timestamp}"
        self.log_dir = self.run_dir / "logs"
        self.result_dir = self.run_dir / "results"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.figure_dir = self.run_dir / "figures"
        
        self._create_dirs()
        self.logger = self._setup_logging()
        self.logger.info(f"Initialized RunContext for experiment: {experiment_name}, run: {run_id}")
        self.logger.info(f"Output directory: {self.run_dir}")

    def _create_dirs(self):
        """Creates the structured directory tree for the run."""
        for d in [self.log_dir, self.result_dir, self.checkpoint_dir, self.figure_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> logging.Logger:
        """Sets up a unified logger that outputs to both console and a log file."""
        logger = logging.getLogger(f"fishy.{self.experiment_name}.run_{self.run_id}")
        logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers if RunContext is re-initialized in the same process
        if logger.handlers:
            return logger

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # File handler
        log_file = self.log_dir / "experiment.log"
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        return logger

    def save_results(self, results: Dict[str, Any], filename: str = "metrics.json"):
        """Saves a dictionary of results/metrics to a JSON file."""
        path = self.result_dir / filename
        with open(path, 'w') as f:
            json.dump(results, f, indent=4)
        self.logger.info(f"Metrics saved to {path}")

    def save_config(self, config: Any, filename: str = "config.json"):
        """Saves the experiment configuration to a JSON file."""
        path = self.run_dir / filename
        
        if is_dataclass(config):
            config_dict = asdict(config)
        elif hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        elif isinstance(config, dict):
            config_dict = config
        else:
            config_dict = {"config_summary": str(config)}
            
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        self.logger.info(f"Configuration saved to {path}")

    def save_dataframe(self, df: pd.DataFrame, filename: str):
        """Saves a pandas DataFrame to the results directory."""
        path = self.result_dir / filename
        if filename.endswith(".csv"):
            df.to_csv(path, index=False)
        elif filename.endswith(".json"):
            df.to_json(path, indent=4)
        else:
            df.to_pickle(path)
        self.logger.info(f"DataFrame saved to {path}")

    def save_figure(self, fig: Any, filename: str):
        """Saves a matplotlib figure to the figures directory."""
        path = self.figure_dir / filename
        # Basic check for matplotlib figure
        if hasattr(fig, 'savefig'):
            fig.savefig(path)
        else:
            # Assume it might be a seaborn/plt object or we use plt.savefig if fig is None
            import matplotlib.pyplot as plt
            plt.savefig(path)
        self.logger.info(f"Figure saved to {path}")

    def get_checkpoint_path(self, filename: str) -> Path:
        """Returns a path within the checkpoint directory."""
        return self.checkpoint_dir / filename

    def log_metric(self, step: int, metrics: Dict[str, float]):
        """
        Logs metrics for a specific step. 
        In the future, this could also write to a PSQL database.
        """
        # For now, just log to info
        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step}: {metrics_str}")
        
        # Append to a csv for easy parsing later
        csv_path = self.result_dir / "step_metrics.csv"
        metrics_with_step = {"step": step, **metrics}
        df = pd.DataFrame([metrics_with_step])
        df.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
