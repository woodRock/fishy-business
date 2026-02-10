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
import wandb
import wandb.sdk.wandb_run


class RunContext:
    """
    Manages the lifecycle of an experiment run, including directory creation,
    unified logging, and result persistence.
    """
    def __init__(self, dataset: str, method: str, model_name: str, base_output_dir: str = "outputs", wandb_run: Optional[wandb.sdk.wandb_run.Run] = None):
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.dataset = dataset  # Store for logger and other uses
        self.method = method
        self.model_name = model_name
        self.wandb_run = wandb_run # Store the wandb run object
        
        # New structured output directory: outputs/{dataset}/{method}/{model_name}_{timestamp}/
        self.run_dir = Path(base_output_dir) / dataset / method / f"{model_name}_{self.timestamp}"
        
        self.log_dir = self.run_dir / "logs"
        self.result_dir = self.run_dir / "results"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.figure_dir = self.run_dir / "figures"

        self._create_dirs()
        self.logger = self._setup_logging() # _setup_logging will now use self.dataset etc.
        self.logger.info(f"Initialized RunContext for dataset: {dataset}, method: {method}, model: {model_name}")
        self.logger.info(f"Output directory: {self.run_dir}")

    def _create_dirs(self):
        """Creates the structured directory tree for the run."""
        for d in [self.log_dir, self.result_dir, self.checkpoint_dir, self.figure_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> logging.Logger:
        """Sets up a unified logger that outputs to both console and a log file."""
        logger = logging.getLogger(f"fishy.{self.dataset}.{self.method}.{self.model_name}")
        logger.setLevel(logging.INFO)

        # Prevent duplicate handlers if RunContext is re-initialized in the same process
        if logger.handlers:
            return logger

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

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
        with open(path, "w") as f:
            json.dump(results, f, indent=4)
        self.logger.info(f"Metrics saved to {path}")
        if self.wandb_run:
            self.wandb_run.log(results, commit=False) # Log metrics to W&B
            self.wandb_run.save(str(path), base_path=str(self.run_dir)) # Log file as artifact

    def save_config(self, config: Any, filename: str = "config.json"):
        """Saves the experiment configuration to a JSON file."""
        path = self.run_dir / filename

        if is_dataclass(config):
            config_dict = asdict(config)
        elif hasattr(config, "to_dict"):
            config_dict = config.to_dict()
        elif isinstance(config, dict):
            config_dict = config
        else:
            config_dict = {"config_summary": str(config)}
            
        
        # Convert Path objects to strings for JSON serialization
        def convert_paths_to_str(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: convert_paths_to_str(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_paths_to_str(elem) for elem in obj]
            return obj

        config_dict = convert_paths_to_str(config_dict)
            
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        self.logger.info(f"Configuration saved to {path}")
        if self.wandb_run:
            self.wandb_run.config.update(config_dict) # Update W&B run config
            self.wandb_run.save(str(path), base_path=str(self.run_dir)) # Log file as artifact

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
        if self.wandb_run:
            self.wandb_run.save(str(path), base_path=str(self.run_dir)) # Log file as artifact

    def save_figure(self, fig: Any, filename: str):
        """Saves a matplotlib figure to the figures directory."""
        path = self.figure_dir / filename
        # Basic check for matplotlib figure
        if hasattr(fig, "savefig"):
            fig.savefig(path)
        else:
            # Assume it might be a seaborn/plt object or we use plt.savefig if fig is None
            import matplotlib.pyplot as plt

            plt.savefig(path)
        self.logger.info(f"Figure saved to {path}")
        if self.wandb_run:
            # wandb.Image expects a path or PIL image
            self.wandb_run.log({f"figure/{filename}": wandb.Image(str(path))}, commit=False)
            self.wandb_run.save(str(path), base_path=str(self.run_dir)) # Log file as artifact

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
        if self.wandb_run:
            self.wandb_run.log(metrics, step=step) # Log metrics to W&B, committing a step

    def log_summary_charts(self, y_true: np.ndarray, y_probs: np.ndarray, class_names: list):
        """Logs advanced metrics like Confusion Matrix and ROC curves to W&B."""
        if self.wandb_run:
            self.logger.info("Logging summary charts to W&B...")
            # 1. Confusion Matrix
            self.wandb_run.log({
                "conf_mat": wandb.plot.confusion_matrix(
                    probs=y_probs,
                    y_true=y_true,
                    class_names=class_names
                )
            }, commit=False)
            
            # 2. ROC Curve
            self.wandb_run.log({
                "roc": wandb.plot.roc_curve(y_true, y_probs, labels=class_names)
            }, commit=False)
            
            # 3. Precision-Recall Curve
            self.wandb_run.log({
                "pr": wandb.plot.pr_curve(y_true, y_probs, labels=class_names)
            }, commit=False)

    def log_prediction_table(self, spectra: np.ndarray, preds: np.ndarray, targets: np.ndarray, probs: np.ndarray, class_names: list, table_name: str = "predictions"):
        """Logs a table of predictions with their corresponding spectral plots."""
        if self.wandb_run:
            self.logger.info(f"Logging {table_name} table to W&B...")
            import matplotlib.pyplot as plt
            columns = ["id", "spectrum", "prediction", "target", "confidence", "is_correct"]
            table = wandb.Table(columns=columns)
            
            # Log a subset to avoid excessive data usage, but enough for meaningful inspection
            num_samples = min(len(spectra), 100)
            for i in range(num_samples):
                # Create a small plot for the spectrum
                plt.figure(figsize=(4, 3))
                plt.plot(spectra[i])
                plt.title(f"Target: {class_names[targets[i]]}")
                plt.xlabel("Wavelength/Feature")
                plt.ylabel("Intensity")
                
                # Use a buffer to avoid saving many small files locally
                img = wandb.Image(plt)
                plt.close()
                
                table.add_data(
                    i, 
                    img, 
                    class_names[preds[i]], 
                    class_names[targets[i]], 
                    float(probs[i].max()),
                    bool(preds[i] == targets[i])
                )
                
            self.wandb_run.log({table_name: table}, commit=False)

    
