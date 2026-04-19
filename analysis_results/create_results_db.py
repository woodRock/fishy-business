# -*- coding: utf-8 -*-
"""
Database creation and statistical analysis for experiment results.
Populates results.db from CSV files and provides leaderboard views.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Make paths relative to the script location
SCRIPT_DIR = Path(__file__).parent.absolute()
DB_PATH = SCRIPT_DIR / "results.db"
ANALYSIS_DIR = SCRIPT_DIR

class ResultsDatabase:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        # Primary table for all individual runs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset TEXT,
                model_base TEXT,
                model_descriptive TEXT,
                train_acc REAL,
                test_acc REAL,
                runtime REAL,
                use_ema BOOLEAN,
                use_muon BOOLEAN,
                use_xsa BOOLEAN,
                use_mla BOOLEAN,
                source_file TEXT
            )
        """)
        self.conn.commit()

    def clear_database(self):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM runs")
        self.conn.commit()

    def add_run(self, data: Dict[str, Any]):
        keys = data.keys()
        columns = ", ".join(keys)
        placeholders = ", ".join(["?" for _ in keys])
        values = tuple(data.values())
        
        cursor = self.conn.cursor()
        cursor.execute(f"INSERT INTO runs ({columns}) VALUES ({placeholders})", values)
        self.conn.commit()

    def get_descriptive_name(self, row: pd.Series) -> str:
        """Constructs a reasonable name like augformer-ema-muon."""
        # Check both "model" and "Method" (common in different CSV formats)
        base = row.get("model", row.get("Method", "unknown"))
        if pd.isna(base):
            base = "unknown"
        base = str(base).lower()
        parts = [base]
        
        def is_true(val):
            if val is None or pd.isna(val) or val == "": return False
            if isinstance(val, bool): return val
            if isinstance(val, (int, float)): return val == 1
            if isinstance(val, str):
                return val.lower() in ["true", "1", "yes", "t"]
            return False

        # Check for various flags
        if is_true(row.get("use_ema")):
            if "ema" not in base: parts.append("ema")
            
        if is_true(row.get("use_muon")):
            parts.append("muon")
            
        if is_true(row.get("use_xsa")):
            parts.append("xsa")
            
        if is_true(row.get("use_mla")):
            parts.append("mla")
            
        if is_true(row.get("use_ttt")):
            parts.append("ttt")
            
        # Special case for "warmup" if we can detect it in the source file name 
        if "warmup" in str(row.get("Name", "")).lower() or "warmup" in str(row.get("source_file", "")).lower():
            parts.append("warmup")
            
        return "-".join(parts)

    def import_csv(self, file_path: Path):
        logger.info(f"Importing {file_path.name}...")
        df = pd.read_csv(file_path)
        
        # Normalize column names (strip quotes if present)
        df.columns = [c.strip('"') for c in df.columns]
        
        count = 0
        skipped = 0
        # Mapping from various CSV formats to our internal schema
        for _, row in df.iterrows():
            # Add source file to row for name generation logic
            row["source_file"] = file_path.name
            
            # Check for missing accuracy - skip rows that have no result
            test_acc_raw = row.get("val_balanced_accuracy", row.get("Test Accuracy"))
            if pd.isna(test_acc_raw) or test_acc_raw == "" or str(test_acc_raw).strip() == "":
                skipped += 1
                continue

            run_data = {
                "dataset": row.get("dataset", row.get("Dataset")),
                "model_base": row.get("model", row.get("Method")),
                "train_acc": row.get("train_balanced_accuracy", row.get("Train Accuracy", 0.0)),
                "test_acc": test_acc_raw,
                "runtime": row.get("Runtime", row.get("runtime", 0.0)),
                "use_ema": row.get("use_ema", False),
                "use_muon": row.get("use_muon", False),
                "use_xsa": row.get("use_xsa", False),
                "use_mla": row.get("use_mla", False),
                "source_file": file_path.name
            }
            
            # Use Descriptive Name
            run_data["model_descriptive"] = self.get_descriptive_name(row)
            
            # Clean up types
            valid = True
            for k in ["train_acc", "test_acc", "runtime"]:
                try:
                    val = run_data[k]
                    if pd.isna(val) or val == "":
                        run_data[k] = 0.0
                    else:
                        run_data[k] = float(val)
                except:
                    run_data[k] = 0.0
            
            self.add_run(run_data)
            count += 1
        
        msg = f"Successfully imported {count} runs from {file_path.name}"
        if skipped > 0:
            msg += f" (skipped {skipped} incomplete runs)"
        logger.info(msg)

    def generate_leaderboard(self):
        """Generates a leaderboard per dataset with statistical significance."""
        df = pd.read_sql_query("SELECT * FROM runs", self.conn)
        if df.empty:
            print("No data in database.")
            return

        datasets = df["dataset"].unique()
        
        for ds in datasets:
            ds_df = df[df["dataset"] == ds]
            print(f"\n{'='*80}")
            print(f" LEADERBOARD: {ds.upper()}")
            print(f"{'='*80}")
            
            # Group by descriptive model name
            summary = ds_df.groupby("model_descriptive").agg({
                "test_acc": ["mean", "std", "count"],
                "train_acc": ["mean"],
                "runtime": ["mean"]
            }).reset_index()
            
            # Flatten multi-index columns
            summary.columns = ["Model", "Test Mean", "Test Std", "N", "Train Mean", "Time"]
            summary = summary.sort_values("Test Mean", ascending=False)
            
            # Identify winner
            winner_model = summary.iloc[0]["Model"]
            winner_scores = ds_df[ds_df["model_descriptive"] == winner_model]["test_acc"].values
            
            print(f"{'Model':<30} | {'Test Acc':<10} | {'Std':<8} | {'N':<5} | {'Sig':<5}")
            print(f"{'-'*30}-|-{'-'*10}-|-{'-'*8}-|-{'-'*5}-|-{'-'*5}")
            
            for _, row in summary.iterrows():
                model = row["Model"]
                mean_score = row["Test Mean"]
                std_score = row["Test Std"]
                n = int(row["N"])
                
                sig_symbol = " "
                if model == winner_model:
                    sig_symbol = "★"
                elif n > 1 and len(winner_scores) > 1:
                    model_scores = ds_df[ds_df["model_descriptive"] == model]["test_acc"].values
                    # Welch's t-test
                    t_stat, p_val = stats.ttest_ind(winner_scores, model_scores, equal_var=False)
                    if p_val < 0.05:
                        sig_symbol = "+" if mean_score > np.mean(winner_scores) else "-"
                    else:
                        sig_symbol = "≈"
                
                print(f"{model:<30} | {mean_score:.4f}   | {std_score:.4f} | {n:<5} | {sig_symbol:<5}")

def main():
    db = ResultsDatabase()
    db.clear_database()
    
    # Import all CSVs from analysis_results
    for csv_file in ANALYSIS_DIR.glob("*.csv"):
        db.import_csv(csv_file)
        
    # Also import the main leaderboard raw if it exists
    raw_leaderboard = SCRIPT_DIR.parent / "docs" / "source" / "_static" / "leaderboard_raw.csv"
    if raw_leaderboard.exists():
        db.import_csv(raw_leaderboard)
    
    # Print model counts
    df = pd.read_sql_query("SELECT model_descriptive, count(*) as count FROM runs GROUP BY model_descriptive", db.conn)
    print("\nMODEL COUNTS:")
    print(df.to_string(index=False))
        
    db.generate_leaderboard()

if __name__ == "__main__":
    main()
