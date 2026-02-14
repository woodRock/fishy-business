# -*- coding: utf-8 -*-
"""
Helper script to prepare leaderboard data for documentation.
This script can be run during doc builds to ensure the docs have the latest data.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# We need to reach into app.py without executing the whole file's UI code
# A cleaner way is to use the underlying statistical tools directly
from fishy.analysis.statistical import summarize_results

def crawl_local_results_standalone():
    import json
    results = []
    output_root = Path("outputs")
    if not output_root.exists():
        return pd.DataFrame()
    for metric_file in output_root.glob("**/results/metrics.json"):
        try:
            with open(metric_file, "r") as f:
                data = json.load(f)
            summary_file = metric_file.parent.parent / "statistical_analysis.csv"
            if summary_file.exists():
                sdf = pd.read_csv(summary_file)
                results.append(sdf)
            else:
                parts = metric_file.parts
                results.append(pd.DataFrame([{
                    "Dataset": parts[1],
                    "Method": parts[3].split("_")[0],
                    "Test": data.get("val_balanced_accuracy", 0),
                    "Train": data.get("train_balanced_accuracy", 0),
                }]))
        except: continue
    return pd.concat(results).drop_duplicates() if results else pd.DataFrame()

def process_wandb_csv_standalone(file_path):
    df = pd.read_csv(file_path)
    # Handle "nested" W&B API script format
    if "summary" in df.columns and "config" in df.columns:
        import ast
        def safe_parse(val):
            if pd.isna(val): return {}
            try: return ast.literal_eval(val)
            except: return {}
        flattened_data = []
        for _, row in df.iterrows():
            s, c = safe_parse(row["summary"]), safe_parse(row["config"])
            flattened_data.append({
                "dataset": c.get("dataset"),
                "model": c.get("model"),
                "train_balanced_accuracy": s.get("train_balanced_accuracy", s.get("train_accuracy", 0)),
                "val_balanced_accuracy": s.get("val_balanced_accuracy", s.get("accuracy", 0))
            })
        df = pd.DataFrame(flattened_data)
    
    # Normalize
    col_map = {
        "val_balanced_accuracy": ["val_balanced_accuracy", "val_acc", "accuracy"],
        "train_balanced_accuracy": ["train_balanced_accuracy", "train_acc"],
        "dataset": ["dataset", "Dataset"],
        "model": ["model", "Method"]
    }
    for standard, alternates in col_map.items():
        if standard not in df.columns:
            for alt in alternates:
                if alt in df.columns:
                    df[standard] = df[alt]; break
                    
    results_map = {}
    for (ds, model), group in df.groupby(["dataset", "model"]):
        if pd.isna(ds) or pd.isna(model): continue
        results_map[f"{ds}|||{model}"] = group.to_dict(orient="records")
    
    return summarize_results(results_map, baseline_model="opls-da")

def generate_doc_data():
    output_dir = Path(__file__).parent / "source" / "_static"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame()
    wb_csv = Path("wanb_export_csv.csv")
    if wb_csv.exists():
        print(f"Reading from {wb_csv}...")
        df = process_wandb_csv_standalone(str(wb_csv))
    
    if df.empty:
        print("Reading local results...")
        df = crawl_local_results_standalone()
    
    if not df.empty:
        df.to_csv(output_dir / "leaderboard_data.csv", index=False)
        print(f"✅ Leaderboard data saved to {output_dir / 'leaderboard_data.csv'}")

if __name__ == "__main__":
    generate_doc_data()
