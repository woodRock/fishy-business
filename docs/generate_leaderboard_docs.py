# -*- coding: utf-8 -*-
"""
Helper script to prepare leaderboard data for documentation.
Exports both summarized and raw results for advanced plotting.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add project root to path BEFORE anything else
root_dir = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(root_dir))

from fishy.analysis.statistical import summarize_results

def process_wandb_csv_standalone(file_path):
    print(f"Processing {file_path}...")
    df = pd.read_csv(file_path)
    
    col_map = {
        "val_balanced_accuracy": ["val_balanced_accuracy", "val_acc", "accuracy"],
        "train_balanced_accuracy": ["train_balanced_accuracy", "train_acc"],
        "dataset": ["dataset", "Dataset"],
        "model": ["model", "Method"]
    }
    
    def normalize(curr_df):
        for standard, alternates in col_map.items():
            if standard not in curr_df.columns:
                for alt in alternates:
                    if alt in curr_df.columns:
                        curr_df[standard] = curr_df[alt]; break
        return curr_df

    df = normalize(df)
    
    if not all(c in df.columns for c in ["dataset", "model", "val_balanced_accuracy"]):
        if "summary" in df.columns and "config" in df.columns:
            import ast
            def safe_parse(val):
                if pd.isna(val): return {}
                try: 
                    if isinstance(val, dict): return val
                    return ast.literal_eval(val)
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
            df = normalize(df)

    for col in ["val_balanced_accuracy", "train_balanced_accuracy"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    
    results_map = {}
    raw_rows = []
    for (ds, model), group in df.groupby(["dataset", "model"]):
        if pd.isna(ds) or pd.isna(model) or ds == "dataset": continue
        results_map[f"{ds}|||{model}"] = group.to_dict(orient="records")
        for _, r in group.iterrows():
            raw_rows.append({
                "Dataset": ds, "Method": model, 
                "Test Accuracy": r["val_balanced_accuracy"], 
                "Train Accuracy": r["train_balanced_accuracy"]
            })
    
    summary = summarize_results(results_map, baseline_model="opls-da")
    return summary, pd.DataFrame(raw_rows)

def generate_doc_data():
    output_dir = root_dir / "docs" / "source" / "_static"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_sum, df_raw = pd.DataFrame(), pd.DataFrame()
    wb_csv = root_dir / "wanb_export_csv.csv"
    if wb_csv.exists():
        df_sum, df_raw = process_wandb_csv_standalone(str(wb_csv))
    
    if not df_sum.empty:
        df_sum.to_csv(output_dir / "leaderboard_data.csv", index=False)
        df_raw.to_csv(output_dir / "leaderboard_raw.csv", index=False)
        print(f"✅ Data saved: {len(df_sum)} summaries, {len(df_raw)} raw runs.")

if __name__ == "__main__":
    generate_doc_data()
