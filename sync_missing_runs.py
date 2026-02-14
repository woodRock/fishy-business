# -*- coding: utf-8 -*-
"""
Programmatically syncs missing runs from W&B to the local CSV.
Uses server-side filtering for efficiency.
"""

import pandas as pd
import wandb
import os
import ast
from pathlib import Path
from tqdm import tqdm

def sync_missing(entity="victoria-university-of-wellington", project="fishy-business", csv_path="wanb_export_csv.csv"):
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found.")
        return

    # 1. Load existing data
    df = pd.read_csv(csv_path)
    
    # We need to know which combinations are missing
    expected_datasets = ['species', 'part', 'oil', 'cross-species']
    expected_models = [
        'moe', 'ensemble', 'dt', 'rf', 'transformer', 'lr', 'xgb', 'kan', 
        'lda', 'opls-da', 'nb', 'svm', 'knn', 'cnn', 'lstm', 'rcnn'
    ]
    expected_runs = 30

    # Helper to get dataset/model from potentially nested rows
    def get_meta(row):
        if "config" in row and isinstance(row["config"], str):
            try:
                c = ast.literal_eval(row["config"])
                return c.get("dataset"), c.get("model")
            except: pass
        return row.get("dataset"), row.get("model")

    # Determine missing combinations
    missing_targets = []
    for ds in expected_datasets:
        for model in expected_models:
            # Simple filter check
            mask = (df['dataset'] == ds) & (df['model'] == model)
            # If the CSV is nested, the above might fail, so we use a more robust check if needed
            if mask.sum() == 0 and "config" in df.columns:
                # Fallback check for nested format
                count = 0
                for _, r in df.iterrows():
                    rds, rmod = get_meta(r)
                    if rds == ds and rmod == model:
                        count += 1
                current_count = count
            else:
                current_count = mask.sum()

            if current_count < expected_runs:
                missing_targets.append((ds, model, current_count))

    if not missing_targets:
        print("✅ All combinations have 30+ runs. Nothing to sync.")
        return

    print(f"Found {len(missing_targets)} combinations needing sync.")
    
    # 2. Connect to W&B
    api = wandb.Api()
    new_records = []
    
    print("Fetching missing runs from W&B...")
    for ds, model, current in tqdm(missing_targets):
        # Server-side filter for EXACT dataset and model
        filters = {
            "config.dataset": ds,
            "config.model": model,
            "state": "finished"
        }
        runs = api.runs(f"{entity}/{project}", filters=filters)
        
        count = 0
        for run in runs:
            # Format to match the script you shared
            new_records.append({
                "Name": run.name,
                "State": run.state,
                "Created": run.created_at,
                "Runtime": run.summary.get("_runtime", 0),
                "dataset": ds,
                "model": model,
                "train_balanced_accuracy": run.summary.get("train_balanced_accuracy", run.summary.get("train_accuracy", 0)),
                "val_balanced_accuracy": run.summary.get("val_balanced_accuracy", run.summary.get("accuracy", 0)),
                # If we want to keep it compatible with the 'nested' format the dashboard now supports:
                "summary": str(run.summary._json_dict),
                "config": str({k: v for k, v in run.config.items() if not k.startswith("_")})
            })
            count += 1
        
        if count > 0:
            print(f"  - {ds}/{model}: Found {count} runs on W&B (Local had {current})")

    if not new_records:
        print("❌ No new runs found on W&B for the missing combinations.")
        return

    # 3. Merge and Save
    new_df = pd.DataFrame(new_records)
    
    # Identify runs we already have by Name to avoid duplicates
    existing_names = set(df["Name"].unique())
    unique_new_df = new_df[~new_df["Name"].isin(existing_names)]
    
    if unique_new_df.empty:
        print("⚠️  Found runs on W&B, but they are already in your CSV (matching by 'Name').")
        return

    print(f"Adding {len(unique_new_df)} unique new runs to CSV.")
    
    # Combine
    updated_df = pd.concat([df, unique_new_df], ignore_index=True)
    
    # Success check: did we actually get closer to the target?
    # (Optional: sort or clean here)
    
    backup_path = csv_path + ".bak"
    try:
        # Move original to backup
        if os.path.exists(backup_path): os.remove(backup_path)
        os.rename(csv_path, backup_path)
        
        # Save new
        updated_df.to_csv(csv_path, index=False)
        print(f"✅ Successfully updated {csv_path}. (Backup saved to {backup_path})")
        
        # Final Verification
        total = len(updated_df)
        print(f"New total runs: {total} / 1920")
    except Exception as e:
        print(f"Failed to save updated CSV: {e}")
        if os.path.exists(backup_path):
            os.rename(backup_path, csv_path)
            print("Restored original CSV from backup.")

if __name__ == "__main__":
    sync_missing()
