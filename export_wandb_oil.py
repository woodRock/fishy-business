# -*- coding: utf-8 -*-
"""
Optimized W&B Export Script
Fetches finished runs and creates a CSV compatible with the Fishy Business Dashboard.
"""

import pandas as pd
import wandb
import os
from tqdm import tqdm

def run_export(entity="victoria-university-of-wellington", project="fishy-business", output_file="wanb_export_csv.csv"):
    api = wandb.Api()
    
    print(f"Connecting to {entity}/{project}...")
    # Server-side filter for finished runs for speed
    runs = api.runs(f"{entity}/{project}", filters={"state": "finished"})
    
    summary_list, config_list, name_list = [], [], []
    
    print("Fetching run data...")
    for run in tqdm(runs):
        # .summary contains the output metrics
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters/metadata
        config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

        # .name is the human-readable name
        name_list.append(run.name)

    runs_df = pd.DataFrame(
        {"summary": summary_list, "config": config_list, "name": name_list}
    )

    runs_df.to_csv(output_file, index=False)
    print(f"✅ Export complete: {output_file}")

    # Verify 'oil' presence
    try:
        # Check if 'dataset' is 'oil' inside the config column
        oil_count = 0
        for cfg in config_list:
            if cfg.get("dataset") == "oil":
                oil_count += 1
        
        if oil_count > 0:
            print(f"⭐ Success: Found {oil_count} runs for dataset 'oil'")
        else:
            print("⚠️  Warning: No runs with dataset='oil' found in the exported data.")
            print("Available datasets in export:", set([c.get("dataset") for c in config_list if c.get("dataset")]))
    except Exception as e:
        print(f"Could not verify oil count: {e}")

if __name__ == "__main__":
    # You can change entity/project here if needed
    run_export()