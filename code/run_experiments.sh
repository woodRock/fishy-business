#!/bin/bash

# Define the SSL methods to iterate over
ssl_methods=("byol" "barlow_twins" "moco" "simsiam")
encoder="transformer"
num_runs=30

# Loop over each SSL method
for ssl_method in "${ssl_methods[@]}"; do
    echo "Running experiment with SSL method: $ssl_method and encoder: $encoder"
    
    # Run the Python script with the specified SSL method and encoder
    python3 -m contrastive.main --contrastive_method "$ssl_method" --encoder_type "$encoder" --num_runs "$num_runs"
    
    echo "Experiment finished for SSL method: $ssl_method and encoder: $encoder"
done

echo "All experiments finished."
