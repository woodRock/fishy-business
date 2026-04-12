#!/bin/bash

# Configuration
NUM_RUNS=10      # Number of seeds to run for statistical significance
EPOCHS=50        # Balanced number of epochs for deep models
DATASETS=("species" "part" "oil" "cross-species")
MODELS=("opls-da" "rf" "svm" "dt" "lr" "lstm" "cnn" "transformer")

echo "🚀 Submitting statistical benchmark to cluster spooler..."

for ds in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        
        # 1. Default (No Preprocessing)
        task -G 1 python3 -m fishy cli train \
            -m "$model" -d "$ds" -n $NUM_RUNS \
            --epochs $EPOCHS --wandb-log \
            --project "fishy-stats-v1"

        # 2. Standard Normalization (L2 dim=1)
        task -G 1 python3 -m fishy cli train \
            -m "$model" -d "$ds" -n $NUM_RUNS \
            --normalize --epochs $EPOCHS --wandb-log \
            --project "fishy-stats-v1"

        # 3. Sign-RP Paradigm (Random Projection + Sign Quantization)
        task -G 1 python3 -m fishy cli train \
            -m "$model" -d "$ds" -n $NUM_RUNS \
            --random-projection --quantize \
            --epochs $EPOCHS --wandb-log \
            --project "fishy-stats-v1"

        # 4. Proper TurboQuant (Residual-based QJL)
        task -G 1 python3 -m fishy cli train \
            -m "$model" -d "$ds" -n $NUM_RUNS \
            --turbo-quant --normalize \
            --epochs $EPOCHS --wandb-log \
            --project "fishy-stats-v1"
            
    done
done

echo "✅ All jobs submitted. Use 'task' to monitor progress."
