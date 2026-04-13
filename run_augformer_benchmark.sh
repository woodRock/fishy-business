#!/bin/bash
# Benchmark AugFormer vs GatedMLP across all datasets (10 runs each).
# Results are logged to WandB for statistical analysis.

RUNS=10
EPOCHS=100
DROPOUT=0.3
DIRECTORY=$(pwd)

MODELS=("gatedmlp" "augformer")
DATASETS=("species" "part" "oil" "cross-species")

echo "Starting AugFormer vs GatedMLP Benchmarking Suite"
echo "  Runs: $RUNS | Epochs: $EPOCHS | Dropout: $DROPOUT"
echo "  Models:   ${MODELS[*]}"
echo "  Datasets: ${DATASETS[*]}"
echo ""

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        LABEL="benchmark_${MODEL}_${DATASET}"
        
        # We use --normalize as it was critical for our recent improvements
        # --wandb-log ensures we can export the CSV for analysis
        CMD="fishy train -m ${MODEL} -d ${DATASET} --normalize --dropout ${DROPOUT} -N ${RUNS} -e ${EPOCHS} --wandb-log --benchmark"
        
        echo "Queuing: $LABEL"
        echo "  $CMD"
        
        # If 'task' (task spooler) is installed, use it for GPU queuing. 
        # Otherwise, run sequentially.
        if command -v task &> /dev/null; then
            task -G 1 -n "$LABEL" -d "$DIRECTORY" $CMD
        else
            eval $CMD
        fi
    done
done

echo ""
echo "Benchmark suite initiated. If using 'task', monitor with: task -l"
