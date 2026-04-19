#!/bin/bash
# Benchmark AugFormer vs OPLS-DA on the batch-detection dataset.
# Runs 30 iterations each for statistical significance.
# Results are logged to WandB.

RUNS=30
EPOCHS=100
DROPOUT=0.3
DIRECTORY=$(pwd)

MODELS=("opls-da" "augformer")
DATASET="batch-detection"

echo "Starting AugFormer vs OPLS-DA Benchmarking Suite"
echo "  Runs: $RUNS | Epochs: $EPOCHS (for deep models) | Dropout: $DROPOUT"
echo "  Models:   ${MODELS[*]}"
echo "  Dataset:  $DATASET"
echo ""

for MODEL in "${MODELS[@]}"; do
    LABEL="benchmark_${MODEL}_${DATASET}"
    
    # --normalize is applied to ensure consistency across methods
    # --wandb-log ensures results are captured for analysis
    # --benchmark enables the multi-run execution engine
    CMD="fishy train -m ${MODEL} -d ${DATASET} --normalize --dropout ${DROPOUT} -N ${RUNS} -e ${EPOCHS} --wandb-log --benchmark"
    
    echo "Queuing: $LABEL"
    echo "  $CMD"
    
    # Use 'task' (task spooler) for GPU queuing if installed. 
    if command -v task &> /dev/null; then
        task -G 1 -n "$LABEL" -d "$DIRECTORY" $CMD
    else
        eval $CMD
    fi
done

echo ""
echo "Benchmark suite initiated. Monitor with: task -l"
