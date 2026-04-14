#!/bin/bash
# Benchmark AugFormer with vs without Exclusive Self-Attention (XSA).
# Each configuration is run 30 times per dataset with consistent seeds.
# Results are logged to WandB for statistical analysis.

RUNS=30
EPOCHS=100
DROPOUT=0.3
DIRECTORY=$(pwd)

DATASETS=("species" "part" "oil" "cross-species")

echo "Starting AugFormer XSA Ablation Suite"
echo "  Runs: $RUNS | Epochs: $EPOCHS | Dropout: $DROPOUT"
echo "  Datasets: ${DATASETS[*]}"
echo ""

for DATASET in "${DATASETS[@]}"; do
    for MODE in "std" "xsa"; do
        if [ "$MODE" == "xsa" ]; then
            XSA_FLAG="--xsa"
            LABEL="benchmark_augformer_xsa_${DATASET}"
        else
            XSA_FLAG=""
            LABEL="benchmark_augformer_std_${DATASET}"
        fi
        
        # We use --normalize as it was critical for our recent improvements
        # -N 30 handles the 30 runs with deterministic seeds (e.g., run 1 uses seed 123 for both modes)
        # Default k_folds is 3, providing 3-fold stratified cross-validation per run.
        CMD="fishy train -m augformer -d ${DATASET} --normalize --dropout ${DROPOUT} -N ${RUNS} -e ${EPOCHS} --wandb-log --benchmark ${XSA_FLAG}"
        
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
echo "XSA benchmark suite initiated. Monitor with: task -l"
