#!/bin/bash
# Compare Transformer vs CoralFormer across all datasets.
# Submits each (model, dataset) pair as a separate GPU task with 5 seeds.
#
# Usage:
#   ./run_coralformer_comparison.sh
#   ./run_coralformer_comparison.sh 10   # override number of runs

RUNS=${1:-5}
EPOCHS=${2:-100}
DROPOUT=0.3
DIRECTORY="/vol/ecrg-solar/woodj4/fishy-business"

MODELS=("transformer" "coralformer")
DATASETS=("species" "part" "oil" "cross-species")

echo "Submitting Transformer vs CoralFormer comparison"
echo "  Runs: $RUNS | Epochs: $EPOCHS | Dropout: $DROPOUT"
echo "  Models:   ${MODELS[*]}"
echo "  Datasets: ${DATASETS[*]}"
echo ""

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        LABEL="${MODEL}_${DATASET}"
        CMD="fishy train -m ${MODEL} -d ${DATASET} --normalize --dropout ${DROPOUT} -N ${RUNS} -e ${EPOCHS} --wandb-log"
        echo "Queuing: $LABEL"
        echo "  $CMD"
        task -G 1 -n "$LABEL" -d "$DIRECTORY" $CMD
    done
done

echo ""
echo "All tasks queued. Monitor with: task -l"
