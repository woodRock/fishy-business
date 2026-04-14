#!/bin/bash
# Benchmark AugFormer vs AugFormerV2 across all datasets (1 seed each).

EPOCHS=100
DROPOUT=0.3
DIRECTORY=$(pwd)
SEED=42

DATASETS=("species" "part" "oil" "cross-species")

echo "Starting AugFormer vs AugFormerV2 (1-seed) Benchmarking"
echo "  Seed: $SEED | Epochs: $EPOCHS | Dropout: $DROPOUT"
echo ""

for DATASET in "${DATASETS[@]}"; do
    # 1. Baseline AugFormer
    MODEL="augformer"
    LABEL="v2test_${MODEL}_${DATASET}"
    CMD="fishy train -m ${MODEL} -d ${DATASET} --normalize --dropout ${DROPOUT} -N 1 -e ${EPOCHS} --seed ${SEED} --benchmark"
    
    echo "Queuing: $LABEL"
    if command -v task &> /dev/null; then
        task -G 1 -n "$LABEL" -d "$DIRECTORY" $CMD
    else
        eval $CMD
    fi

    # 2. AugFormerV2 with Golf Features
    MODEL="augformer_v2"
    LABEL="v2test_${MODEL}_${DATASET}"
    # Using QK-Gain, Parallel Residuals, Recurrence (layers 1, 2), and LeakyReLU^2
    CMD="fishy train -m ${MODEL} -d ${DATASET} --normalize --dropout ${DROPOUT} -N 1 -e ${EPOCHS} --seed ${SEED} --benchmark --qk-gain --parallel-residuals --recurrence 1 2 --leaky-sq"
    
    echo "Queuing: $LABEL"
    if command -v task &> /dev/null; then
        task -G 1 -n "$LABEL" -d "$DIRECTORY" $CMD
    else
        eval $CMD
    fi
done

echo ""
echo "Comparison suite initiated. Monitor with: task -l"
