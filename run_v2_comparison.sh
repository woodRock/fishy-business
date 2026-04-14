#!/bin/bash
# Benchmark AugFormer vs AugFormerV2 (AdamW) vs AugFormerV2 (Muon) 
# across all datasets (30 runs each).

RUNS=30
EPOCHS=100
DROPOUT=0.3
DIRECTORY=$(pwd)

DATASETS=("species" "part" "oil" "cross-species")

echo "Starting AugFormer vs AugFormerV2 ($RUNS runs) Benchmarking"
echo "  Runs: $RUNS | Epochs: $EPOCHS | Dropout: $DROPOUT"
echo "  Optimizers: AdamW, Muon"
echo "  Logging: WandB"
echo ""

for DATASET in "${DATASETS[@]}"; do
    # 1. Baseline AugFormer (AdamW)
    MODEL="augformer"
    LABEL="v2_bench_${MODEL}_adamw_${DATASET}"
    CMD="fishy train -m ${MODEL} -d ${DATASET} --normalize --dropout ${DROPOUT} -N ${RUNS} -e ${EPOCHS} --benchmark --wandb-log --optimizer adamw"
    
    echo "Queuing: $LABEL"
    if command -v task &> /dev/null; then
        task -G 1 -n "$LABEL" -d "$DIRECTORY" $CMD
    else
        eval $CMD
    fi

    # 2. AugFormerV2 (AdamW) + Golf Features + TTT
    MODEL="augformer_v2"
    LABEL="v2_bench_${MODEL}_adamw_${DATASET}"
    CMD="fishy train -m ${MODEL} -d ${DATASET} --normalize --dropout ${DROPOUT} -N ${RUNS} -e ${EPOCHS} --benchmark --wandb-log --optimizer adamw --qk-gain --parallel-residuals --recurrence 1 2 --leaky-sq --ttt"
    
    echo "Queuing: $LABEL"
    if command -v task &> /dev/null; then
        task -G 1 -n "$LABEL" -d "$DIRECTORY" $CMD
    else
        eval $CMD
    fi

    # 3. AugFormerV2 (Muon) + Golf Features + TTT
    MODEL="augformer_v2"
    LABEL="v2_bench_${MODEL}_muon_${DATASET}"
    # Muon typically handles 100 epochs very well due to fast convergence
    CMD="fishy train -m ${MODEL} -d ${DATASET} --normalize --dropout ${DROPOUT} -N ${RUNS} -e ${EPOCHS} --benchmark --wandb-log --optimizer muon --qk-gain --parallel-residuals --recurrence 1 2 --leaky-sq --ttt"
    
    echo "Queuing: $LABEL"
    if command -v task &> /dev/null; then
        task -G 1 -n "$LABEL" -d "$DIRECTORY" $CMD
    else
        eval $CMD
    fi
done

echo ""
echo "Comparison suite initiated. Monitor with: task -l"
