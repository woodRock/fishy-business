#!/bin/bash
# Date: 2025-09-25
# Author: Gemini

ENCODERS=("cnn" "ensemble" "transformer" "kan" "mamba" "moe" "rcnn" "vae")

for encoder in "${ENCODERS[@]}"; do
    echo "Running simclr for encoder: $encoder"
    python3 -m contrastive.main --encoder_type "$encoder" --contrastive_method simclr --num_runs 30
done