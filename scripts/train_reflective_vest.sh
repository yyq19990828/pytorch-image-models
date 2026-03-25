#!/bin/bash
# Reflective Vest Binary Classification Training Script
# Distributed training with 2 GPUs

# Set directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Training parameters
DATA_DIR="${PROJECT_DIR}/data/reflective_vest"
OUTPUT_DIR="${PROJECT_DIR}/output/reflective_vest_resnet18"
NUM_GPUS=2

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Launch distributed training
"${PROJECT_DIR}/distributed_train.sh" $NUM_GPUS \
    --data-dir "$DATA_DIR" \
    --val-split val \
    --num-classes 2 \
    --model resnet18 \
    --pretrained \
    --batch-size 32 \
    --epochs 50 \
    --lr 0.001 \
    --opt sgd \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --sched cosine \
    --warmup-epochs 5 \
    --warmup-lr 1e-6 \
    --min-lr 1e-5 \
    --amp \
    --checkpoint-hist 5 \
    --output "$OUTPUT_DIR" \
    --workers 4 \
    --pin-mem
