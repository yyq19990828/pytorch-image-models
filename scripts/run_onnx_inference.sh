#!/bin/bash
# Run ONNX inference on validation dataset

# Set directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default paths
MODEL_PATH="${PROJECT_DIR}/output/reflective_vest_resnet18/20251117-192035-resnet18-224/model_best.onnx"
VAL_DIR="${PROJECT_DIR}/data/reflective_vest/val"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please update MODEL_PATH in this script"
    exit 1
fi

# Check if validation directory exists
if [ ! -d "$VAL_DIR" ]; then
    echo "Error: Validation directory not found at $VAL_DIR"
    exit 1
fi

echo "Running ONNX inference..."
echo "Model: $MODEL_PATH"
echo "Validation directory: $VAL_DIR"
echo ""

cd "$PROJECT_DIR"

# Run inference with error display
python scripts/onnx_inference.py \
    --model "$MODEL_PATH" \
    --val-dir "$VAL_DIR" \
    --show-errors

echo ""
echo "Inference completed!"
