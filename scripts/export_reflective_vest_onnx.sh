#!/bin/bash
# Export Reflective Vest Model to ONNX Format

# Set directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Model paths
MODEL_DIR="${PROJECT_DIR}/output/reflective_vest_resnet18/20251121-185707-resnet18-224"
CHECKPOINT="${MODEL_DIR}/model_best.pth.tar"
OUTPUT_ONNX="${MODEL_DIR}/model_best.onnx"

# Export parameters
MODEL_NAME="resnet18"
NUM_CLASSES=2
IMG_SIZE=224
BATCH_SIZE=1  # Batch size (use 1 for automatic dynamic batch dimension)
OPSET=12

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

echo "Exporting model to ONNX..."
echo "  Checkpoint: $CHECKPOINT"
echo "  Output: $OUTPUT_ONNX"
echo "  Model: $MODEL_NAME"
echo "  Classes: $NUM_CLASSES"
echo "  Input size: [dynamic]x3x${IMG_SIZE}x${IMG_SIZE} (dynamic batch, fixed HxW)"
echo "  OPSET: $OPSET"
echo ""

cd "$PROJECT_DIR"

# Build export command
EXPORT_CMD="python onnx_export.py \
    --model \"$MODEL_NAME\" \
    --num-classes \"$NUM_CLASSES\" \
    --checkpoint \"$CHECKPOINT\" \
    --batch-size \"$BATCH_SIZE\" \
    --img-size \"$IMG_SIZE\" \
    --opset \"$OPSET\" \
    --verbose"

# Note: --dynamic-size in timm makes all dimensions dynamic (batch, height, width)
# We only want dynamic batch, so we don't use --dynamic-size
# Instead, the ONNX export will automatically make batch dimension dynamic
# when batch-size=1 is used without --dynamic-size flag

EXPORT_CMD="$EXPORT_CMD \"$OUTPUT_ONNX\""

# Execute export
eval $EXPORT_CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "Export successful!"
    echo "ONNX model saved to: $OUTPUT_ONNX"
    ls -lh "$OUTPUT_ONNX"
else
    echo ""
    echo "Export failed!"
    exit 1
fi
