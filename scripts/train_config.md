# Reflective Vest Training Configuration

## Current Settings

### Hardware
- **GPUs**: 2 x RTX 3090 (24GB each)
- **Distributed Training**: Yes (via torchrun)

### Model
- **Architecture**: ResNet18
- **Pretrained**: Yes (ImageNet weights)
- **Classes**: 2 (no_wear, wear)

### Training Parameters
- **Epochs**: 50
- **Batch Size**: 32 per GPU (total 64)
- **Mixed Precision**: Enabled (--amp)
- **Workers**: 4

### Optimizer
- **Type**: SGD
- **Learning Rate**: 0.001
- **Momentum**: 0.9
- **Weight Decay**: 1e-4

### Learning Rate Scheduler
- **Type**: Cosine Annealing
- **Warmup Epochs**: 5
- **Warmup LR**: 1e-6
- **Min LR**: 1e-5

### Checkpointing
- **Save Interval**: Every 5 epochs (--checkpoint-hist 5)
- **Best Model**: Saved automatically
- **Last Model**: Saved automatically

### Data
- **Training Samples**: 11,928
- **Validation Samples**: 1,242
- **Data Augmentation**: Default timm augmentations
- **Pin Memory**: Enabled

## Expected Training Time
- **Estimated**: ~10-15 minutes (50 epochs on 2x RTX 3090)
- **Per Epoch**: ~15-20 seconds

## Output Structure
```
output/reflective_vest_resnet18/
├── checkpoint-0.pth.tar
├── checkpoint-5.pth.tar
├── checkpoint-10.pth.tar
├── ...
├── checkpoint-45.pth.tar
├── last.pth.tar
├── model_best.pth.tar
├── args.yaml
└── summary.csv
```

## AMP Benefits
- **Memory Savings**: ~40-50% reduction
- **Speed Improvement**: ~2-3x faster training
- **Accuracy**: Minimal impact (usually <0.1% difference)

## Usage
```bash
# Run training
./scripts/train_reflective_vest.sh

# Monitor with nvidia-smi
watch -n 1 nvidia-smi

# View logs (if using redirected output)
tail -f output/reflective_vest_resnet18/train.log
```

## Modifying Parameters

Edit `scripts/train_reflective_vest.sh`:

- Change epochs: Modify `--epochs 50`
- Change batch size: Modify `--batch-size 32`
- Disable AMP: Remove `--amp`
- Change save interval: Modify `--checkpoint-hist 5`
- Change model: Modify `--model resnet18` (e.g., resnet50, efficientnet_b0)
