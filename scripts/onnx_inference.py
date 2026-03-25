#!/usr/bin/env python3
"""Batch inference on validation dataset using ONNX model"""
import argparse
import os
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm


def preprocess_image(image_path, img_size=224):
    """Preprocess image for ResNet18 inference"""
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')

        # Resize
        image = image.resize((img_size, img_size), Image.BILINEAR)

        # Convert to numpy array
        img_array = np.array(image).astype(np.float32)

        # Normalize (ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
        img_array = (img_array - mean) / std

        # HWC to CHW
        img_array = img_array.transpose(2, 0, 1)

        # Add batch dimension
        img_array = np.expand_dims(img_array, 0)

        return img_array
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def collect_images(val_dir):
    """Collect all images from validation directory"""
    class_names = ['no_wear', 'wear']
    class_dirs = {
        'no_wear': os.path.join(val_dir, 'normal_pedestrian'),
        'wear': os.path.join(val_dir, 'reflective_vest')
    }

    image_files = []
    labels = []

    for class_idx, (class_name, class_dir) in enumerate(class_dirs.items()):
        if not os.path.exists(class_dir):
            print(f"Warning: Directory not found: {class_dir}")
            continue

        # Collect all jpg images
        images = sorted(Path(class_dir).glob('*.jpg'))
        print(f"Found {len(images)} images in {class_name}")

        for img_path in images:
            image_files.append(str(img_path))
            labels.append(class_idx)

    return image_files, labels, class_names


def main():
    parser = argparse.ArgumentParser(description='Batch ONNX inference on validation dataset')
    parser.add_argument('--model', required=True, help='Path to ONNX model')
    parser.add_argument('--val-dir', required=True, help='Path to validation directory')
    parser.add_argument('--img-size', type=int, default=224, help='Input image size')
    parser.add_argument('--show-errors', action='store_true', help='Show misclassified images')
    args = parser.parse_args()

    print("="*80)
    print("ONNX Model Batch Inference")
    print("="*80)

    # Load ONNX model
    print(f"\nLoading ONNX model: {args.model}")
    session = ort.InferenceSession(args.model)
    input_name = session.get_inputs()[0].name
    print(f"Model loaded successfully")

    # Collect images
    print(f"\nCollecting images from: {args.val_dir}")
    image_files, labels, class_names = collect_images(args.val_dir)
    total_images = len(image_files)

    if total_images == 0:
        print("Error: No images found!")
        return

    print(f"Total images to process: {total_images}")
    print(f"Classes: {class_names}")

    # Statistics
    correct = 0
    class_correct = [0, 0]
    class_total = [0, 0]
    errors = []

    # Progress bar
    print(f"\nRunning inference...")
    print("-"*80)
    print(f"{'Image':<60} {'True':<10} {'Pred':<10} {'Conf_0':<10} {'Conf_1':<10} {'Status':<10}")
    print("-"*80)

    start_time = time.time()

    for idx, (img_path, true_label) in enumerate(tqdm(zip(image_files, labels), total=total_images, desc="Processing")):
        # Preprocess
        input_data = preprocess_image(img_path, args.img_size)
        if input_data is None:
            continue

        # Inference
        outputs = session.run(None, {input_name: input_data})
        logits = outputs[0][0]

        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        # Get prediction
        pred_label = np.argmax(probabilities)
        conf_0 = probabilities[0]
        conf_1 = probabilities[1]

        # Update statistics
        is_correct = (pred_label == true_label)
        if is_correct:
            correct += 1
        else:
            errors.append({
                'path': img_path,
                'true': true_label,
                'pred': pred_label,
                'conf_0': conf_0,
                'conf_1': conf_1
            })

        class_total[true_label] += 1
        if is_correct:
            class_correct[true_label] += 1

        # Print result
        img_name = os.path.basename(img_path)
        true_class = class_names[true_label]
        pred_class = class_names[pred_label]
        status = "✓" if is_correct else "✗"

        print(f"{img_name:<60} {true_class:<10} {pred_class:<10} {conf_0:>9.4f} {conf_1:>9.4f} {status:<10}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print summary
    print("-"*80)
    print("\n" + "="*80)
    print("INFERENCE SUMMARY")
    print("="*80)

    accuracy = 100.0 * correct / total_images if total_images > 0 else 0
    print(f"\nOverall Accuracy: {correct}/{total_images} = {accuracy:.2f}%")
    print(f"Processing Time: {elapsed_time:.2f} seconds")
    print(f"Average Time per Image: {elapsed_time/total_images*1000:.2f} ms")

    print(f"\nPer-Class Results:")
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            class_acc = 100.0 * class_correct[i] / class_total[i]
            print(f"  {class_name:<15} : {class_correct[i]:>4}/{class_total[i]:<4} = {class_acc:>6.2f}%")
        else:
            print(f"  {class_name:<15} : No samples")

    # Confusion matrix
    print(f"\nConfusion Matrix:")
    print(f"{'':>15} {'Predicted:':<15}")
    print(f"{'True:':<15} {'no_wear':<15} {'wear':<15}")

    # Calculate confusion matrix
    tp_0 = class_correct[0]  # True no_wear
    fn_0 = class_total[0] - class_correct[0]  # no_wear predicted as wear
    fn_1 = class_total[1] - class_correct[1]  # wear predicted as no_wear
    tp_1 = class_correct[1]  # True wear

    print(f"{'no_wear':<15} {tp_0:<15} {fn_0:<15}")
    print(f"{'wear':<15} {fn_1:<15} {tp_1:<15}")

    # Show errors if requested
    if args.show_errors and errors:
        print(f"\n{'='*80}")
        print(f"MISCLASSIFIED IMAGES ({len(errors)} total)")
        print("="*80)
        for i, error in enumerate(errors[:20], 1):  # Show first 20 errors
            print(f"\n{i}. {os.path.basename(error['path'])}")
            print(f"   True: {class_names[error['true']]}, Predicted: {class_names[error['pred']]}")
            print(f"   Confidence: no_wear={error['conf_0']:.4f}, wear={error['conf_1']:.4f}")

        if len(errors) > 20:
            print(f"\n... and {len(errors) - 20} more errors")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
