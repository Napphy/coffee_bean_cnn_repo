#!/usr/bin/env python3
"""Classify images using a saved train_fixed_v2 checkpoint.

Examples:
    python test_saved_checkpoint.py \
        --run_dir linuxruns/fixed_NORLR_arabica_squeezenet_sgd_lr1e-4_bs16_do0.2 \
        --image path/to/bean.jpg

    python test_saved_checkpoint.py \
        --checkpoint linuxruns/fixed_NORLR_arabica_squeezenet_sgd_lr1e-4_bs16_do0.2/best_model.pth \
        --model squeezenet --dropout 0.2 --image path/to/images --recursive \
        --output predictions.csv
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image

from model_factory import get_model, get_transforms

CLASSES = ["black", "broken", "moldy"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MODEL_NAMES = ["squeezenet", "shufflenet", "mobilenetv1", "mobilenetv2", "mobilenetv3", "efficientnet"]


def parse_args():
    parser = argparse.ArgumentParser(description="Classify images with a saved coffee-defect checkpoint")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run_dir", help="Run directory containing best_model.pth and results.json")
    source.add_argument("--checkpoint", help="Path to best_model.pth")
    parser.add_argument("--image", required=True, help="One image file or a directory of images")
    parser.add_argument("--model", choices=MODEL_NAMES, help="Required with --checkpoint; inferred with --run_dir")
    parser.add_argument("--dropout", type=float, help="Required with --checkpoint; inferred with --run_dir")
    parser.add_argument("--recursive", action="store_true", help="Search image directories recursively")
    parser.add_argument("--output", help="Optional CSV path for predictions")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Default: CUDA when available")
    return parser.parse_args()


def load_run_config(args):
    if args.run_dir:
        run_dir = Path(args.run_dir)
        checkpoint = run_dir / "best_model.pth"
        metadata_path = run_dir / "results.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Run metadata not found: {metadata_path}")
        with metadata_path.open(encoding="utf-8") as file:
            metadata = json.load(file)
        model_name = metadata.get("model")
        dropout = metadata.get("dropout", 0.2)
    else:
        checkpoint = Path(args.checkpoint)
        model_name = args.model
        dropout = args.dropout
        if model_name is None or dropout is None:
            raise ValueError("--model and --dropout are required when using --checkpoint")

    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unsupported or missing model in run metadata: {model_name}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return checkpoint, model_name, float(dropout)


def collect_images(image_path: Path, recursive: bool):
    if image_path.is_file():
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {image_path.suffix}")
        return [image_path]
    if not image_path.is_dir():
        raise FileNotFoundError(f"Image path not found: {image_path}")

    pattern = "**/*" if recursive else "*"
    images = sorted(
        path for path in image_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No supported images found in: {image_path}")
    return images


def load_model(checkpoint: Path, model_name: str, dropout: float, device: torch.device):
    model = get_model(model_name, num_classes=len(CLASSES), dropout=dropout, pretrained=False)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if "model_state_dict" not in state:
        raise KeyError(f"Checkpoint does not contain model_state_dict: {checkpoint}")
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict(model, transform, image_path: Path, device: torch.device):
    with Image.open(image_path) as image:
        image_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(image_tensor), dim=1)[0]
    confidence, class_index = probabilities.max(dim=0)
    return CLASSES[class_index.item()], confidence.item(), probabilities.cpu().tolist()


def main():
    args = parse_args()
    checkpoint, model_name, dropout = load_run_config(args)
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(device_name)

    _, eval_transform = get_transforms(model_name, train=False)
    model = load_model(checkpoint, model_name, dropout, device)
    images = collect_images(Path(args.image), args.recursive)

    print(f"Model: {model_name} | checkpoint: {checkpoint} | device: {device}")
    rows = []
    for image_path in images:
        predicted_class, confidence, probabilities = predict(model, eval_transform, image_path, device)
        row = {
            "image": str(image_path),
            "prediction": predicted_class,
            "confidence": confidence,
            "black_probability": probabilities[0],
            "broken_probability": probabilities[1],
            "moldy_probability": probabilities[2],
        }
        rows.append(row)
        print(
            f"{image_path}: {predicted_class} ({confidence:.2%}) | "
            f"black={probabilities[0]:.2%}, broken={probabilities[1]:.2%}, moldy={probabilities[2]:.2%}"
        )

    if args.output:
        output_path = Path(args.output)
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved predictions: {output_path}")


if __name__ == "__main__":
    main()
