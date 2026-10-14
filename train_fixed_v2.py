"""
train_fixed_v2.py — single-stage coffee defect trainer
======================================================
A simplified trainer that uses the unified model factory with ImageNet
pretrained backbones, trains the full model in one stage, and does not
freeze any backbone or classifier layers.

Usage example:
    python train_fixed_v2.py --variety arabica --model mobilenetv2 --optimizer adam --lr 1e-4 --batch_size 32 --dropout 0.3 --max_epochs 100
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torchvision import datasets

from model_factory import get_model, get_transforms

CLASSES = ["black", "broken", "moldy"]
IMG_SIZE = 224
SEED = 42
PATIENCE = 100
DATASET_ROOT = Path("coffee_dataset")


def parse_args():
    p = argparse.ArgumentParser(description="Coffee defect classifier — single stage training")
    p.add_argument("--variety", required=True, choices=["arabica", "robusta"])
    p.add_argument(
        "--model",
        required=True,
        choices=["squeezenet", "shufflenet", "mobilenetv1", "mobilenetv2", "mobilenetv3", "efficientnet"],
    )
    p.add_argument("--optimizer", required=True, choices=["adam", "adamw", "sgd", "rmsprop", "adan"])
    p.add_argument("--lr", type=float, required=True, help="Learning rate")
    p.add_argument("--batch_size", type=int, required=True, help="Batch size")
    p.add_argument("--dropout", type=float, required=True, help="Dropout rate")
    p.add_argument("--max_epochs", type=int, default=100, help="Max training epochs")
    p.add_argument("--dataset_root", default=str(DATASET_ROOT))
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--run_name", default=None)
    p.add_argument("--pretrained_from", default=None, help="Optional Stage 1 checkpoint to load before fine-tuning")
    return p.parse_args()


def load_dataset(variety_dir: Path, model_name: str, split: str, batch_size: int, shuffle: bool = False) -> DataLoader:
    train_tf, eval_tf = get_transforms(model_name, train=(split == "train"))
    transform = train_tf if split == "train" else eval_tf

    ds = datasets.ImageFolder(root=str(variety_dir / split), transform=transform)
    assert set(ds.classes) == set(CLASSES), f"Expected classes {CLASSES}, found {ds.classes}"

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )


def compute_class_weights(variety_dir: Path, device: torch.device) -> torch.Tensor:
    ds = datasets.ImageFolder(root=str(variety_dir / "train"))
    labels = [label for _, label in ds.samples]
    class_indices = np.array([ds.class_to_idx[class_name] for class_name in CLASSES])
    weights = compute_class_weight("balanced", classes=class_indices, y=np.array(labels))
    print(
        f"  Class weights → black:{weights[0]:.3f}  "
        f"broken:{weights[1]:.3f}  moldy:{weights[2]:.3f}"
    )
    return torch.tensor(weights, dtype=torch.float32).to(device)


def get_optimizer(name: str, params, lr: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    if name == "adan":
        from adan import Adan
        return Adan(params, lr=lr, weight_decay=2e-2, betas=(0.98, 0.92, 0.99))
    raise ValueError(f"Unknown optimizer: {name}")


def run_epoch(model, loader, criterion, optimizer, device, phase="train"):
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, loader, device, label: str) -> dict:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    print(f"\n── {label} ──")
    print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))
    print("Confusion matrix [black / broken / moldy]:")
    print(np.array(cm))

    return {"label": label, "report": report, "confusion_matrix": cm}


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 65}")
    print(f"  Device    : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  Variety   : {args.variety}")
    print(f"  Model     : {args.model}")
    print(f"  Optimizer : {args.optimizer}  LR={args.lr}")
    print(f"  Batch     : {args.batch_size}   Dropout={args.dropout}")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"{'=' * 65}\n")

    dataset_root = Path(args.dataset_root)
    train_dir = dataset_root / args.variety
    other_variety = "robusta" if args.variety == "arabica" else "arabica"
    other_dir = dataset_root / other_variety

    if args.run_name:
        run_name = args.run_name
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"single_NORLR_{args.variety}_{args.model}_{args.optimizer}_lr{args.lr}_bs{args.batch_size}_do{args.dropout}_{ts}"

    run_dir = Path(args.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(run_dir / "best_model.pth")

    train_loader = load_dataset(train_dir, args.model, "train", args.batch_size, shuffle=True)
    val_loader = load_dataset(train_dir, args.model, "val", args.batch_size)
    test_same_loader = load_dataset(train_dir, args.model, "test", args.batch_size)
    test_other_loader = load_dataset(other_dir, args.model, "test", args.batch_size)

    class_weights = compute_class_weights(train_dir, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = get_model(args.model, num_classes=len(CLASSES), dropout=args.dropout, pretrained=True)
    model = model.to(device)

    if args.pretrained_from:
        pretrained_path = Path(args.pretrained_from)
        if not pretrained_path.is_file():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")
        pretrained_state = torch.load(pretrained_path, map_location=device, weights_only=True)
        model.load_state_dict(pretrained_state["model_state_dict"])
        print(f"[INFO] Loaded Stage 1 checkpoint: {pretrained_path}")

    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr)
    best_val_loss = float("inf")
    best_epoch = 0
    history = []

    print("\n[TRAINING] Single-stage full-model fine-tune with ImageNet pretrained backbone...")
    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, "train")
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion, optimizer, device, "val")
        elapsed = time.time() - t0

        print(f"  Epoch {epoch:3d}/{args.max_epochs} | train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={vl_loss:.4f} acc={vl_acc:.4f} | {elapsed:.1f}s")
        history.append({
            "epoch": epoch,
            "tr_loss": tr_loss,
            "tr_acc": tr_acc,
            "vl_loss": vl_loss,
            "vl_acc": vl_acc,
        })

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": vl_loss}, ckpt_path)
            print(f"    Checkpoint saved (val_loss={vl_loss:.4f})")

    print(f"\n[INFO] Loading best checkpoint (epoch {best_epoch}, val_loss={best_val_loss:.4f})")
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])

    results = [
        evaluate(model, test_same_loader, device, f"{args.model} | trained:{args.variety} → tested:{args.variety}"),
        evaluate(model, test_other_loader, device, f"{args.model} | trained:{args.variety} → tested:{other_variety}"),
    ]

    meta = {
        "run_name": run_name,
        "variety": args.variety,
        "model": args.model,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "max_epochs": args.max_epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "runs_dir": str(args.runs_dir),
        "results": results,
        "history": history,
    }

    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[DONE]")
    print(f"  Run dir   : {run_dir}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Results   : {results_path}")


if __name__ == "__main__":
    main()
