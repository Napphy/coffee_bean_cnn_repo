"""
train.py — Coffee Defect Classifier (PyTorch)
==============================================
Trains one model on one variety with one hyperparameter configuration.
Designed to be called repeatedly by run_hyperparameter_search.ps1.

Usage
-----
# Phase 1/2 — train from ImageNet weights
python train.py --variety arabica --model mobilenetv2 --optimizer adam --lr 1e-4 --batch_size 32 --dropout 0.3 --max_epochs 100

# Phase 3 — fine-tune a variety-trained checkpoint on the other variety
python train.py --variety robusta --model mobilenetv2 --optimizer adam --lr 1e-4 --batch_size 32 --dropout 0.3 --max_epochs 100 \
    --pretrained_from runs/phase12_arabica_mobilenetv2_adam_lr1e-4_bs32_do0.3/best_model.pth

Dependencies
------------
pip install torch torchvision
pip install git+https://github.com/sail-sg/Adan.git   # official Adan
pip install scikit-learn
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
CLASSES      = ["black", "broken", "moldy"]
IMG_SIZE     = 224
SEED         = 42
PATIENCE     = 7        # early stopping patience (epochs without val_loss improvement)
DATASET_ROOT = Path("coffee_dataset")

BACKBONE_MAP = {
    "squeezenet":   "squeezenet1_1",
    "shufflenet":   "shufflenet_v2_x1_0",
    "mobilenetv2":  "mobilenet_v2",
    "mobilenetv3":  "mobilenet_v3_large",
    "efficientnet": "efficientnet_b0",
}

# SqueezeNet1.1  — ~1.2M params, very fast, fire modules
# ShuffleNetV2   — ~2.3M params, channel shuffle, efficient on CPU+GPU
# MobileNetV2    — ~3.4M params, inverted residuals
# MobileNetV3L   — ~5.4M params, hard-swish, SE blocks
# EfficientNetB0 — ~5.3M params, compound scaling
 

# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Coffee defect classifier — PyTorch")
    p.add_argument("--variety",          required=True, choices=["arabica", "robusta"])
    p.add_argument("--model",            required=True,
                   choices=["squeezenet", "shufflenet", "mobilenetv2", "mobilenetv3", "efficientnet"])
    p.add_argument("--optimizer",        required=True,
                   choices=["adam", "adamw", "sgd", "rmsprop", "adan"])
    p.add_argument("--lr",               type=float, required=True,
                   help="Learning rate (e.g. 1e-3, 1e-4, 1e-5)")
    p.add_argument("--batch_size",       type=int,   required=True,
                   help="Batch size (8, 16, 32)")
    p.add_argument("--dropout",          type=float, required=True,
                   help="Dropout rate before classifier (0.3, 0.5, 0.7)")
    p.add_argument("--max_epochs",       type=int,   default=100,
                   help="Maximum epochs — early stopping will likely kick in first")
    p.add_argument("--pretrained_from",  default=None,
                   help="Path to .pth checkpoint from other variety (Phase 3)")
    p.add_argument("--dataset_root",     default=str(DATASET_ROOT))
    return p.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────
def get_transforms(split: str):
    """ImageNet-style normalisation; augmentation only for train."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    )
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.05),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            normalize,
        ])


def load_dataset(variety_dir: Path, split: str, batch_size: int,
                 shuffle: bool = False) -> DataLoader:
    ds = datasets.ImageFolder(
        root=str(variety_dir / split),
        transform=get_transforms(split),
    )
    # Reorder class_to_idx to match CLASSES = [black, broken, moldy]
    assert set(ds.classes) == set(CLASSES), \
        f"Expected classes {CLASSES}, found {ds.classes}"
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )


def compute_class_weights(variety_dir: Path, device: torch.device) -> torch.Tensor:
    ds = datasets.ImageFolder(root=str(variety_dir / "train"))
    labels = [label for _, label in ds.samples]
    weights = compute_class_weight("balanced",
                                    classes=np.array([0, 1, 2]),
                                    y=np.array(labels))
    print(f"  Class weights → black:{weights[0]:.3f}  "
          f"broken:{weights[1]:.3f}  moldy:{weights[2]:.3f}")
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
def get_backbone(model_name: str, dropout: float,
                 pretrained_from: str | None) -> nn.Module:
    """
    Build model.
    - pretrained_from=None  → ImageNet weights, head replaced, backbone frozen.
    - pretrained_from=<path>→ load checkpoint, replace head, partial freeze.
    """
    tv_name = BACKBONE_MAP[model_name]
 
    if pretrained_from:
        # ── Phase 3: load existing checkpoint ──────────────────────────────
        print(f"  Loading checkpoint: {pretrained_from}")
        model = _build_fresh(tv_name, dropout, imagenet=False)
        state = torch.load(pretrained_from, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        # Replace classifier head for the new variety
        model = _replace_head(model, tv_name, dropout)
        # Freeze all but the last block + new head
        _freeze_except_top(model, tv_name, n_unfreeze=20)
    else:
        # ── Phase 1/2: ImageNet weights, frozen backbone ────────────────────
        model = _build_fresh(tv_name, dropout, imagenet=True)
        _freeze_backbone(model, tv_name)
 
    return model
 
 
def _build_fresh(tv_name: str, dropout: float, imagenet: bool) -> nn.Module:
    weights = "IMAGENET1K_V1" if imagenet else None
 
    if tv_name == "squeezenet1_1":
        model = models.squeezenet1_1(weights=weights)
        # SqueezeNet's classifier is a Conv2d(512, 1000) — replace entirely
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(512, len(CLASSES), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        model.num_classes = len(CLASSES)
 
    elif tv_name == "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, len(CLASSES)),
        )
 
    elif tv_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.last_channel, len(CLASSES)),
        )
 
    elif tv_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, len(CLASSES))
        model.classifier[0] = nn.Sequential(
            nn.Dropout(p=dropout), model.classifier[0]
        )
 
    elif tv_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, len(CLASSES)),
        )
 
    return model
 
 
def _replace_head(model: nn.Module, tv_name: str, dropout: float) -> nn.Module:
    """Swap out the final classifier only (keeps backbone weights intact)."""
    if tv_name == "squeezenet1_1":
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(512, len(CLASSES), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        model.num_classes = len(CLASSES)
 
    elif tv_name == "shufflenet_v2_x1_0":
        in_features = model.fc[-1].in_features \
            if hasattr(model.fc[-1], "in_features") else 1024
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, len(CLASSES)),
        )
 
    elif tv_name == "mobilenet_v2":
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.last_channel, len(CLASSES)),
        )
    elif tv_name == "mobilenet_v3_large":
        in_features = model.classifier[-1].in_features \
            if hasattr(model.classifier[-1], "in_features") else 1280
        model.classifier[-1] = nn.Linear(in_features, len(CLASSES))
    elif tv_name == "efficientnet_b0":
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, len(CLASSES)),
        )
    return model
 
 
def _freeze_backbone(model: nn.Module, tv_name: str):
    """Freeze everything except the classifier head."""
    if tv_name == "squeezenet1_1":
        for param in model.features.parameters():
            param.requires_grad = False
    elif tv_name == "shufflenet_v2_x1_0":
        # Freeze all conv stages; leave fc unfrozen
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False
    elif tv_name in ("mobilenet_v2", "mobilenet_v3_large", "efficientnet_b0"):
        for param in model.features.parameters():
            param.requires_grad = False
 
 
def _freeze_except_top(model: nn.Module, tv_name: str, n_unfreeze: int = 20):
    """Freeze backbone, unfreeze the last n_unfreeze feature layers + head."""
    if tv_name == "squeezenet1_1":
        feature_layers = list(model.features.children())
        freeze_up_to = max(0, len(feature_layers) - n_unfreeze)
        for i, layer in enumerate(feature_layers):
            for param in layer.parameters():
                param.requires_grad = (i >= freeze_up_to)
    elif tv_name == "shufflenet_v2_x1_0":
        # ShuffleNet stages: conv1, maxpool, stage2, stage3, stage4, conv5, fc
        # Unfreeze stage4, conv5, fc
        unfreeze_names = ("stage4", "conv5", "fc")
        for name, param in model.named_parameters():
            param.requires_grad = any(name.startswith(u) for u in unfreeze_names)
    elif tv_name in ("mobilenet_v2", "mobilenet_v3_large", "efficientnet_b0"):
        feature_layers = list(model.features.children())
        freeze_up_to = max(0, len(feature_layers) - n_unfreeze)
        for i, layer in enumerate(feature_layers):
            for param in layer.parameters():
                param.requires_grad = (i >= freeze_up_to)
 
 
def unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True

# ──────────────────────────────────────────────────────────────────────────────
# Optimizer factory
# ──────────────────────────────────────────────────────────────────────────────
def get_optimizer(name: str, params, lr: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    elif name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9,
                                weight_decay=1e-4, nesterov=True)
    elif name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, momentum=0.9,
                                    weight_decay=1e-4)
    elif name == "adan":
        try:
            from adan import Adan
        except ImportError:
            raise ImportError(
                "Adan not installed. Run:\n"
                "  pip install git+https://github.com/sail-sg/Adan.git"
            )
        # Paper recommends lr 5-10x higher than Adam; we honour the user's
        # explicit --lr so all optimizers are compared on equal footing.
        return Adan(params, lr=lr, weight_decay=2e-2,
                    betas=(0.98, 0.92, 0.99))
    else:
        raise ValueError(f"Unknown optimizer: {name}")


# ──────────────────────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, phase="train"):
    is_train = (phase == "train")
    model.train() if is_train else model.eval()

    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping — helps SGD and Adan
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            preds = outputs.argmax(dim=1)
            total_loss    += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


class EarlyStopping:
    def __init__(self, patience: int, ckpt_path: str):
        self.patience   = patience
        self.ckpt_path  = ckpt_path
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_epoch = 0

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """Returns True when training should stop."""
        if val_loss < self.best_loss - 1e-4:
            self.best_loss  = val_loss
            self.best_epoch = epoch
            self.counter    = 0
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss}, self.ckpt_path)
            print(f"    ✓ Checkpoint saved (val_loss={val_loss:.4f})")
        else:
            self.counter += 1
            print(f"    Early stop counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print(f"    ✗ Early stopping triggered at epoch {epoch}. "
                      f"Best was epoch {self.best_epoch}.")
                return True
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────
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

    report = classification_report(
        y_true, y_pred,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    print(f"\n── {label} ──")
    print(classification_report(y_true, y_pred,
                                  target_names=CLASSES, zero_division=0))
    print("Confusion matrix [black / broken / moldy]:")
    print(np.array(cm))

    return {"label": label, "report": report, "confusion_matrix": cm}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print(f"  Device    : {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  Variety   : {args.variety}")
    print(f"  Model     : {args.model}")
    print(f"  Optimizer : {args.optimizer}  LR={args.lr}")
    print(f"  Batch     : {args.batch_size}   Dropout={args.dropout}")
    print(f"  Max epochs: {args.max_epochs}  (early stop patience={PATIENCE})")
    if args.pretrained_from:
        print(f"  Phase 3   : fine-tuning from {args.pretrained_from}")
    print(f"{'='*65}\n")

    # ── Paths ────────────────────────────────────────────────────────────────
    dataset_root  = Path(args.dataset_root)
    train_dir     = dataset_root / args.variety
    other_variety = "robusta" if args.variety == "arabica" else "arabica"
    other_dir     = dataset_root / other_variety

    lr_str  = str(args.lr).replace("0.", "").replace("-", "")   # "1e-4" → "1e4"
    phase   = "phase3" if args.pretrained_from else "phase12"
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (f"{phase}_{args.variety}_{args.model}_{args.optimizer}"
                f"_lr{args.lr}_bs{args.batch_size}_do{args.dropout}_{ts}")
    run_dir  = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(run_dir / "best_model.pth")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader      = load_dataset(train_dir, "train", args.batch_size, shuffle=True)
    val_loader        = load_dataset(train_dir, "val",   args.batch_size)
    test_same_loader  = load_dataset(train_dir, "test",  args.batch_size)
    test_other_loader = load_dataset(other_dir, "test",  args.batch_size)

    class_weights = compute_class_weights(train_dir, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Model ────────────────────────────────────────────────────────────────
    model = get_backbone(args.model, args.dropout, args.pretrained_from)
    model = model.to(device)

    stopper = EarlyStopping(patience=PATIENCE, ckpt_path=ckpt_path)
    history = []

    # ── Phase A: train head only (backbone frozen) ───────────────────────────
    print("\n[PHASE A] Training classification head (backbone frozen)...")
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = get_optimizer(args.optimizer, head_params, args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    head_epochs = min(10, args.max_epochs)
    for epoch in range(1, head_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, "train")
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, device, "val")
        scheduler.step(vl_loss)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{head_epochs} | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val loss={vl_loss:.4f} acc={vl_acc:.4f} | {elapsed:.1f}s")
        history.append({"phase": "A", "epoch": epoch,
                         "tr_loss": tr_loss, "tr_acc": tr_acc,
                         "vl_loss": vl_loss, "vl_acc": vl_acc})
        if stopper.step(vl_loss, model, epoch):
            break

    # ── Phase B: unfreeze all, fine-tune with lower LR ───────────────────────
    print("\n[PHASE B] Full fine-tune (all layers unfrozen)...")
    unfreeze_all(model)
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr / 10)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    stopper.counter = 0   # reset patience for phase B

    remaining = args.max_epochs - head_epochs
    for epoch in range(head_epochs + 1, head_epochs + remaining + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, "train")
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, device, "val")
        scheduler.step(vl_loss)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{args.max_epochs} | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val loss={vl_loss:.4f} acc={vl_acc:.4f} | {elapsed:.1f}s")
        history.append({"phase": "B", "epoch": epoch,
                         "tr_loss": tr_loss, "tr_acc": tr_acc,
                         "vl_loss": vl_loss, "vl_acc": vl_acc})
        if stopper.step(vl_loss, model, epoch):
            break

    # ── Load best checkpoint ─────────────────────────────────────────────────
    print(f"\n[INFO] Loading best checkpoint (epoch {stopper.best_epoch}, "
          f"val_loss={stopper.best_loss:.4f})")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    # ── Evaluate ─────────────────────────────────────────────────────────────
    results = [
        evaluate(model, test_same_loader,  device,
                 f"{args.model} | trained:{args.variety} → tested:{args.variety}"),
        evaluate(model, test_other_loader, device,
                 f"{args.model} | trained:{args.variety} → tested:{other_variety}"),
    ]

    # ── Save results.json ─────────────────────────────────────────────────────
    meta = {
        "run_name":        run_name,
        "variety":         args.variety,
        "model":           args.model,
        "optimizer":       args.optimizer,
        "lr":              args.lr,
        "batch_size":      args.batch_size,
        "dropout":         args.dropout,
        "max_epochs":      args.max_epochs,
        "best_epoch":      stopper.best_epoch,
        "best_val_loss":   stopper.best_loss,
        "pretrained_from": args.pretrained_from,
        "results":         results,
        "history":         history,
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