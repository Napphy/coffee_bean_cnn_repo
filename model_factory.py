"""
model_factory.py
================================================================
Unified model + preprocessing factory for the coffee defect
classifier pipeline. Adds MobileNetV1 (via `timm`, using
Ross Wightman's timm/mobilenetv1_100.ra4_e3600_r224_in1k
checkpoint) alongside the existing torchvision models
(squeezenet, shufflenet, mobilenetv2, mobilenetv3, efficientnet).

Usage in train.py:

    from model_factory import get_model, get_transforms

    model = get_model(args.model, num_classes=3, dropout=args.dropout)
    train_tf, eval_tf = get_transforms(args.model)

`get_model` always returns an nn.Module whose forward(x) returns
raw logits of shape (batch, num_classes) - same contract regardless
of whether the backbone came from torchvision or timm, so the rest
of train.py (loss, optimizer, eval loop) doesn't need to know the
difference.

Install requirement for the new path:
    pip install timm
================================================================
"""

import torch
import torchvision.models as tvm
import torchvision.transforms as T
import torch.nn as nn

TORCHVISION_MODELS = {
    "squeezenet",
    "shufflenet",
    "mobilenetv2",
    "mobilenetv3",
    "efficientnet",
}

TIMM_MODELS = {
    # our short name -> timm checkpoint id
    "mobilenetv1": "mobilenetv1_100.ra4_e3600_r224_in1k",
    # trained w/ extra head hidden layer variant, if you want to compare:
    "mobilenetv1_h": "mobilenetv1_100h.ra4_e3600_r224_in1k",
    # wider variant:
    "mobilenetv1_125": "mobilenetv1_125.ra4_e3600_r224_in1k",
}


# ─────────────────────────────────────────────────────────────
# torchvision path (existing models) — unchanged behavior
# ─────────────────────────────────────────────────────────────
def _build_torchvision_model(name: str, num_classes: int, dropout: float, pretrained: bool):
    if name == "squeezenet":
        m = tvm.squeezenet1_1(weights=tvm.SqueezeNet1_1_Weights.DEFAULT if pretrained else None)
        m.classifier[0] = nn.Dropout(p=dropout)
        m.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        m.num_classes = num_classes

    elif name == "shufflenet":
        m = tvm.shufflenet_v2_x1_0(weights=tvm.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None)
        in_features = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))

    elif name == "mobilenetv2":
        m = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        in_features = m.classifier[-1].in_features
        m.classifier[0] = nn.Dropout(p=dropout)
        m.classifier[-1] = nn.Linear(in_features, num_classes)

    elif name == "mobilenetv3":
        m = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        in_features = m.classifier[-1].in_features
        m.classifier[-2] = nn.Dropout(p=dropout)
        m.classifier[-1] = nn.Linear(in_features, num_classes)

    elif name == "efficientnet":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_features = m.classifier[-1].in_features
        m.classifier[0] = nn.Dropout(p=dropout)
        m.classifier[-1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown torchvision model: {name}")

    return m


# ─────────────────────────────────────────────────────────────
# timm path (MobileNetV1) — new
# ─────────────────────────────────────────────────────────────
def _build_timm_model(name: str, num_classes: int, dropout: float, pretrained: bool):
    import timm

    checkpoint = TIMM_MODELS[name]
    # timm handles head replacement + dropout natively via num_classes/
    # drop_rate, so no manual pooling/wrapper module is needed (unlike
    # the earlier HF `transformers` approach).
    model = timm.create_model(
        checkpoint,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
    )
    return model


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
def get_model(name: str, num_classes: int = 3, dropout: float = 0.3, pretrained: bool = True) -> nn.Module:
    """Build a model by name, returning an nn.Module with forward(x) -> logits."""
    if name in TORCHVISION_MODELS:
        return _build_torchvision_model(name, num_classes, dropout, pretrained)
    elif name in TIMM_MODELS:
        return _build_timm_model(name, num_classes, dropout, pretrained)
    else:
        known = sorted(TORCHVISION_MODELS | TIMM_MODELS.keys())
        raise ValueError(f"Unknown model '{name}'. Known models: {known}")


def get_transforms(name: str, train: bool = True):
    """
    Returns (train_transform, eval_transform) as torchvision.transforms
    -compatible callables, so the rest of the data-loading code stays
    unchanged (Dataset/DataLoader don't need to know which backend built
    the model).

    For timm models, normalization/resize/crop settings come straight from
    the checkpoint's own pretrained_cfg (via timm.data.resolve_model_data_config
    + timm.data.create_transform) rather than being guessed - this matters
    for MobileNetV1 specifically, since its config uses mean=std=(0.5,0.5,0.5)
    rather than the standard ImageNet stats used by the torchvision models.
    """
    if name in TORCHVISION_MODELS:
        # import torchvision.transforms as T

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # torchvision ImageNet defaults
        size = 224

        train_tf = T.Compose([
            T.RandomResizedCrop(size),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        eval_tf = T.Compose([
            T.Resize(int(size * 1.14)),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return train_tf, eval_tf

    elif name in TIMM_MODELS:
        import timm
        import timm.data

        # pretrained=False here just reads the checkpoint's pretrained_cfg
        # (architecture + normalization/resize metadata) without downloading
        # the actual weights - cheap, and avoids fetching weights twice.
        probe_model = timm.create_model(TIMM_MODELS[name], pretrained=False)
        data_config = timm.data.resolve_model_data_config(probe_model)

        base_train_tf = timm.data.create_transform(**data_config, is_training=True)
        train_tf = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            base_train_tf,
        ])
        eval_tf = timm.data.create_transform(**data_config, is_training=False)
        return train_tf, eval_tf

    else:
        raise ValueError(f"Unknown model '{name}'")


if __name__ == "__main__":
    # Quick smoke test
    for model_name in sorted(TORCHVISION_MODELS | TIMM_MODELS.keys()):
        model = get_model(model_name, num_classes=3, dropout=0.3, pretrained=True)
        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (2, 3), f"{model_name}: unexpected output shape {out.shape}"
        print(f"[OK] {model_name}: logits shape {tuple(out.shape)}")