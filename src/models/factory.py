"""Model factory: build any supported architecture from a config dict.

The registry holds more than one kind of model (a from-scratch baseline and a
fine-tuned ResNet18), and the serving code must be able to rebuild whichever
one was promoted to champion. Every checkpoint therefore records its
`architecture`, and this factory is the single place that maps that string to
a concrete nn.Module.

Supported architectures
    baseline_cnn        - src/models/cnn.py, 389k params, trained from scratch
    resnet18_finetune   - torchvision ResNet18 with ImageNet weights, new
                          2-class head. With freeze_backbone=true only the
                          final layer trains, which is what makes this feasible
                          on CPU: the backward pass touches ~1k parameters
                          instead of 11.7M.
"""

from __future__ import annotations

import sys
from pathlib import Path

from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.cnn import BaselineCNN  # noqa: E402

BASELINE = "baseline_cnn"
RESNET18 = "resnet18_finetune"
SUPPORTED = (BASELINE, RESNET18)


def build_resnet18(
    num_classes: int = 2,
    freeze_backbone: bool = True,
    pretrained: bool = True,
) -> nn.Module:
    """ResNet18 with an ImageNet backbone and a fresh classification head."""
    from torchvision import models

    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    # Replacing fc after freezing means the new head is always trainable.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_from_config(model_cfg: dict, data_cfg: dict) -> nn.Module:
    """Build the architecture named in model_cfg['architecture'].

    Defaults to the baseline so older checkpoints, which predate this field,
    still load correctly.
    """
    architecture = model_cfg.get("architecture", BASELINE)

    if architecture == BASELINE:
        return BaselineCNN(
            num_classes=int(model_cfg["num_classes"]),
            conv_blocks=int(model_cfg["conv_blocks"]),
            base_filters=int(model_cfg["base_filters"]),
            dropout=float(model_cfg["dropout"]),
            in_channels=int(data_cfg.get("channels", 3)),
        )

    if architecture == RESNET18:
        return build_resnet18(
            num_classes=int(model_cfg["num_classes"]),
            freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
            pretrained=bool(model_cfg.get("pretrained", True)),
        )

    raise ValueError(
        f"Unknown architecture {architecture!r}. Supported: {SUPPORTED}"
    )


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
