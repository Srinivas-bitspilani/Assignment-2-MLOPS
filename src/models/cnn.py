"""Baseline CNN for binary Cats vs Dogs classification.

Deliberately trained from scratch (no pretrained backbone) -- the assignment
asks for a *baseline* CNN, and a from-scratch model is the honest baseline that
later improvements can be measured against.

Architecture (all sizes driven by params.yaml -> model):

    input  3 x 224 x 224
    block1 Conv3x3(32)  + BN + ReLU + MaxPool2  ->  32 x 112 x 112
    block2 Conv3x3(64)  + BN + ReLU + MaxPool2  ->  64 x  56 x  56
    block3 Conv3x3(128) + BN + ReLU + MaxPool2  -> 128 x  28 x  28
    block4 Conv3x3(256) + BN + ReLU + MaxPool2  -> 256 x  14 x  14
    head   AdaptiveAvgPool -> Flatten -> Dropout(0.5) -> Linear(256 -> 2)

Design notes:
  * BatchNorm after every conv: lets us train from scratch at lr=1e-3 without
    the loss diverging in the first epochs.
  * AdaptiveAvgPool2d(1) instead of a big flatten: keeps the classifier at
    ~0.5k parameters instead of ~25M, which massively reduces overfitting on a
    3200-image training set and makes the model input-size agnostic.
  * Two output logits (not one sigmoid): pairs with nn.CrossEntropyLoss and
    gives the API a clean per-class softmax probability vector in M2.

Run:  python src/models/cnn.py     (architecture summary + forward-pass test)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import load_params  # noqa: E402


class BaselineCNN(nn.Module):
    """Simple VGG-style stack of conv blocks with a global-average-pool head."""

    def __init__(
        self,
        num_classes: int = 2,
        conv_blocks: int = 4,
        base_filters: int = 32,
        dropout: float = 0.5,
        in_channels: int = 3,
    ):
        super().__init__()
        self.num_classes = num_classes

        layers: list[nn.Module] = []
        channels = in_channels
        for block in range(conv_blocks):
            out_channels = base_filters * (2 ** block)   # 32, 64, 128, 256
            layers += [
                nn.Conv2d(channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            channels = out_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """He initialisation -- the right choice for ReLU networks."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape (batch, num_classes)."""
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_model(params: dict | None = None) -> BaselineCNN:
    """Construct the model described by params.yaml -> model."""
    params = params or load_params()
    cfg = params["model"]
    return BaselineCNN(
        num_classes=int(cfg["num_classes"]),
        conv_blocks=int(cfg["conv_blocks"]),
        base_filters=int(cfg["base_filters"]),
        dropout=float(cfg["dropout"]),
        in_channels=int(params["data"]["channels"]),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _sanity_check() -> None:
    params = load_params()
    size = int(params["data"]["image_size"])
    model = build_model(params)

    print(model)
    print(f"\ntrainable parameters : {count_parameters(model):,}")

    # Shape test with the exact batch shape the dataloaders produce.
    batch = torch.randn(4, params["data"]["channels"], size, size)
    model.eval()
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)

    print(f"input  shape         : {tuple(batch.shape)}")
    print(f"logits shape         : {tuple(logits.shape)}   (expected (4, 2))")
    print(f"softmax row sums     : {probs.sum(dim=1).tolist()}  (expected all 1.0)")
    print(f"predicted classes    : "
          f"{[params['data']['classes'][i] for i in probs.argmax(1).tolist()]}")

    # Feature-map sizes, block by block.
    print("\nfeature map trace:")
    x = batch
    stage = 0
    for layer in model.features:
        x = layer(x)
        if isinstance(layer, nn.MaxPool2d):
            stage += 1
            print(f"  after block {stage}: {tuple(x.shape[1:])}")


if __name__ == "__main__":
    _sanity_check()
