"""Inference-time image preprocessing for the FastAPI service.

This must mirror the *evaluation* transform used during training
(src/data/dataset.py -> build_transforms(train=False)):

    decode -> RGB -> resize to 224x224 -> ToTensor -> ImageNet normalise

The parameters are not hardcoded here: they are read from the `data_config`
block stored inside artifacts/model.pt. That way the served preprocessing can
never silently drift away from the preprocessing the model was trained with.

Deliberately NO augmentation: augmentation is a training-only device.
"""

from __future__ import annotations

import io

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms


class InvalidImageError(ValueError):
    """Raised when the uploaded bytes are not a decodable image."""


def build_inference_transform(data_config: dict) -> transforms.Compose:
    """Build the deterministic transform described by the checkpoint."""
    size = int(data_config["image_size"])
    normalize = data_config["normalize"]
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize["mean"], std=normalize["std"]),
        ]
    )


def load_image(image_bytes: bytes) -> Image.Image:
    """Decode raw upload bytes into an RGB PIL image.

    Raises InvalidImageError for anything that is not a real image, so the API
    can answer 400 instead of leaking a stack trace.
    """
    if not image_bytes:
        raise InvalidImageError("Empty file upload")
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()          # force a full decode: catches truncated files
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(f"Could not decode image: {exc}") from exc
    return image.convert("RGB")


def preprocess(image_bytes: bytes, data_config: dict) -> torch.Tensor:
    """Turn upload bytes into a model-ready batch of shape (1, 3, H, W)."""
    image = load_image(image_bytes)
    transform = build_inference_transform(data_config)
    return transform(image).unsqueeze(0)
