"""Unit tests for the preprocessing pipeline (M1 offline + M2 inference-time).

Two things are worth testing here, because both are classic silent-failure
sources in image pipelines:

  1. Geometry: every image must come out exactly 224x224 RGB, whatever went in.
  2. Train/serve consistency: the transform used by the API must match the
     evaluation transform used during training, and must NOT augment.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from api.preprocessing import (
    InvalidImageError,
    build_inference_transform,
    load_image,
    preprocess,
)
from src.data.dataset import build_transforms
from src.data.preprocess import resize_center_crop, split_indices
from tests.conftest import make_image_bytes


# --------------------------------------------------------------------------- #
# M1: offline resize
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "in_size",
    [(500, 375), (375, 500), (224, 224), (64, 64), (1200, 300), (300, 1200)],
)
def test_resize_center_crop_always_224_rgb(in_size):
    """Any input geometry -> exactly 224x224 RGB, including upscales."""
    image = Image.new("RGB", in_size, (10, 200, 30))
    result = resize_center_crop(image, 224)
    assert result.size == (224, 224)
    assert result.mode == "RGB"


def test_resize_center_crop_converts_grayscale_to_rgb():
    result = resize_center_crop(Image.new("L", (300, 300), 128), 224)
    assert result.mode == "RGB"
    assert len(result.split()) == 3


def test_resize_center_crop_preserves_aspect_ratio():
    """A centred square must stay square, not be stretched.

    A 400x200 image with a centred 200x200 red square: after short-side resize
    + centre crop the red region must still be square-ish, which is only true
    if the aspect ratio was preserved.
    """
    image = Image.new("RGB", (400, 200), (255, 255, 255))
    image.paste(Image.new("RGB", (200, 200), (255, 0, 0)), (100, 0))

    array = np.asarray(resize_center_crop(image, 224))
    red = (array[:, :, 0] > 200) & (array[:, :, 1] < 80)
    rows = np.where(red.any(axis=1))[0]
    cols = np.where(red.any(axis=0))[0]
    height = rows[-1] - rows[0] + 1
    width = cols[-1] - cols[0] + 1

    # Squashing 400x200 into a square would double the width relative to height.
    assert abs(height - width) / max(height, width) < 0.1


# --------------------------------------------------------------------------- #
# M1: the 80/10/10 split
# --------------------------------------------------------------------------- #
def test_split_sizes_are_80_10_10(params):
    ratios = params["data"]["split"]
    result = split_indices(2000, ratios, seed=42)
    assert len(result["train"]) == 1600
    assert len(result["val"]) == 200
    assert len(result["test"]) == 200


def test_split_is_deterministic_for_a_fixed_seed(params):
    ratios = params["data"]["split"]
    assert split_indices(500, ratios, 42) == split_indices(500, ratios, 42)


def test_split_changes_with_a_different_seed(params):
    ratios = params["data"]["split"]
    assert split_indices(500, ratios, 42) != split_indices(500, ratios, 7)


def test_split_is_disjoint_and_lossless(params):
    """No image may appear in two splits, and none may be dropped."""
    ratios = params["data"]["split"]
    n = 997  # deliberately not divisible by 10
    result = split_indices(n, ratios, seed=42)

    train, val, test = set(result["train"]), set(result["val"]), set(result["test"])
    assert not (train & val) and not (train & test) and not (val & test)
    assert train | val | test == set(range(n))


# --------------------------------------------------------------------------- #
# M2: inference-time transform
# --------------------------------------------------------------------------- #
def test_preprocess_output_shape_and_dtype(jpeg_bytes, data_config):
    tensor = preprocess(jpeg_bytes, data_config)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32


def test_preprocess_applies_normalisation(data_config):
    """A mid-grey image must land near the normalised value, not near 0.5.

    This is the test that catches a missing Normalize step -- the single most
    damaging train/serve mismatch possible.
    """
    tensor = preprocess(make_image_bytes(color=(128, 128, 128)), data_config)
    mean = torch.tensor(data_config["normalize"]["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(data_config["normalize"]["std"]).view(1, 3, 1, 1)
    expected = ((128 / 255) - mean) / std
    assert torch.allclose(tensor.mean(dim=(0, 2, 3)), expected.flatten(), atol=0.02)


def test_inference_transform_is_deterministic(data_config):
    """Serving must never augment: same bytes in -> identical tensor out."""
    image_bytes = make_image_bytes()
    assert torch.equal(
        preprocess(image_bytes, data_config), preprocess(image_bytes, data_config)
    )


def test_train_transform_is_random_but_eval_transform_is_not(params):
    """Augmentation must be live for train and absent for val/test."""
    image = Image.new("RGB", (300, 300), (100, 150, 200))

    train_transform = build_transforms(params, train=True)
    assert not torch.equal(train_transform(image), train_transform(image))

    eval_transform = build_transforms(params, train=False)
    assert torch.equal(eval_transform(image), eval_transform(image))


def test_train_and_inference_transforms_agree_on_output_shape(params, data_config):
    """The API transform and the training eval transform must be interchangeable."""
    image = Image.new("RGB", (640, 480), (10, 20, 30))
    from_training = build_transforms(params, train=False)(image)
    from_api = build_inference_transform(data_config)(image)
    assert from_training.shape == from_api.shape
    assert torch.allclose(from_training, from_api, atol=1e-6)


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "BMP"])
def test_preprocess_accepts_common_formats(fmt, data_config):
    tensor = preprocess(make_image_bytes(fmt=fmt), data_config)
    assert tensor.shape == (1, 3, 224, 224)


def test_preprocess_handles_grayscale(data_config):
    tensor = preprocess(make_image_bytes(mode="L"), data_config)
    assert tensor.shape[1] == 3  # expanded to 3 channels


# --------------------------------------------------------------------------- #
# M2: bad input handling
# --------------------------------------------------------------------------- #
def test_load_image_rejects_non_image_bytes():
    with pytest.raises(InvalidImageError):
        load_image(b"definitely not an image")


def test_load_image_rejects_empty_upload():
    with pytest.raises(InvalidImageError):
        load_image(b"")


def test_load_image_rejects_truncated_jpeg():
    """Half a JPEG must raise, not silently decode to garbage."""
    full = make_image_bytes()
    with pytest.raises(InvalidImageError):
        load_image(full[: len(full) // 2])
