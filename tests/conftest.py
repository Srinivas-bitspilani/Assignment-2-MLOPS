"""Shared pytest fixtures.

Design rule for this suite: **no test may depend on data/ or on a trained
model existing.** CI checks out the repo without DVC-tracked data, so tests
build synthetic images with PIL and fall back to a freshly-initialised
checkpoint when artifacts/model.pt is not present. That keeps the pipeline
green on a clean clone while still testing the real code paths.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_params  # noqa: E402
from src.models.cnn import build_model  # noqa: E402


@pytest.fixture(scope="session")
def params() -> dict:
    return load_params()


@pytest.fixture(scope="session")
def classes(params) -> list:
    return list(params["data"]["classes"])


def make_image_bytes(
    size: tuple[int, int] = (300, 240),
    color: tuple[int, int, int] = (120, 140, 160),
    mode: str = "RGB",
    fmt: str = "JPEG",
) -> bytes:
    """Build an in-memory image. Used everywhere instead of reading data/."""
    if mode == "L":
        image = Image.new("L", size, color[0])
    else:
        image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, fmt)
    return buffer.getvalue()


@pytest.fixture
def jpeg_bytes() -> bytes:
    return make_image_bytes()


@pytest.fixture(scope="session")
def checkpoint_path(tmp_path_factory, params) -> Path:
    """Path to a usable checkpoint.

    Prefers the real trained artifact so CI tests the actual shipped model;
    creates an untrained one with identical structure when it is missing.
    """
    real = PROJECT_ROOT / params["model"]["artifact_dir"] / params["model"]["artifact_name"]
    if real.exists():
        return real

    path = tmp_path_factory.mktemp("model") / "model.pt"
    torch.save(
        {
            "state_dict": build_model(params).state_dict(),
            "classes": list(params["data"]["classes"]),
            "model_config": params["model"],
            "data_config": {
                "image_size": params["data"]["image_size"],
                "channels": params["data"]["channels"],
                "normalize": params["data"]["normalize"],
            },
            "metrics": {"test_accuracy": None},
            "mlflow_run_id": "untrained-fixture",
        },
        path,
    )
    return path


@pytest.fixture(scope="session")
def data_config(params) -> dict:
    return {
        "image_size": params["data"]["image_size"],
        "channels": params["data"]["channels"],
        "normalize": params["data"]["normalize"],
    }


@pytest.fixture
def client(checkpoint_path, monkeypatch):
    """TestClient wired to the fixture checkpoint via MODEL_PATH."""
    from fastapi.testclient import TestClient

    import api.model_loader as model_loader

    monkeypatch.setenv("MODEL_PATH", str(checkpoint_path))
    model_loader.reset_model_service()  # drop any singleton from a previous test

    from api.main import app

    with TestClient(app) as test_client:
        yield test_client

    model_loader.reset_model_service()
