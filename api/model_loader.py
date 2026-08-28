"""Load the trained artifact once and serve predictions from it.

The checkpoint written by src/training/train.py is self-describing:

    state_dict     - the learned weights
    classes        - class order, e.g. ['cat', 'dog']  (index 0 = cat)
    model_config   - architecture hyperparameters
    data_config    - image size / channels / normalisation
    metrics        - test metrics of that run
    mlflow_run_id  - provenance: which MLflow run produced these weights

So the API rebuilds the identical architecture and the identical preprocessing
without reading params.yaml and without any hardcoded assumptions.

The model is loaded ONCE at startup (not per request) and kept in memory.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from threading import Lock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.preprocessing import preprocess  # noqa: E402
from src.models.factory import build_from_config  # noqa: E402

# Overridable so the container / tests can point at a different artifact.
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "model.pt"


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested before the model is available."""


class ModelService:
    """Holds the loaded model and turns image bytes into a prediction."""

    def __init__(self, model_path: Path | str | None = None):
        self.model_path = Path(
            model_path or os.getenv("MODEL_PATH") or DEFAULT_MODEL_PATH
        )
        self.model: BaselineCNN | None = None
        self.classes: list[str] = []
        self.data_config: dict = {}
        self.metrics: dict = {}
        self.mlflow_run_id: str | None = None
        self.loaded_at: float | None = None
        self._lock = Lock()

    # ------------------------------------------------------------------ #
    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> "ModelService":
        """Read the checkpoint and put the model into eval mode."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. "
                "Run `python train.py` first."
            )

        # weights_only=False: our checkpoint stores config dicts, not just tensors.
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)

        model_config = dict(checkpoint["model_config"])
        self.classes = list(checkpoint["classes"])
        self.data_config = checkpoint["data_config"]
        self.metrics = checkpoint.get("metrics", {})
        self.mlflow_run_id = checkpoint.get("mlflow_run_id")

        # Which architecture the promoted champion actually is. Older
        # checkpoints predate this field, so fall back to the baseline.
        self.architecture = checkpoint.get(
            "architecture", model_config.get("architecture", "baseline_cnn")
        )
        model_config["architecture"] = self.architecture

        # pretrained=False: the weights come from the checkpoint, so the
        # container must never try to download ImageNet weights at startup.
        model_config["pretrained"] = False

        model = build_from_config(model_config, self.data_config)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()  # disables dropout / freezes batch-norm statistics

        # Serving is single-request-at-a-time on CPU; one thread avoids
        # oversubscribing the container's CPU limit.
        torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))

        self.model = model
        self.loaded_at = time.time()
        return self

    # ------------------------------------------------------------------ #
    def predict(self, image_bytes: bytes) -> dict:
        """Classify one image.

        Returns the predicted label, its confidence, and the full probability
        distribution over every class.
        """
        if self.model is None:
            raise ModelNotLoadedError("Model is not loaded")

        tensor = preprocess(image_bytes, self.data_config)

        started = time.perf_counter()
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
        inference_ms = (time.perf_counter() - started) * 1000

        index = int(probabilities.argmax().item())
        return {
            "predicted_label": self.classes[index],
            "predicted_index": index,
            "confidence": float(probabilities[index].item()),
            "probabilities": {
                name: float(probabilities[i].item())
                for i, name in enumerate(self.classes)
            },
            "inference_time_ms": round(inference_ms, 2),
        }

    def info(self) -> dict:
        """Metadata for the /health endpoint."""
        return {
            "model_path": str(self.model_path),
            "model_loaded": self.is_loaded,
            "architecture": getattr(self, "architecture", None),
            "classes": self.classes,
            "image_size": self.data_config.get("image_size"),
            "mlflow_run_id": self.mlflow_run_id,
            "test_accuracy": self.metrics.get("test_accuracy"),
        }


# --------------------------------------------------------------------------- #
# process-wide singleton
# --------------------------------------------------------------------------- #
_service: ModelService | None = None
_service_lock = Lock()


def get_model_service(model_path: Path | str | None = None) -> ModelService:
    """Return the shared ModelService, loading it on first use."""
    global _service
    with _service_lock:
        if _service is None:
            _service = ModelService(model_path).load()
    return _service


def reset_model_service() -> None:
    """Drop the cached instance (used by tests)."""
    global _service
    with _service_lock:
        _service = None
