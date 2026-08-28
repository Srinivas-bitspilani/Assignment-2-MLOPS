"""Unit tests for the model and the inference API (M1 model + M2 service).

Covers three layers:
  1. the nn.Module itself  - shapes, probability validity, eval determinism
  2. ModelService          - checkpoint loading and the predict() contract
  3. the HTTP surface      - GET /health and POST /predict, success and failure
"""

from __future__ import annotations

import pytest
import torch

from torch import nn

from api.model_loader import ModelService, ModelNotLoadedError
from src.models.cnn import build_model, count_parameters
from src.models.factory import SUPPORTED
from tests.conftest import make_image_bytes


# --------------------------------------------------------------------------- #
# 1. the model
# --------------------------------------------------------------------------- #
def test_model_output_shape(params):
    model = build_model(params).eval()
    with torch.no_grad():
        logits = model(torch.randn(4, 3, 224, 224))
    assert logits.shape == (4, 2)


def test_model_probabilities_are_valid(params):
    model = build_model(params).eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(torch.randn(8, 3, 224, 224)), dim=1)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(8), atol=1e-5)
    assert (probabilities >= 0).all() and (probabilities <= 1).all()


def test_model_is_deterministic_in_eval_mode(params):
    """Dropout must be off in eval mode, or the API would return
    a different answer for the same image on every call."""
    model = build_model(params).eval()
    batch = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.equal(model(batch), model(batch))


def test_model_is_stochastic_in_train_mode(params):
    """The mirror image of the test above: dropout must be active for training."""
    model = build_model(params).train()
    batch = torch.randn(4, 3, 224, 224)
    assert not torch.equal(model(batch), model(batch))


def test_model_accepts_a_single_image_batch(params):
    model = build_model(params).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 224, 224)).shape == (1, 2)


def test_model_parameter_count_is_reasonable(params):
    """Guards against an accidental architecture change (e.g. losing the
    global-average-pool head, which would balloon this to ~25M)."""
    count = count_parameters(build_model(params))
    assert 300_000 < count < 500_000


def test_model_is_input_size_agnostic(params):
    """AdaptiveAvgPool means a non-224 input must not crash."""
    model = build_model(params).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 160, 160)).shape == (1, 2)


# --------------------------------------------------------------------------- #
# 2. ModelService
# --------------------------------------------------------------------------- #
def test_service_loads_checkpoint(checkpoint_path, classes):
    """The service must load whichever architecture was promoted to champion.

    Deliberately architecture-agnostic: asserting BaselineCNN here would fail
    the moment a better model (e.g. resnet18_finetune) is promoted, which is a
    normal event, not a regression.
    """
    service = ModelService(checkpoint_path).load()
    assert service.is_loaded
    assert service.classes == classes
    assert isinstance(service.model, nn.Module)
    assert service.architecture in SUPPORTED
    assert not service.model.training  # eval mode


def test_service_class_order_is_cat_then_dog(checkpoint_path):
    """cat must be index 0 and dog index 1. If this ever flips, every
    prediction the service returns is inverted."""
    service = ModelService(checkpoint_path).load()
    assert service.classes[0] == "cat"
    assert service.classes[1] == "dog"


def test_service_raises_for_missing_artifact(tmp_path):
    with pytest.raises(FileNotFoundError):
        ModelService(tmp_path / "does_not_exist.pt").load()


def test_service_predict_before_load_raises(checkpoint_path):
    with pytest.raises(ModelNotLoadedError):
        ModelService(checkpoint_path).predict(make_image_bytes())


def test_service_predict_contract(checkpoint_path, classes):
    service = ModelService(checkpoint_path).load()
    result = service.predict(make_image_bytes())

    assert set(result) == {
        "predicted_label",
        "predicted_index",
        "confidence",
        "probabilities",
        "inference_time_ms",
    }
    assert result["predicted_label"] in classes
    assert set(result["probabilities"]) == set(classes)
    assert result["probabilities"][result["predicted_label"]] == pytest.approx(
        result["confidence"]
    )
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["predicted_index"] == classes.index(result["predicted_label"])


def test_service_predictions_are_repeatable(checkpoint_path):
    service = ModelService(checkpoint_path).load()
    image_bytes = make_image_bytes()
    first = service.predict(image_bytes)
    second = service.predict(image_bytes)
    assert first["probabilities"] == second["probabilities"]


# --------------------------------------------------------------------------- #
# 3. HTTP endpoints
# --------------------------------------------------------------------------- #
def test_health_returns_ok(client, classes):
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["classes"] == classes
    assert body["image_size"] == 224
    assert "api_version" in body


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "cats-vs-dogs-classifier"


def test_predict_returns_label_and_probabilities(client, classes):
    response = client.post(
        "/predict", files={"file": ("test.jpg", make_image_bytes(), "image/jpeg")}
    )
    assert response.status_code == 200

    body = response.json()
    assert body["predicted_label"] in classes
    assert set(body["probabilities"]) == set(classes)
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert body["filename"] == "test.jpg"
    assert body["inference_time_ms"] > 0


@pytest.mark.parametrize("fmt", ["JPEG", "PNG"])
def test_predict_accepts_multiple_formats(client, fmt):
    response = client.post(
        "/predict", files={"file": (f"t.{fmt.lower()}", make_image_bytes(fmt=fmt), None)}
    )
    assert response.status_code == 200


def test_predict_rejects_non_image_with_400(client):
    response = client.post(
        "/predict", files={"file": ("bad.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400
    assert "decode" in response.json()["detail"].lower()


def test_predict_rejects_empty_file_with_400(client):
    response = client.post("/predict", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert response.status_code == 400


def test_predict_requires_a_file_field(client):
    """A malformed request must be a 422, not a 500."""
    assert client.post("/predict").status_code == 422


def test_openapi_documents_both_endpoints(client):
    schema = client.get("/openapi.json").json()
    assert "/health" in schema["paths"]
    assert "/predict" in schema["paths"]
