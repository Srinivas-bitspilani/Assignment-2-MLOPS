"""FastAPI inference service for the Cats vs Dogs classifier.

Endpoints
    GET  /health   - liveness/readiness probe; reports whether the model loaded
    POST /predict  - multipart image upload -> predicted label + class probabilities
    GET  /         - tiny service description

The model is loaded once during application startup (lifespan), never per
request, so /predict stays fast.

Run locally:
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
Interactive docs:
    http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.model_loader import ModelService, get_model_service  # noqa: E402
from api.preprocessing import InvalidImageError  # noqa: E402

API_VERSION = "1.0.0"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB: refuse absurd uploads early

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("api")


# --------------------------------------------------------------------------- #
# response schemas (these also generate the OpenAPI docs)
# --------------------------------------------------------------------------- #
class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    api_version: str
    model_loaded: bool
    classes: list[str] = Field(default_factory=list)
    image_size: int | None = None
    mlflow_run_id: str | None = None
    test_accuracy: float | None = None


class PredictionResponse(BaseModel):
    filename: str | None = None
    predicted_label: str = Field(..., examples=["dog"])
    predicted_index: int = Field(..., examples=[1])
    confidence: float = Field(..., examples=[0.9731])
    probabilities: dict[str, float] = Field(
        ..., examples=[{"cat": 0.0269, "dog": 0.9731}]
    )
    inference_time_ms: float


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup so the first request is not slow."""
    try:
        service = get_model_service()
        app.state.model_service = service
        logger.info(
            "Model loaded from %s (classes=%s, run_id=%s)",
            service.model_path,
            service.classes,
            service.mlflow_run_id,
        )
    except Exception as exc:  # keep serving so /health can report the problem
        app.state.model_service = None
        app.state.load_error = str(exc)
        logger.error("Model failed to load: %s", exc)
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Cats vs Dogs Classifier API",
    description=(
        "Binary image classification service for a pet adoption platform. "
        "Upload an image and get the predicted label with class probabilities."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)


def _service(require_loaded: bool = True) -> ModelService | None:
    service = getattr(app.state, "model_service", None)
    if require_loaded and service is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded: "
                f"{getattr(app.state, 'load_error', 'unknown error')}"
            ),
        )
    return service


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "service": "cats-vs-dogs-classifier",
        "version": API_VERSION,
        "endpoints": {"health": "GET /health", "predict": "POST /predict", "docs": "/docs"},
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> JSONResponse:
    """Liveness/readiness probe.

    Returns 200 with status 'ok' when the model is loaded and able to serve,
    503 with status 'degraded' when it is not, so Kubernetes can act on it.
    """
    service = _service(require_loaded=False)
    if service is None or not service.is_loaded:
        return JSONResponse(
            status_code=503,
            content=HealthResponse(
                status="degraded", api_version=API_VERSION, model_loaded=False
            ).model_dump(),
        )

    info = service.info()
    return JSONResponse(
        status_code=200,
        content=HealthResponse(
            status="ok",
            api_version=API_VERSION,
            model_loaded=True,
            classes=info["classes"],
            image_size=info["image_size"],
            mlflow_run_id=info["mlflow_run_id"],
            test_accuracy=info["test_accuracy"],
        ).model_dump(),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(file: UploadFile = File(..., description="Image file (JPEG/PNG)")):
    """Classify one uploaded image as cat or dog."""
    service = _service()

    image_bytes = await file.read()
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
        )

    try:
        result = service.predict(image_bytes)
    except InvalidImageError as exc:
        # A bad upload is the caller's fault -> 400, not a 500.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info(
        "predict filename=%s label=%s confidence=%.4f latency_ms=%.2f",
        file.filename,
        result["predicted_label"],
        result["confidence"],
        result["inference_time_ms"],
    )
    return PredictionResponse(filename=file.filename, **result)
