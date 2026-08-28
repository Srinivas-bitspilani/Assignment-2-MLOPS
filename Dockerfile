# Serving image for the Cats vs Dogs FastAPI inference service.
#
# Two deliberate choices keep this image small and reproducible:
#   1. torch/torchvision come from the PyTorch CPU wheel index. The default
#      PyPI wheels bundle CUDA and add ~2 GB we can never use in this service.
#   2. Dependencies are installed BEFORE the source is copied, so editing code
#      does not invalidate the (slow) dependency layer on rebuild.
#
# Build:  docker build -t cats-dogs-api:latest .
# Run:    docker run --rm -p 8000:8000 cats-dogs-api:latest

FROM python:3.12-slim

# Python behaves better in containers with these set.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/artifacts/model.pt \
    TORCH_NUM_THREADS=1

WORKDIR /app

# ---- dependencies (cached layer) ----------------------------------------- #
COPY requirements-api.txt .
RUN pip install --upgrade pip && \
    pip install \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements-api.txt

# ---- application code ---------------------------------------------------- #
# Only what serving actually needs. src/data and src/training are excluded by
# .dockerignore because the container never trains.
COPY src/__init__.py src/config.py ./src/
COPY src/models/ ./src/models/
COPY api/ ./api/
COPY params.yaml ./
COPY artifacts/model.pt ./artifacts/model.pt

# ---- run as a non-root user ---------------------------------------------- #
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Container-level health check. python is used instead of curl because
# python:slim has no curl and we are not installing one just for this.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
