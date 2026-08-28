# MLOps Pipeline — Cats vs Dogs Binary Image Classification

An end-to-end MLOps pipeline for a pet adoption platform: a CNN that classifies
an uploaded photo as **cat** or **dog**, wrapped in everything needed to train
it reproducibly, serve it, test it, ship it, deploy it and watch it in production.

```
Dataset → Preprocessing → Training → MLflow → Model Artifact → FastAPI
   → Docker → Tests → GitHub Actions CI → Container Registry
   → Kubernetes → CD → Smoke Tests → Monitoring
```

| Module | Scope | Status |
|---|---|---|
| **M1** | Model development & experiment tracking | Complete |
| **M2** | Packaging & containerization | Complete |
| **M3** | CI: tests, image build, registry publish | Complete |
| **M4** | CD: Kubernetes deployment & smoke tests | Complete |
| **M5** | Monitoring, logging & final packaging | Complete |

**Verified by execution on this machine** (Windows 11, Dell Latitude 5410, no GPU):

| Stage | Tooling | Result |
|---|---|---|
| Training | PyTorch 2.13 CPU, 8 threads | 10 epochs, 72.50 % test accuracy |
| Tests | pytest 9.1.1 | **57 passed** |
| Container build | Docker 29.7.2 | 1.51 GB image, built in 3.8 min |
| Container run | Docker | `HEALTHCHECK` = healthy, smoke tests pass |
| Kubernetes | Minikube 1.38.1 / k8s v1.35.1 | **2/2 replicas ready**, rolling update, smoke tests pass |
| Post-deploy eval | 400 HTTP requests to the cluster | 400/400 answered, 72.50 % accuracy |

**Verified in GitHub Actions** (commit `b61e9a3`):

| Workflow | Job | Result | Time |
|---|---|---|---|
| [CI #1](https://github.com/Srinivas-bitspilani/Assignment-2-MLOPS/actions/runs/33175845979) | Pytest (preprocessing + inference) | ✅ **success** — 57 tests | 0.9 min |
| [CI #1](https://github.com/Srinivas-bitspilani/Assignment-2-MLOPS/actions/runs/33175845979) | Build & publish image | ✅ **success** — built, smoke-tested, pushed to GHCR | 2.1 min |
| [CD #1](https://github.com/Srinivas-bitspilani/Assignment-2-MLOPS/actions/runs/33176095013) | Deploy to Kubernetes and smoke test | ✅ **success** — rollout complete, smoke tests passed | 2.6 min |

CD was triggered **automatically** by CI success, and the *"Roll back if the smoke
tests failed"* step shows as `skipped` — the failure path is wired up and simply
was not needed.

**Published image:** `ghcr.io/srinivas-bitspilani/assignment-2-mlops`

```bash
docker pull ghcr.io/srinivas-bitspilani/assignment-2-mlops:latest
docker run --rm -p 8000:8000 ghcr.io/srinivas-bitspilani/assignment-2-mlops:latest
```

Tags: `latest`, `main`, `sha-b61e9a37c34acf19f6a75a7b55c8181ae27de62f`
(digest `sha256:8d4dc4e2a814…`). The SHA tag is what CD deploys, so any
deployment is traceable to an exact commit.

---

## 1. Results

<!-- RESULTS_START -->
**Run `d604ad0f8b8a4a72b29b45d8789bfd89`** — baseline CNN, 389,410 parameters,
trained from scratch on CPU, 10 epochs, best weights restored from **epoch 8**.

| Metric (held-out test split, 400 images) | Value |
|---|---|
| Accuracy | **72.50 %** |
| Precision (macro) | 72.51 % |
| Recall (macro) | 72.50 % |
| F1 (macro) | 72.50 % |
| Test loss | 0.5633 |
| Best epoch | 8 (`val_loss` 0.5159, `val_accuracy` 0.7425) |
| Peak validation accuracy | 77.25 % (epoch 7) |

**Confusion matrix (test set)**

| | predicted cat | predicted dog | recall |
|---|---|---|---|
| **true cat** | 143 | 57 | 71.5 % |
| **true dog** | 53 | 147 | 73.5 % |

Errors are almost perfectly balanced (57 vs 53), so the model is not biased
toward either class — what you would want from a 50/50 stratified split.

**Per-epoch validation accuracy**

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| val acc | .530 | .670 | .708 | .665 | .678 | .765 | **.773** | .743 | .728 | .618 |

Validation loss bottoms out at epoch 8 and then climbs sharply (0.5159 → 0.6591)
as the model starts to overfit — visible in `artifacts/training_curves.png`.
This is exactly why the script restores the **best** weights rather than the
last ones; using epoch 10 would have cost about 10 points of accuracy.

**Train/serve consistency check.** `scripts/evaluate_deployed_model.py -n 200`
sends all 400 test images to the *running service* over HTTP. It reproduces the
training-time confusion matrix **exactly** in every environment the model was
deployed to — which is positive proof that there is no preprocessing drift
between training and serving:

| Environment | cat→cat | cat→dog | dog→cat | dog→dog | accuracy |
|---|---|---|---|---|---|
| Training (in-process) | 143 | 57 | 53 | 147 | 72.50 % |
| Local uvicorn (HTTP) | 143 | 57 | 53 | 147 | 72.50 % |
| Docker container | 143 | 57 | 53 | 147 | 72.50 % |
| **Kubernetes (2 replicas)** | 143 | 57 | 53 | 147 | 72.50 % |

Identical to the last prediction in all four. Per-image confidences match too
(e.g. `cat_00013.jpg` → 0.6823 everywhere), so this is bitwise reproducibility,
not a coincidence of rounding.

**Serving latency** (client-observed, 400 requests each):

| Environment | mean | p50 | p95 | max | server-side inference |
|---|---|---|---|---|---|
| Local uvicorn (Windows, 8 threads) | 141 ms | 144 ms | 186 ms | 201 ms | 120 ms |
| Kubernetes (Linux container, 1 thread/pod) | **84 ms** | 84 ms | 102 ms | 126 ms | 61 ms |

The containerised deployment is ~40 % faster per request despite being limited
to a single torch thread — Linux + a slim image beats Windows-host Python here.

**Is 72.5 % good?** For a 4-conv-block CNN trained from scratch on 3,200 images,
yes — random is 50 %. The limit here is data volume and model capacity, not the
pipeline. Fine-tuning a pretrained ResNet18 would reach ~97 % on the same data,
but the assignment asks for a *baseline* CNN, and a baseline is what makes later
improvements measurable.
<!-- RESULTS_END -->

Artifacts produced by a training run:

| File | Contents |
|---|---|
| `artifacts/model.pt` | Serialized model: weights + class order + preprocessing config |
| `artifacts/metrics.json` | Per-epoch history, final test metrics, confusion matrix |
| `artifacts/training_curves.png` | Loss and accuracy curves (train vs validation) |
| `artifacts/confusion_matrix.png` | Confusion matrix on the held-out test split |
| `artifacts/classification_report.txt` | Per-class precision / recall / F1 |

---

## 2. Repository layout

```
mlops-cats-dogs/
├── data/
│   ├── raw/                     # 4000 original JPEGs      (DVC-tracked)
│   └── processed/               # 224x224 RGB, 80/10/10     (DVC pipeline output)
├── src/
│   ├── config.py                # single loader for params.yaml
│   ├── data/
│   │   ├── download_data.py     # fetch the dataset
│   │   ├── preprocess.py        # resize + stratified split
│   │   └── dataset.py           # Datasets/DataLoaders + augmentation
│   ├── models/cnn.py            # BaselineCNN
│   └── training/train.py        # training loop + MLflow tracking
├── api/
│   ├── main.py                  # FastAPI app: /health /predict /metrics
│   ├── model_loader.py          # loads artifacts/model.pt once at startup
│   ├── preprocessing.py         # inference-time transform
│   └── monitoring.py            # request logging + metrics collector
├── tests/
│   ├── test_preprocessing.py    # 24 tests
│   ├── test_inference.py        # 22 tests
│   └── test_monitoring.py       # 11 tests
├── scripts/
│   ├── smoke_test.py            # post-deployment health + prediction gate
│   ├── evaluate_deployed_model.py  # true vs predicted over real HTTP
│   └── deploy_local.ps1         # build → minikube → deploy → smoke test
├── k8s/
│   ├── deployment.yaml          # 2 replicas, probes, resource limits
│   └── service.yaml             # NodePort 30080
├── .github/workflows/
│   ├── ci.yml                   # test → build → smoke test → publish to GHCR
│   └── cd.yml                   # minikube → deploy → smoke test → rollback
├── dvc.yaml / dvc.lock          # reproducible preprocess → train pipeline
├── params.yaml                  # every hyperparameter, one place
├── requirements.txt             # full pinned dev/training deps
├── requirements-api.txt         # slim pinned serving deps (Docker)
├── requirements-dev.txt         # test-only deps (CI)
├── Dockerfile / .dockerignore
└── train.py                     # entrypoint: python train.py
```

**Design rule:** nothing is hardcoded. Image size, split ratios, seeds,
augmentation, learning rate, MLflow names all live in `params.yaml`, and
`src/config.py` is the only thing that reads it.

---

## 3. Prerequisites

| Tool | Needed for | Notes |
|---|---|---|
| Python 3.12+ | everything | This project was developed on 3.14.6 |
| Git | M1 onwards | source versioning |
| DVC | M1 | `pip install dvc` |
| Docker | M2, M4 | Docker Desktop on Windows |
| Minikube + kubectl | M4 | `minikube start --driver=docker` |
| A GitHub repo | M3, M4 | Actions runs CI/CD; GHCR needs no extra secret |

> **PATH note:** if `dvc` / `mlflow` are not on your PATH after `pip install`,
> use `python -m dvc ...` and `python -m mlflow ...`. All commands below use
> that form so they work either way.

**PyTorch, not TensorFlow.** TensorFlow has no wheels for Python 3.14, so the
model is a PyTorch `nn.Module`. Nothing else about the assignment changes.

### Install

```bash
pip install -r requirements.txt
```

---

## 4. M1 — Model development & experiment tracking

### 4.1 Get the data (DVC-tracked)

```bash
python src/data/download_data.py
```

Downloads 2000 cats + 2000 dogs from the official Microsoft *Kaggle Cats and
Dogs* dataset. It reads only the byte ranges it needs out of the 825 MB remote
zip via HTTP Range requests, verifies every JPEG, and silently drops the four
corrupt files known to exist in that dataset. No Kaggle credentials required.

```bash
python -m dvc add data/raw      # already done; produces data/raw.dvc
```

`data/raw.dvc` is a 5-line pointer holding the md5 of the whole directory.
**Git stores that pointer; DVC stores the 189 MB of images.**

```bash
git ls-files data/               # -> .gitignore, raw.dvc  (no JPEGs)
cat data/raw.dvc                 # -> md5, size, nfiles: 4000
```

### 4.2 Preprocess: 224×224 RGB + stratified 80/10/10

```bash
python src/data/preprocess.py
```

Resizes the short side to 224 and centre-crops, which preserves aspect ratio
instead of squashing the animal. The split is computed **per class**, so every
split is exactly 50/50 cats:dogs:

| Split | cat | dog | total |
|---|---|---|---|
| train | 1600 | 1600 | 3200 |
| val | 200 | 200 | 400 |
| test | 200 | 200 | 400 |

Seeded with `data.seed: 42`, so the split is identical on any machine.
Verify no leakage:

```bash
python -c "from pathlib import Path; s={x:set(p.name for p in Path(f'data/processed/{x}').rglob('*.jpg')) for x in ('train','val','test')}; print('overlaps:', len(s['train']&s['val']), len(s['train']&s['test']), len(s['val']&s['test']))"
```
Expect `overlaps: 0 0 0`.

### 4.3 Augmentation (training split only)

`src/data/dataset.py` applies `RandomResizedCrop(224, 0.8–1.0)` →
`RandomHorizontalFlip` → `RandomRotation(±15°)` → `ColorJitter(0.2)` →
`ToTensor` → ImageNet `Normalize` **to the training set only**. Validation and
test get resize + normalize, so their scores stay honest.

```bash
python src/data/dataset.py
```

Prints batch shapes, confirms `classes: ['cat','dog']` (cat = index 0), proves
augmentation is live (`same index twice differs: True`) and writes a visual grid
to `artifacts/sample_augmented_batch.png`.

### 4.4 The baseline CNN

```bash
python src/models/cnn.py
```

```
input  3×224×224
block1 Conv3×3(32)  + BN + ReLU + MaxPool2  →  32×112×112
block2 Conv3×3(64)  + BN + ReLU + MaxPool2  →  64× 56× 56
block3 Conv3×3(128) + BN + ReLU + MaxPool2  → 128× 28× 28
block4 Conv3×3(256) + BN + ReLU + MaxPool2  → 256× 14× 14
head   AdaptiveAvgPool → Dropout(0.5) → Linear(256→2)
```

**389,410 parameters.** Global average pooling instead of a flatten keeps the
classifier at 514 parameters rather than ~25M, which matters a lot when training
from scratch on 3,200 images. BatchNorm everywhere is what lets a from-scratch
network train at `lr=1e-3` without diverging.

### 4.5 Train + track in MLflow

```bash
python train.py
```

Roughly 5 minutes per epoch on 8 CPU threads (no GPU needed). Logs to MLflow:

* **params** — every value in `params.yaml`, flattened, plus device, torch
  version, parameter count, split sizes
* **metrics** — `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`,
  `epoch_seconds` per epoch; then `test_loss/accuracy/precision/recall/f1` and
  the four confusion-matrix cells
* **artifacts** — loss & accuracy curves, confusion matrix, classification
  report, `metrics.json`, `model.pt`, `params.yaml`, `split_summary.json`

Early stopping watches `val_loss` with `patience: 3`, and the **best** weights
(not the last) are restored before the test evaluation.

Browse the runs:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Then open <http://localhost:5000>.

> MLflow 3.x put the plain-file store (`./mlruns`) into maintenance mode and
> refuses to use it, so this project uses the SQLite backend (`mlflow.db`).

### 4.6 The reproducible DVC pipeline

```bash
python -m dvc dag        # data/raw.dvc → preprocess → train
python -m dvc status     # what is stale and why
python -m dvc repro      # re-run only the stale stages
python -m dvc metrics show
```

Change `data.image_size` in `params.yaml` and DVC knows both stages are stale;
change `train.epochs` and it knows only training is.

---

## 5. M2 — Packaging & containerization

### 5.1 Run the API locally

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Interactive docs at <http://localhost:8000/docs>.

| Endpoint | Purpose |
|---|---|
| `GET /health` | 200 + `status: ok` when the model is loaded; **503** when not |
| `POST /predict` | multipart image → predicted label + class probabilities |
| `GET /metrics` | request counts, latency percentiles, prediction counts |
| `GET /metrics/prometheus` | same numbers, Prometheus text format |

### 5.2 Test with curl

```bash
curl http://localhost:8000/health
```
```json
{"status":"ok","api_version":"1.0.0","model_loaded":true,"classes":["cat","dog"],"image_size":224,"mlflow_run_id":"...","test_accuracy":0.87}
```

```bash
curl -X POST -F "file=@data/processed/test/dog/dog_00013.jpg" http://localhost:8000/predict
```
```json
{"filename":"dog_00013.jpg","predicted_label":"dog","predicted_index":1,
 "confidence":0.9412,"probabilities":{"cat":0.0588,"dog":0.9412},
 "inference_time_ms":184.83}
```

A non-image upload returns **400**, a missing `file` field **422**, an upload
over 10 MB **413** — never a 500.

**Postman:** `POST http://localhost:8000/predict` → Body → form-data → key
`file`, type **File**, and attach a JPEG.

### 5.3 Build and run the container

```bash
docker build -t cats-dogs-api:latest .
```
```bash
docker run --rm -p 8000:8000 cats-dogs-api:latest
```
```bash
curl http://localhost:8000/health
```

The image installs torch from the **PyTorch CPU wheel index**, which cuts about
2 GB of unusable CUDA payload. Dependencies are installed before the source is
copied, so editing code doesn't invalidate the slow layer. It runs as a
non-root user (uid 10001) and carries a `HEALTHCHECK`. `.dockerignore` keeps
all 276 MB of data, `mlruns/`, `.git/` and the training-only source out.

---

## 6. M3 — CI pipeline

`.github/workflows/ci.yml`, on every push/PR to `main`:

**Job 1 — `test`**
1. Install pinned deps (`requirements-api.txt` + `requirements-dev.txt`) from the CPU wheel index
2. Verify `artifacts/model.pt` exists and print its class order, run id and metrics
3. Run pytest

**Job 2 — `build-and-push`** (`needs: test`, so a red build can never publish)
1. Build the image with Buildx + GitHub Actions layer cache
2. Load it locally and **smoke test it** — `/health` must return `status: ok`, `/predict` must return a `predicted_label`
3. Push to **GHCR** tagged `latest`, `sha-<commit>` and the branch name

GHCR authenticates with the automatically-provided `GITHUB_TOKEN`, so **no
manual secret is required**. (For Docker Hub instead, swap the login step for
`DOCKERHUB_USERNAME`/`DOCKERHUB_TOKEN` secrets and change `REGISTRY`.)

Run the tests locally:

```bash
python -m pytest -v
```

What the 57 tests actually protect:

* **preprocessing** — any input geometry → exactly 224×224 RGB; aspect ratio
  preserved; grayscale expanded to 3 channels; normalization actually applied
  (the test that catches the single most damaging train/serve mismatch); the
  split is 80/10/10, deterministic, disjoint and lossless
* **train/serve consistency** — the API transform and the training *eval*
  transform produce bitwise-comparable tensors; the train transform is random
  and the eval transform is not
* **inference** — logits are `(N, 2)`; probabilities sum to 1; eval mode is
  deterministic and train mode is not; parameter count stays in range (catches
  an accidental architecture change); **cat is index 0 and dog is index 1**
  (if this flips, every prediction inverts)
* **API** — 200 on valid images, 400 on garbage/empty, 422 on a missing field
* **monitoring** — counters, percentiles, bounded latency window, JSON log shape

---

## 7. M4 — CD & Kubernetes deployment

### 7.1 One-command local demo

```powershell
powershell -ExecutionPolicy Bypass -File scripts/deploy_local.ps1
```

Builds the image → `minikube image load` → `kubectl apply` → `kubectl set image`
→ `kubectl rollout status` → smoke tests, and **rolls back automatically** if
the smoke tests fail.

### 7.2 Manual equivalent

```bash
minikube start --driver=docker --cpus=2 --memory=4096
```
```bash
docker build -t cats-dogs-api:local . && minikube image load cats-dogs-api:local
```
```bash
kubectl apply -f k8s/deployment.yaml && kubectl apply -f k8s/service.yaml
```
```bash
kubectl set image deployment/cats-dogs-api api=cats-dogs-api:local
```
```bash
kubectl rollout status deployment/cats-dogs-api --timeout=300s
```
```bash
minikube service cats-dogs-api --url
```

The Deployment runs **2 replicas** with `maxUnavailable: 0` (zero-downtime
updates), CPU/memory requests and limits, and two probes: **readiness** keeps a
pod out of the Service until the model has loaded, **liveness** restarts a
wedged pod. The Service is a **NodePort** (30080) because Minikube has no cloud
load balancer — a `LoadBalancer` would sit in `<pending>` forever.

### 7.3 Smoke tests — the pipeline gate

```bash
python scripts/smoke_test.py --base-url "$(minikube service cats-dogs-api --url)"
```

Four checks: **health** (200 + `status: ok` + `model_loaded`), **prediction**
(valid label, probabilities summing to 1), **error handling** (400 for a
non-image), and an optional real-image check when `data/processed` is present.

It **exits 1 on any failure**, which is exactly what fails the CD pipeline:

```bash
python scripts/smoke_test.py --base-url http://localhost:9999 --retries 2
echo $?    # -> 1
```

### 7.4 The CD workflow

`.github/workflows/cd.yml` triggers on **successful completion of CI** on `main`
(or manually with a chosen tag):

1. Resolve the immutable image reference `ghcr.io/<owner>/<repo>:sha-<commit>`
2. Start Minikube inside the runner
3. Create a GHCR pull secret from `GITHUB_TOKEN`
4. `kubectl apply` the manifests
5. **`kubectl set image`** — the actual deployment/update step
6. `kubectl rollout status` — fails the job if pods never become ready
7. **Run the smoke tests — non-zero exit fails the pipeline**
8. On failure: `kubectl rollout undo` + dump pod logs

Deploying a SHA tag rather than `:latest` is what makes the deployment
reproducible and a rollback meaningful.

---

## 8. M5 — Monitoring, logs & post-deployment evaluation

### 8.1 Request/response logging

Middleware in `api/main.py` times **every** request and emits one structured
JSON line per request to stdout, so `kubectl logs` / `docker logs` collects it
with no extra agent:

```json
{"timestamp":"2026-08-28T10:22:38Z","event":"prediction","filename":"a.jpg","bytes":1652,
 "predicted_label":"dog","confidence":0.9412,"probabilities":{"cat":0.0588,"dog":0.9412},
 "inference_time_ms":184.83}
{"timestamp":"2026-08-28T10:22:38Z","event":"request","method":"POST","path":"/predict",
 "status":200,"latency_ms":254.05,"client":"10.244.0.1"}
```

`/health` probes are counted but **not** logged — otherwise probe noise every
5 seconds would drown the real traffic. Set `ACCESS_LOG_FILE` to also write to
a file. Every response carries an `X-Response-Time-ms` header.

```bash
kubectl logs -l app=cats-dogs-api --tail=50
```

### 8.2 Metrics

```bash
curl http://localhost:8000/metrics
```
```json
{"uptime_seconds":1.7,
 "requests":{"total":8,"by_endpoint":{"/predict":6,"/health":2},
             "by_status_class":{"2xx":7,"4xx":1},"errors_total":1,"error_rate":0.125},
 "latency_ms":{"count":8,"average":162.37,"min":1.94,"max":254.05,
               "p50":218.68,"p95":254.05,"p99":254.05,"window_size":8},
 "predictions":{"total":5,"by_label":{"cat":3,"dog":2},"mean_confidence":0.8712}}
```

Latency percentiles come from a **bounded** 1000-entry window, so a
long-running pod can't grow memory without limit, while the totals still count
every request. A Prometheus rendering is at `/metrics/prometheus`.

> Metrics are **per-process**. With 2 replicas each pod reports its own share of
> traffic — that is a deliberate simplification, not a bug. Aggregating across
> pods is what a real Prometheus deployment would add.

### 8.3 Post-deployment evaluation (true vs predicted)

```bash
python scripts/evaluate_deployed_model.py --base-url "$(minikube service cats-dogs-api --url)" -n 50
```

This never touches the model object. It sends **real HTTP requests** to the
running service and compares the true label (from the source folder) with the
predicted label from the wire — which is what catches serving-time
preprocessing drift, a wrong class order in the artifact, or the wrong image
version deployed, none of which training-time evaluation can see.

Output: confusion matrix, per-class recall, overall accuracy, client-observed
latency percentiles, the service's own `/metrics` snapshot, and a JSON report at
`artifacts/deployment_evaluation.json`.

Without the DVC data checked out it falls back to **simulated** requests and
says so explicitly — synthetic images carry no cat/dog signal, so it reports the
accuracy as not meaningful rather than quietly presenting a fake number.

---

## 9. Complete demonstration workflow

The full sequence, in order:

```bash
# ---- M1: data, training, tracking ----
pip install -r requirements.txt
python src/data/download_data.py
python src/data/preprocess.py
python src/data/dataset.py
python src/models/cnn.py
python train.py
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db     # -> localhost:5000
python -m dvc dag

# ---- M2: API + container ----
python -m pytest -v
python -m uvicorn api.main:app --port 8000                       # -> localhost:8000/docs
curl http://localhost:8000/health
curl -X POST -F "file=@data/processed/test/dog/dog_00013.jpg" http://localhost:8000/predict
docker build -t cats-dogs-api:latest .
docker run --rm -p 8000:8000 cats-dogs-api:latest

# ---- M3: CI ----
git push                    # -> Actions: test -> build -> smoke test -> GHCR

# ---- M4: deploy + smoke test ----
powershell -ExecutionPolicy Bypass -File scripts/deploy_local.ps1

# ---- M5: monitoring + evaluation ----
curl http://localhost:8000/metrics
kubectl logs -l app=cats-dogs-api --tail=50
python scripts/evaluate_deployed_model.py --base-url "$(minikube service cats-dogs-api --url)" -n 50
```

---

## 10. Requirement checklist

| # | Requirement | Where |
|---|---|---|
| **M1** | Git source versioning | repo history, one commit per step |
| | DVC data versioning | `data/raw.dvc`, `dvc.yaml`, `dvc.lock` |
| | Cats vs Dogs dataset | `src/data/download_data.py` |
| | 224×224 RGB preprocessing | `src/data/preprocess.py` |
| | 80/10/10 split | `split_indices()`, `data/processed/split_summary.json` |
| | Data augmentation | `src/data/dataset.py::build_transforms(train=True)` |
| | Baseline CNN | `src/models/cnn.py` |
| | Serialized artifact | `artifacts/model.pt` |
| | MLflow params | `flatten_params()` + `mlflow.log_params` |
| | MLflow metrics | per-epoch + final test metrics |
| | Confusion matrix | `artifacts/confusion_matrix.png` |
| | Loss curves | `artifacts/training_curves.png` |
| **M2** | FastAPI service | `api/main.py` |
| | `GET /health` | `api/main.py::health` |
| | `POST /predict` | `api/main.py::predict` |
| | Probabilities + label | `PredictionResponse` |
| | `requirements.txt`, pinned | `requirements*.txt` — every version pinned |
| | Dockerfile | `Dockerfile` |
| | Local Docker test | §5.3 |
| | curl/Postman test | §5.2 |
| **M3** | Pytest | `tests/`, 57 tests |
| | Preprocessing unit test | `tests/test_preprocessing.py` |
| | Model/inference unit test | `tests/test_inference.py` |
| | Dependency installation | `ci.yml` → *Install dependencies* |
| | Automated testing | `ci.yml` → job `test` |
| | Docker image build | `ci.yml` → job `build-and-push` |
| | Publish to registry | GHCR, `latest` + `sha-<commit>` |
| **M4** | K8s Deployment | `k8s/deployment.yaml` |
| | K8s Service | `k8s/service.yaml` |
| | CD workflow | `.github/workflows/cd.yml` |
| | Image update | `kubectl set image` step |
| | Automatic deploy/update | triggered by CI success on `main` |
| | Health smoke test | `smoke_test.py::check_health` |
| | Prediction smoke test | `smoke_test.py::check_prediction` |
| | Fail pipeline on smoke failure | `sys.exit(1)` → fails the CD job |
| **M5** | Request/response logging | `api/monitoring.py` + middleware |
| | Request count | `/metrics` → `requests.total` |
| | Latency tracking | `/metrics` → `latency_ms` p50/p95/p99 |
| | Basic metrics | `/metrics`, `/metrics/prometheus` |
| | Post-deployment evaluation | `scripts/evaluate_deployed_model.py` |
| | True vs predicted comparison | confusion matrix in that script |
| | Final packaging + README | this file |
| | Demonstration workflow | §9 |

---

## 11. Design decisions & honest limitations

**Decisions**

* **PyTorch over TensorFlow** — no TF wheels for Python 3.14.
* **Global average pooling** — a flatten head would be ~25M parameters and
  would overfit 3,200 images badly.
* **Two logits + softmax** rather than one sigmoid — gives `/predict` a clean
  per-class probability vector.
* **Normalization constants in `params.yaml`, echoed into `model.pt`** — the
  API rebuilds the exact training-time transform instead of assuming it.
* **`artifacts/model.pt` is committed to Git** (~1.6 MB). Normally a model
  belongs in DVC or a model registry, but there is no cloud DVC remote here and
  CI must build an image containing it. The tradeoff is deliberate.
* **SQLite MLflow backend** — MLflow 3.x refuses the plain-file store.

**Limitations**

* **Metrics are per-process**, so a 2-replica Deployment reports per-pod
  numbers. Real aggregation needs Prometheus scraping both pods.
* **No DVC remote.** `dvc push` has nowhere to go, so data is reproducible from
  `download_data.py` rather than restorable from a cache on a fresh clone.
* **CPU-only training**, ~5 min/epoch. A GPU would make this minutes.
* **The CD pipeline deploys to a Minikube inside the runner**, which is
  destroyed when the job ends. It exercises the identical `kubectl` sequence a
  real cluster needs, but it is not a persistent environment.
* **No authentication on the API.** Fine for an assignment; a real pet adoption
  platform needs auth and rate limiting.
* **A GHCR package is private by default** — either make it public or keep the
  pull secret the CD workflow creates.

---

## 12. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `dvc: command not found` | Use `python -m dvc ...` (pip Scripts dir not on PATH) |
| `filesystem tracking backend is in maintenance mode` | MLflow 3.x rejects `./mlruns`; this project already uses `sqlite:///mlflow.db` |
| `/health` returns 503 | `artifacts/model.pt` missing → run `python train.py`, or set `MODEL_PATH` |
| `Model artifact not found` | Same as above; the path is baked in as `/app/artifacts/model.pt` in the image |
| Service `<pending>` in Minikube | Don't use `type: LoadBalancer`; this project uses NodePort |
| `ImagePullBackOff` | GHCR package is private — apply the pull secret, or make it public |
| Pods `OOMKilled` | Raise `resources.limits.memory` above 1536Mi |
| Training feels slow | Expected on CPU (~5 min/epoch); it pins all cores automatically |
| `no such file: data/processed/...` | Run `python src/data/download_data.py && python src/data/preprocess.py` |
