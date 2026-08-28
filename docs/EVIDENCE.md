# Verification Evidence

Captured output from real runs, committed in the repository so it can be read
without clicking through to any external service. Every figure below was
produced by the commands shown next to it.

Raw captures live in [`docs/evidence/`](.):

| File | Contents |
|---|---|
| `registry-state.txt` | Current model registry: versions, metrics, aliases |
| `registry-promote-v1.txt` | Promotion of v1 to champion (initial state) |
| `registry-promote-v2.txt` | Gate **promoting** v2 over v1 (+0.2500 margin) |
| `registry-gate-reject.txt` | Gate **rejecting** v1 against champion v2 (−0.2500) |
| `mlflow-runs.txt` | Full MLflow experiment dump: params, metrics, artifacts per run |
| `pytest-local.txt` | 79-test run output |
| `junit-local.xml` | JUnit XML report |
| `coverage.xml` | Cobertura coverage report |
| `champion-evaluation.txt` | 400 HTTP requests against the promoted champion |

---

## 1. MLflow Model Registry

Two candidate models are registered under `cats-vs-dogs-classifier`, and the
`champion` alias controls which one is served.

```
 ver  architecture        test_acc  test_f1    loss  aliases                run
------------------------------------------------------------------------------
   1  baseline_cnn          0.7250   0.7250  0.5633  -                      d604ad0f
   2  resnet18_finetune     0.9750   0.9750  0.0654  champion               3594a3ac <<
```

```bash
python scripts/promote_model.py --show
```

Every version carries queryable tags: `architecture`, `test_accuracy`,
`test_f1`, `test_loss`, `best_epoch`, `classes`, `source_run_id`.

### 1.1 Promotion is gated, and a training run never promotes itself

Each run registers a new version and tags it `challenger`.
[`scripts/promote_model.py`](../scripts/promote_model.py) is the only thing that
can move the `champion` alias, and it applies one rule:

```
promote if   challenger.test_accuracy  >  champion.test_accuracy + min_delta
```

**Accepted** — ResNet18 challenger beats the baseline champion:

```
DECISION
  candidate : v2 (resnet18_finetune) acc=0.9750
  champion  : v1 (baseline_cnn) acc=0.7250
  reason    : challenger v2 accuracy 0.9750 vs champion v1 0.7250
              (margin +0.2500, required > 0.005)
  outcome   : PROMOTE
  alias 'champion' -> v2
  alias 'challenger' cleared (it is now champion)
  exported resnet18_finetune.pt -> model.pt (44.79 MB)
```

**Rejected** — the reverse comparison, proving the gate is not a rubber stamp:

```
DECISION
  candidate : v1 (baseline_cnn) acc=0.7250
  champion  : v2 (resnet18_finetune) acc=0.9750
  reason    : challenger v1 accuracy 0.7250 vs champion v2 0.9750
              (margin -0.2500, required > 0.005)
  outcome   : REJECT
```

A marginal gain is also rejected: with `min_delta=0.005`, a challenger that is
only 0.2 pp better does not churn production. That boundary is unit-tested in
[`tests/test_registry.py`](../tests/test_registry.py).

### 1.2 Promotion is wired to serving

Promotion exports the winning version's checkpoint to `artifacts/model.pt` —
the exact file the Docker image bakes in — so **only a promoted model can be
served**. Verified end to end:

```bash
curl http://localhost:8000/health
```
```json
{"status":"ok","api_version":"1.0.0","model_loaded":true,
 "architecture":"resnet18_finetune","classes":["cat","dog"],"image_size":224,
 "mlflow_run_id":"3594a3ace0f9427998c21a6468c8236e","test_accuracy":0.975}
```

`architecture` and `mlflow_run_id` on `/health` identify precisely which
registry version is answering requests.

---

## 2. Model comparison

| | v1 baseline_cnn | v2 resnet18_finetune |
|---|---|---|
| Approach | 4-block CNN, trained from scratch | ImageNet ResNet18, frozen backbone, new head |
| Total parameters | 389,410 | 11,177,538 |
| **Trainable** parameters | 389,410 | **1,026** |
| Epochs | 10 | 4 |
| Time per epoch | ~250 s | ~215 s |
| Test accuracy | 72.50 % | **97.50 %** |
| Test F1 (macro) | 0.7250 | **0.9750** |
| Test loss | 0.5633 | **0.0654** |
| Confusion matrix | `[[143,57],[53,147]]` | `[[195,5],[5,195]]` |
| Artifact size | 1.57 MB | 44.79 MB |
| MLflow run | `d604ad0f` | `3594a3ace` |

Fine-tuning trains **1,026 parameters** — 0.009 % of the network — and gains
25 percentage points. That is the entire argument for transfer learning, and
keeping the baseline registered as v1 is what makes the gain measurable.

Per-model plots: `artifacts/training_curves.png` /
`artifacts/confusion_matrix.png` (v1) and
`artifacts/resnet18_finetune_training_curves.png` /
`artifacts/resnet18_finetune_confusion_matrix.png` (v2).

---

## 3. Train/serve consistency

`scripts/evaluate_deployed_model.py` sends all 400 test images to the running
service over HTTP and compares the true label with the label on the wire. The
deployed confusion matrix matches the training-time matrix **exactly**, for
both models, in every environment they were deployed to:

| Model | Environment | Confusion matrix | Accuracy |
|---|---|---|---|
| v1 baseline | Training (in-process) | `[[143,57],[53,147]]` | 72.50 % |
| v1 baseline | Local uvicorn | `[[143,57],[53,147]]` | 72.50 % |
| v1 baseline | Docker container | `[[143,57],[53,147]]` | 72.50 % |
| v1 baseline | Kubernetes, 2 replicas | `[[143,57],[53,147]]` | 72.50 % |
| **v2 champion** | Training (in-process) | `[[195,5],[5,195]]` | **97.50 %** |
| **v2 champion** | Local uvicorn | `[[195,5],[5,195]]` | **97.50 %** |

Per-image confidences match too, so this is bitwise reproducibility rather than
rounding coincidence. This is the check that catches serving-time preprocessing
drift, a wrong class order in the artifact, or the wrong version deployed —
none of which training-time evaluation can see.

```bash
python scripts/evaluate_deployed_model.py --base-url http://localhost:8000 -n 200
```

---

## 4. Tests

**79 tests, all passing** (`docs/evidence/pytest-local.txt`):

| Module | Tests | What it protects |
|---|---|---|
| `test_preprocessing.py` | 24 | Geometry, normalisation, the 80/10/10 split, bad input |
| `test_inference.py` | 22 | Model shapes, checkpoint loading, HTTP contract |
| `test_registry.py` | 22 | Model factory and the promotion gate |
| `test_monitoring.py` | 11 | Counters, latency percentiles, log format |

Run in CI against **Python 3.12 and 3.13** (3.12 is the floor: `numpy==2.5.2` declares `requires_python >= 3.12`).

### Coverage

Overall line coverage is 50 %, but that single number is misleading — the
**serving path**, which is what runs in production, is covered 88–100 %:

| Module | Coverage | |
|---|---|---|
| `api/model_loader.py` | **100 %** | serving |
| `api/preprocessing.py` | **100 %** | serving |
| `src/models/factory.py` | **100 %** | serving |
| `api/monitoring.py` | 96 % | serving |
| `api/main.py` | 88 % | serving |
| `src/models/cnn.py` | 65 % | model |
| `src/data/preprocess.py` | 44 % | training only |
| `src/data/dataset.py` | 44 % | training only |
| `src/training/train.py` | 11 % | training only |
| `src/data/download_data.py` | 0 % | one-off data fetch |

The training scripts are verified by **being executed** — the runs in §2 are
their test — rather than by unit tests that would need the DVC dataset present.
Unit-testing a 40-minute training loop is not the right tool; running it and
checking the artifacts is.

---

## 5. Confirmed pipeline runs

Commit `ac83a70`, with the artifacts each job uploaded:

| Workflow | Job | Result | Time | Uploaded |
|---|---|---|---|---|
| [CI #6](https://github.com/Srinivas-bitspilani/Assignment-2-MLOPS/actions/runs/33197730399) | Tests (py3.12) | **success** | 1.0 min | `test-evidence-py3.12` (115 KB) |
| [CI #6](https://github.com/Srinivas-bitspilani/Assignment-2-MLOPS/actions/runs/33197730399) | Tests (py3.13) | **success** | 1.1 min | `test-evidence-py3.13` (115 KB) |
| [CI #6](https://github.com/Srinivas-bitspilani/Assignment-2-MLOPS/actions/runs/33197730399) | Build & publish image | **success** | 1.7 min | `build-evidence` (2 KB) |
| [CD #6](https://github.com/Srinivas-bitspilani/Assignment-2-MLOPS/actions/runs/33197945076) | Deploy to Kubernetes and smoke test | **success** | 2.3 min | `deployment-evidence` (10 KB) |

### The gate demonstrated under failure

CI #5 failed on purpose-revealing bugs, and the run record shows the pipeline
refusing to ship:

| Job | Result |
|---|---|
| `Tests (py3.11)` | **failure** — `numpy==2.5.2` requires Python >= 3.12 |
| `Tests (py3.12)` | **failure** — a test imported mlflow, which CI does not install |
| `Build & publish image` | **skipped** — blocked by `needs: test` |
| CD #5 | **skipped** — blocked by `workflow_run` requiring CI success |

No image was published and no deployment happened. That is stronger evidence
that the gates work than any green run: they were tested by a real failure.

The fixes were a pin-compatible matrix (3.12 / 3.13, verified against every
pinned package on PyPI first) and moving the gate logic into
[`src/promotion.py`](../src/promotion.py), which imports no MLflow.
[`scripts/check_ci_imports.py`](../scripts/check_ci_imports.py) now reproduces
CI's serving-only environment locally so this class of failure cannot recur:

```bash
python scripts/check_ci_imports.py
```
```
Hiding training-only imports: ['dvc', 'matplotlib', 'mlflow', 'sklearn']
79 passed
```

---

## 6. What every CI run uploads

Both workflows attach downloadable artifacts (90-day retention) and render
summary tables onto the run page, so results are visible without downloading
anything.

**CI** (`.github/workflows/ci.yml`)

| Evidence | Detail |
|---|---|
| `test-evidence-py3.11`, `test-evidence-py3.12` | JUnit XML, `coverage.xml`, HTML coverage, model-artifact inspection |
| `build-evidence` | `/health`, `/predict`, `/metrics` responses from the built image, plus container logs |
| Rendered summaries | Environment versions, per-module test counts, coverage per package, image size/layers/user/entrypoint, smoke-test predictions, published digest and tags |

The `build-and-push` job declares `needs: test`, so a red suite can never
publish an image. The image is smoke-tested **before** it is pushed.

**CD** (`.github/workflows/cd.yml`)

| Evidence | Detail |
|---|---|
| `deployment-evidence` | `cluster-info`, `nodes`, `apply`, `rollout`, `get-all`, `describe deployment`, `describe service`, `rollout-history`, full smoke-test stdout, evaluation JSON, pod logs, live `/metrics` and Prometheus output |
| Rendered summaries | Cluster version, image deployed, replicas ready, service type/nodePort, rollout strategy, smoke-test table, evaluation table, sample structured log events |

CD adds an explicit assertion that `readyReplicas == spec.replicas` rather than
trusting the rollout message, and on failure captures pod logs and describes
**before** running `kubectl rollout undo`.

---

## 7. Reproducing all of it

```bash
pip install -r requirements.txt
python src/data/download_data.py
python src/data/preprocess.py

# candidate 1: the from-scratch baseline
python train.py

# candidate 2: transfer learning
python train.py --architecture resnet18_finetune --epochs 4 \
  --learning-rate 0.005 --run-name resnet18-finetune \
  --artifact-name resnet18_finetune.pt

# registry: inspect, then let the gate decide
python scripts/promote_model.py --show
python scripts/promote_model.py --dry-run
python scripts/promote_model.py

# serve the champion and verify it end to end
python -m uvicorn api.main:app --port 8000
python scripts/smoke_test.py --base-url http://localhost:8000
python scripts/evaluate_deployed_model.py --base-url http://localhost:8000 -n 200

# browse every run and registry version
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

`python -m dvc repro` re-runs only the stages whose dependencies or params
actually changed.
