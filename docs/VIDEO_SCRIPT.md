# Demo Video Script

Target length **9–12 minutes**. Everything slow (training, Docker build, first
Minikube start) is done *before* recording; on camera you show the result and
run only the fast commands.

---

## PART 0 — Before you hit record

### 0.1 Warm everything up (20–30 min, not recorded)

```bash
# 1. kubectl must point at minikube, not docker-desktop
kubectl config use-context minikube
minikube status                       # start it if not Running

# 2. Redeploy so the cluster serves the CHAMPION (97.5%), not the old baseline
powershell -ExecutionPolicy Bypass -File scripts/deploy_local.ps1

# 3. Confirm the image is already built and cached
docker images cats-dogs-api
```

Without step 2 your Kubernetes pods still serve the 72.5 % baseline while your
local API serves the 97.5 % champion — the mismatch looks careless on camera.

### 0.2 Start the background services (leave running)

Open three terminals and leave these running the whole time:

| Terminal | Command | Used in |
|---|---|---|
| T1 | `python -m mlflow ui --backend-store-uri sqlite:///mlflow.db` | Part 3 |
| T2 | `kubectl port-forward service/cats-dogs-api 8080:80` | Parts 6–7 |
| T3 | *(your working terminal — you type here)* | everywhere |

### 0.3 Pre-open browser tabs, in this order

1. GitHub repo — Code tab
2. GitHub — **Actions** tab (CI list)
3. GitHub — the **failed CI #5** run (showing skipped jobs)
4. GitHub — **Packages** (the published GHCR image)
5. `http://localhost:5000` — MLflow
6. `http://localhost:8080/docs` — Swagger on the deployed service
7. `http://localhost:8080/metrics`

### 0.4 Recording hygiene

- Terminal font **16–18 pt**, maximised. Tiny text is the #1 reason demo videos
  lose marks.
- `Clear-Host` between sections so each command starts on a clean screen.
- Close Slack/Outlook/notifications.
- OBS Studio (free) or Windows **Win+G** Game Bar. Record at 1080p.
- Do a 20-second test recording first and check the text is readable on playback.

---

## PART 1 — Introduction (0:00–0:45)

**Show:** the GitHub repo landing page (README rendered).

> "This is my MLOps assignment: an end-to-end pipeline for binary cats-versus-dogs
> image classification. The workflow runs from dataset versioning through
> preprocessing, training with experiment tracking, a model registry with gated
> promotion, a FastAPI service, Docker, automated tests, GitHub Actions CI/CD,
> Kubernetes deployment, smoke tests and monitoring.
> I'll walk through all five modules."

Scroll slowly past the badges — point out CI and CD are green.

---

## PART 2 — M1: Data versioning and preprocessing (0:45–2:15)

**Terminal T3:**

```bash
git log --oneline | head -12
```
> "Source is versioned in Git — one commit per stage of the assignment."

```bash
git ls-files data/
```
> "But the dataset is *not* in Git. Only these pointer files are."

```bash
cat data/raw.dvc
```
> "This five-line file is DVC's pointer: the md5 of the whole directory, its size
> and 4,000 files. Git stores this; DVC stores the 189 megabytes of images."

```bash
python -m dvc dag
```
> "The pipeline is declared, so `dvc repro` re-runs only what actually changed."

```bash
cat data/processed/split_summary.json
```
> "Preprocessing resizes everything to 224 by 224 RGB and splits it 80/10/10,
> stratified per class — 3,200 train, 400 validation, 400 test, exactly 50/50
> cats to dogs, with a fixed seed so anyone reproduces the same split."

**Show:** `artifacts/sample_augmented_batch.png`
> "Augmentation — random crop, flip, rotation and colour jitter — is applied to
> the training split only. Validation and test get resize and normalisation
> alone, so their scores stay honest."

---

## PART 3 — M1: Training, MLflow and the Model Registry (2:15–4:30)

This is the section that carries the most marks. Do not rush it.

**Browser → MLflow (localhost:5000) → `cats-vs-dogs` experiment.**

> "Every training run is tracked in MLflow with a SQLite backend."

Click into **baseline-cnn**:
> "41 parameters logged — every value from params.yaml — plus per-epoch metrics
> and the final test metrics. This is the from-scratch baseline CNN: 389,000
> parameters, 72.5 % test accuracy."

Show the **Artifacts** tab: curves, confusion matrix, classification report, the
model, params.yaml, split summary.

Go back, tick **both runs → Compare**:
> "The second candidate is a fine-tuned ResNet18. Same data, same split, same
> evaluation — 97.5 % against the baseline's 72.5 %. Keeping the baseline is what
> makes that 25-point gain measurable rather than just asserted."

**Browser → Models tab → `cats-vs-dogs-classifier`:**
> "Both candidates are registered versions of one model. Version 2 carries the
> champion alias."

**Terminal T3 — the part graders care about:**

```bash
python scripts/promote_model.py --show
```
> "Two versions, with their metrics and aliases."

```bash
python scripts/promote_model.py --dry-run
```
> "Promotion is gated. A training run never promotes itself — it registers a
> version and tags it challenger. This script decides. The rule is: promote only
> if the challenger beats the champion by more than a minimum delta, so a
> meaningless 0.2 % gain can't churn production.
> Right now the champion already wins, so the baseline is **rejected** —
> margin minus 0.25. The gate isn't a rubber stamp."

```bash
cat artifacts/model_registry.json
```
> "Every decision is recorded with its reason."

> "And promotion is wired to serving: promoting a version exports its checkpoint
> to artifacts/model.pt, which is the exact file the Docker image bakes in. Only
> a promoted model can ever be served."

---

## PART 4 — M2: API and containerisation (4:30–6:00)

**Terminal T3:**

```bash
docker images cats-dogs-api
```
> "The service is containerised. Torch comes from the CPU wheel index, which
> keeps about two gigabytes of unusable CUDA out of the image, and it runs as a
> non-root user."

**Browser → `http://localhost:8080/docs`** (Swagger, served by Kubernetes):

> "FastAPI generates this documentation from the same models that shape every
> response."

Expand **GET /health** → Try it out → Execute:
> "Health reports the live architecture and the MLflow run id, so the running
> service tells you exactly which registry version is answering — resnet18,
> run 3594a3ac, 97.5 %."

Expand **POST /predict** → Try it out → upload
`data/processed/test/dog/dog_00013.jpg` → Execute:
> "A real dog image: predicted dog with 99.9 % confidence, and the full
> probability vector for both classes."

Now show the failure path — upload `README.md` as the file:
> "A non-image is rejected with a clean 400, not a 500 or a stack trace."

Optional terminal shot:
```bash
curl -X POST -F "file=@data/processed/test/cat/cat_00013.jpg" http://localhost:8080/predict
```

---

## PART 5 — M3: Tests and CI (6:00–7:30)

**Terminal T3:**

```bash
python -m pytest -q
```
> "79 tests covering preprocessing geometry and the split, the model and the
> HTTP contract, the registry and promotion gate, and the monitoring layer."

```bash
python scripts/check_ci_imports.py
```
> "And this runs the same suite with MLflow, scikit-learn and matplotlib hidden,
> reproducing CI's serving-only environment — because a test that imports a
> training-time module passes locally and fails in CI."

**Browser → GitHub → Actions → latest CI run:**
> "CI runs on every push: a matrix across Python 3.12 and 3.13, dependency
> install, artifact inspection, the tests with coverage, then the Docker build."

Scroll to the **run summary**:
> "Each run renders its results — test counts per module, coverage per package,
> the image size and the actual prediction the smoke test got — and uploads
> JUnit XML, coverage reports and the container's own responses as artifacts."

Show the **Artifacts** section at the bottom of the run page.

**Browser → the FAILED CI run (tab 3):**
> "This one matters more than the green runs. A test failed, and look —
> the image build was **skipped**, and the whole CD workflow was skipped. No
> image published, no deployment. The gates were proven by a real failure."

**Browser → Packages tab:**
> "Green runs publish to the GitHub Container Registry, tagged latest and with
> the commit SHA."

---

## PART 6 — M4: Kubernetes and CD (7:30–9:30)

**Terminal T3:**

```bash
kubectl get deployment,pods,svc -o wide
```
> "The service runs on Kubernetes — two replicas, both Running, behind a
> NodePort service. NodePort rather than LoadBalancer because Minikube has no
> cloud load balancer; a LoadBalancer would sit pending forever."

```bash
kubectl describe deployment cats-dogs-api | Select-String -Pattern "Strategy|Replicas|Liveness|Readiness|Limits|Requests" -Context 0,1
```
> "Rolling update with maxUnavailable zero for zero-downtime, resource limits,
> and two distinct probes: readiness keeps a pod out of the service until its
> model has loaded, liveness restarts a wedged pod."

**Now the smoke tests — the CD gate:**

```bash
python scripts/smoke_test.py --base-url http://localhost:8080 --per-class 5
```
> "Four checks: health, a real prediction, error handling, and real test images
> compared true-versus-predicted. All pass, exit code zero."

**Then prove the gate fails properly:**

```bash
python scripts/smoke_test.py --base-url http://localhost:9999 --retries 2
echo $LASTEXITCODE
```
> "Against a service that isn't there, it exits 1 — and that non-zero exit is
> what fails the CD pipeline, so a deployment that can't actually serve
> predictions is never accepted."

**Browser → GitHub → Actions → latest CD run:**
> "CD triggers automatically on CI success. It starts Minikube in the runner,
> applies the manifests, deploys the SHA-tagged image — immutable, so the
> deployment is reproducible — waits for the rollout, asserts that ready
> replicas equals desired, then runs those same smoke tests. On failure it
> captures the logs first and then rolls back."

Show the CD run summary and its `deployment-evidence` artifact.

---

## PART 7 — M5: Monitoring and post-deployment evaluation (9:30–11:00)

**Terminal T3:**

```bash
kubectl logs -l app=cats-dogs-api --tail=8 --all-containers
```
> "Every request is logged as one structured JSON line — method, path, status,
> latency — and predictions log the label, confidence and full probability
> vector, so predictions can be audited later. Health probes are counted but not
> logged, otherwise probe noise every five seconds would bury the real traffic."

**Browser → `http://localhost:8080/metrics`:**
> "Request counts by endpoint and status class, error rate, latency mean and the
> 50th, 95th and 99th percentiles over a bounded window so memory can't grow
> without limit, and prediction counts by label with mean confidence."

Mention: `/metrics/prometheus` returns the same numbers in Prometheus format.

**The strongest single demo — post-deployment evaluation:**

```bash
python scripts/evaluate_deployed_model.py --base-url http://localhost:8080 -n 50
```
> "This never touches the model object. It sends 100 real test images over HTTP
> and compares the true label — from the folder each image came from — against
> what the service returned. It reports the confusion matrix, per-class recall,
> accuracy and latency."

When it finishes, land the key point:
> "And this reproduces the training-time confusion matrix exactly. Same numbers
> in training, in the container, and on Kubernetes. That's proof there's no
> preprocessing drift between training and serving — which is the failure mode
> that silently breaks most deployed image models, and training-time evaluation
> can never catch it."

---

## PART 8 — Close (11:00–11:30)

**Browser → `docs/EVIDENCE.md` on GitHub.**

> "All of this is captured in the repository — both promotion decisions, the
> MLflow dump, coverage, the consistency tables and the confirmed pipeline runs
> — so the results can be checked without depending on anything external.
> That's the full pipeline: dataset versioning, preprocessing, training,
> tracking, a gated registry, containerised serving, CI, a container registry,
> CD to Kubernetes, smoke-test gating and monitoring. Thank you."

---

## Fallbacks if something breaks on camera

| Problem | Do this |
|---|---|
| `services "cats-dogs-api" not found` | `kubectl config use-context minikube` — Docker Desktop steals the context |
| Port-forward drops | Restart T2; it dies when its terminal closes |
| Swagger won't load | Fall back to `curl http://localhost:8080/health` in the terminal |
| MLflow UI empty | Check you launched it from the repo root so it finds `mlflow.db` |
| Pods not Ready | `kubectl rollout status deployment/cats-dogs-api` and wait |

## Timing summary

| Part | Content | Duration |
|---|---|---|
| 1 | Introduction | 0:45 |
| 2 | M1 data, DVC, preprocessing | 1:30 |
| 3 | M1 training, MLflow, registry | 2:15 |
| 4 | M2 API and Docker | 1:30 |
| 5 | M3 tests and CI | 1:30 |
| 6 | M4 Kubernetes and CD | 2:00 |
| 7 | M5 monitoring and evaluation | 1:30 |
| 8 | Close | 0:30 |
| | **Total** | **~11:30** |

If you need it shorter, cut Part 2 to the `data/raw.dvc` shot alone and trim
Part 5's local pytest run — the CI run page shows the same result.
