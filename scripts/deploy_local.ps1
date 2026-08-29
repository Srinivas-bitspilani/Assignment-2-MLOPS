<#
.SYNOPSIS
    Build, deploy to local Minikube, and smoke test - the whole M2+M4 demo.

.DESCRIPTION
    Runs the same sequence the CD pipeline runs, but against your own Minikube:
      1. docker build
      2. minikube image load        (so the cluster can use a local image)
      3. kubectl apply  k8s/
      4. kubectl set image          (the deployment step)
      5. kubectl rollout status     (fails if pods never become ready)
      6. python scripts/smoke_test.py  (fails the script if the API is broken)

    Prerequisites: Docker Desktop, minikube and kubectl on PATH, and
    artifacts/model.pt present (run `python train.py` first).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/deploy_local.ps1
#>

$ErrorActionPreference = "Stop"

# A UNIQUE tag per deploy, mirroring what CD does with the commit SHA.
# Reusing one fixed tag is a trap: `minikube image load` skips the copy when the
# tag already exists in the cluster, and `kubectl set image` with an unchanged
# tag is a no-op, so the cluster silently keeps serving the previous model.
$SHA = (git rev-parse --short HEAD 2>$null)
if (-not $SHA) { $SHA = "nogit" }
$STAMP = Get-Date -Format "HHmmss"
$IMAGE = "cats-dogs-api:local-$SHA-$STAMP"
$DEPLOYMENT = "cats-dogs-api"

function Step($message) {
    Write-Host ""
    Write-Host ("=" * 68) -ForegroundColor Cyan
    Write-Host ">>> $message" -ForegroundColor Cyan
    Write-Host ("=" * 68) -ForegroundColor Cyan
}

# Run from the repository root regardless of where this was invoked.
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))

# ---- preflight ---------------------------------------------------------- #
Step "Preflight checks"
foreach ($tool in @("docker", "minikube", "kubectl")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "$tool is not installed or not on PATH"
    }
    Write-Host "  $tool found"
}
if (-not (Test-Path "artifacts/model.pt")) {
    throw "artifacts/model.pt is missing. Run 'python train.py' first."
}
Write-Host "  artifacts/model.pt found"
$modelInfo = python -c "import torch;c=torch.load('artifacts/model.pt',map_location='cpu',weights_only=False);print(c.get('architecture','baseline_cnn'), c.get('metrics',{}).get('test_accuracy'))"
Write-Host "  shipping model: $modelInfo"

# ---- 1. build ----------------------------------------------------------- #
Step "1/6  Building the Docker image"
docker build -t $IMAGE .
if ($LASTEXITCODE -ne 0) { throw "docker build failed" }

# ---- 2. cluster --------------------------------------------------------- #
Step "2/6  Ensuring Minikube is running"
$status = (minikube status --format "{{.Host}}" 2>$null)
if ($status -ne "Running") {
    minikube start --driver=docker --cpus=2 --memory=4096
    if ($LASTEXITCODE -ne 0) { throw "minikube start failed" }
} else {
    Write-Host "  Minikube already running"
}

Step "3/6  Loading the image into the cluster"
# Without this the node cannot see a locally-built image. --overwrite is belt
# and braces now that the tag is unique per deploy.
minikube image load $IMAGE --overwrite
if ($LASTEXITCODE -ne 0) { throw "minikube image load failed" }
Write-Host "  loaded $IMAGE"

# ---- 3. deploy ---------------------------------------------------------- #
Step "4/6  Applying manifests and setting the image"
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl set image "deployment/$DEPLOYMENT" "api=$IMAGE"

Step "5/6  Waiting for the rollout"
kubectl rollout status "deployment/$DEPLOYMENT" --timeout=300s
if ($LASTEXITCODE -ne 0) {
    Write-Host "Rollout failed - pod diagnostics:" -ForegroundColor Red
    kubectl get pods
    kubectl describe "deployment/$DEPLOYMENT"
    kubectl logs -l "app=$DEPLOYMENT" --tail=100 --all-containers
    throw "rollout failed"
}
kubectl get deployment,pods,svc -o wide

# ---- 4. smoke test ------------------------------------------------------ #
Step "6/6  Smoke testing the deployment"
$url = (minikube service $DEPLOYMENT --url | Select-Object -First 1)
Write-Host "  Service URL: $url"
python scripts/smoke_test.py --base-url $url --retries 30 --delay 2
if ($LASTEXITCODE -ne 0) {
    Write-Host "Smoke tests FAILED - rolling back" -ForegroundColor Red
    kubectl rollout undo "deployment/$DEPLOYMENT"
    throw "smoke tests failed"
}

Write-Host ""
Write-Host "DEPLOYMENT SUCCESSFUL" -ForegroundColor Green
Write-Host "  API:  $url"
Write-Host "  Docs: $url/docs"
Write-Host ""
Write-Host "Try it:" -ForegroundColor Yellow
Write-Host "  curl $url/health"
Write-Host "  curl -X POST -F `"file=@data/processed/test/dog/dog_00013.jpg`" $url/predict"
