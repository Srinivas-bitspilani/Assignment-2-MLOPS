"""Post-deployment smoke tests.

Run against a freshly deployed service to prove it is actually usable, not just
"running". Two checks are mandatory (both required by M4):

    1. health     - GET  /health returns 200 with status "ok" and model_loaded
    2. prediction - POST /predict returns a valid label + probability vector

A third check runs only when the DVC-tracked test images are available locally:
it sends real cats and dogs and compares true vs predicted labels.

**Exit code is 1 on any failure**, which is what makes the CD pipeline fail
instead of silently shipping a broken deployment.

Usage:
    python scripts/smoke_test.py --base-url http://localhost:8000
    python scripts/smoke_test.py --base-url "$(minikube service cats-dogs-api --url)"
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def log(status: str, message: str) -> None:
    print(f"[{status:^6}] {message}", flush=True)


def synthetic_image_bytes() -> bytes:
    """A plain image, used so the prediction check needs no dataset."""
    buffer = io.BytesIO()
    Image.new("RGB", (300, 300), (120, 140, 160)).save(buffer, "JPEG")
    return buffer.getvalue()


def wait_for_service(base_url: str, retries: int, delay: float) -> bool:
    """Poll /health until the service answers, so we do not race startup."""
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                log("ok", f"service reachable after {attempt} attempt(s)")
                return True
            log("wait", f"attempt {attempt}/{retries}: HTTP {response.status_code}")
        except requests.RequestException as exc:
            log("wait", f"attempt {attempt}/{retries}: {type(exc).__name__}")
        time.sleep(delay)
    return False


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def check_health(base_url: str) -> bool:
    log("run", "health check: GET /health")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
    except requests.RequestException as exc:
        log("FAIL", f"request error: {exc}")
        return False

    if response.status_code != 200:
        log("FAIL", f"expected HTTP 200, got {response.status_code}: {response.text[:200]}")
        return False

    body = response.json()
    print(f"          response: {json.dumps(body)}")

    for key, expected in (("status", "ok"), ("model_loaded", True)):
        if body.get(key) != expected:
            log("FAIL", f"{key} was {body.get(key)!r}, expected {expected!r}")
            return False

    if not body.get("classes"):
        log("FAIL", "no classes reported by the service")
        return False

    log("PASS", f"healthy, classes={body['classes']}, run_id={body.get('mlflow_run_id')}")
    return True


def check_prediction(base_url: str) -> bool:
    log("run", "prediction check: POST /predict")
    try:
        response = requests.post(
            f"{base_url}/predict",
            files={"file": ("smoke.jpg", synthetic_image_bytes(), "image/jpeg")},
            timeout=30,
        )
    except requests.RequestException as exc:
        log("FAIL", f"request error: {exc}")
        return False

    if response.status_code != 200:
        log("FAIL", f"expected HTTP 200, got {response.status_code}: {response.text[:200]}")
        return False

    body = response.json()
    print(f"          response: {json.dumps(body)}")

    for field in ("predicted_label", "confidence", "probabilities"):
        if field not in body:
            log("FAIL", f"missing field '{field}' in response")
            return False

    probabilities = body["probabilities"]
    total = sum(probabilities.values())
    if abs(total - 1.0) > 1e-4:
        log("FAIL", f"probabilities sum to {total}, expected 1.0")
        return False

    if body["predicted_label"] not in probabilities:
        log("FAIL", f"label {body['predicted_label']!r} is not one of {list(probabilities)}")
        return False

    log("PASS", f"predicted '{body['predicted_label']}' "
                f"({body['confidence']:.4f}), latency {body.get('inference_time_ms')} ms")
    return True


def check_error_handling(base_url: str) -> bool:
    """A non-image upload must be rejected with 400, not crash the service."""
    log("run", "error handling check: POST /predict with a non-image")
    try:
        response = requests.post(
            f"{base_url}/predict",
            files={"file": ("bad.txt", b"not an image", "text/plain")},
            timeout=15,
        )
    except requests.RequestException as exc:
        log("FAIL", f"request error: {exc}")
        return False

    if response.status_code != 400:
        log("FAIL", f"expected HTTP 400 for a bad upload, got {response.status_code}")
        return False

    log("PASS", "bad upload correctly rejected with 400")
    return True


def check_real_images(base_url: str, per_class: int = 5) -> bool:
    """Optional: send real test images and compare true vs predicted labels.

    Skipped (and treated as a pass) when data/processed is not present, which
    is the normal case in CI where DVC data is not checked out.
    """
    test_dir = PROJECT_ROOT / "data" / "processed" / "test"
    if not test_dir.is_dir():
        log("skip", "real-image check: data/processed/test not available")
        return True

    log("run", f"real-image check: {per_class} images per class")
    correct = total = 0
    for class_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        for image_path in sorted(class_dir.glob("*.jpg"))[:per_class]:
            try:
                response = requests.post(
                    f"{base_url}/predict",
                    files={"file": (image_path.name, image_path.read_bytes(), "image/jpeg")},
                    timeout=30,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                log("FAIL", f"request error for {image_path.name}: {exc}")
                return False

            predicted = response.json()["predicted_label"]
            total += 1
            correct += predicted == class_dir.name
            print(f"          {image_path.name:<20} true={class_dir.name:<4} "
                  f"pred={predicted:<4} {'OK' if predicted == class_dir.name else 'MISS'}")

    accuracy = correct / total if total else 0.0
    log("PASS", f"real-image accuracy {correct}/{total} = {accuracy:.2%} "
                "(informational, does not gate the pipeline)")
    return True


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a deployed classifier")
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="Base URL of the deployed service")
    parser.add_argument("--retries", type=int, default=30,
                        help="How many times to poll /health before giving up")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds between startup polls")
    parser.add_argument("--per-class", type=int, default=5,
                        help="Real test images per class (when data is available)")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    print("=" * 68)
    print(f"SMOKE TESTS against {base_url}")
    print("=" * 68)

    if not wait_for_service(base_url, args.retries, args.delay):
        log("FAIL", f"service never became reachable at {base_url}")
        return 1

    results = {
        "health": check_health(base_url),
        "prediction": check_prediction(base_url),
        "error_handling": check_error_handling(base_url),
        "real_images": check_real_images(base_url, args.per_class),
    }

    print("=" * 68)
    for name, passed in results.items():
        print(f"  {name:<16} {'PASS' if passed else 'FAIL'}")
    print("=" * 68)

    if all(results.values()):
        print("ALL SMOKE TESTS PASSED")
        return 0
    print("SMOKE TESTS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
