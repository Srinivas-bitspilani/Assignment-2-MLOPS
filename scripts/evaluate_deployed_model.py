"""Post-deployment evaluation of the DEPLOYED model (M5).

This is different from the evaluation inside train.py: it does not touch the
model object at all. It sends real HTTP requests to a running service, exactly
as a client would, and compares the **true label** (taken from the folder the
image came from) with the **predicted label** returned over the wire.

That is what catches problems training-time evaluation cannot see:
serving-time preprocessing drift, a wrong class order in the artifact, the
wrong model version deployed, or a broken image build.

Two data sources:
  * real      - images from data/processed/test (default when available)
  * simulated - synthetic images, when the DVC data is not checked out

Outputs an accuracy figure, a confusion matrix, a per-class breakdown, latency
statistics, and the service's own /metrics snapshot. Writes a JSON report to
artifacts/deployment_evaluation.json.

Usage:
    python scripts/evaluate_deployed_model.py --base-url http://localhost:8000
    python scripts/evaluate_deployed_model.py --base-url "$(minikube service cats-dogs-api --url)" -n 100
"""

from __future__ import annotations

import argparse
import io
import json
import random
import statistics
import sys
import time
from pathlib import Path

import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = PROJECT_ROOT / "data" / "processed" / "test"


# --------------------------------------------------------------------------- #
# request sources
# --------------------------------------------------------------------------- #
def collect_real_samples(per_class: int, seed: int) -> list[tuple[str, str, bytes]]:
    """(true_label, filename, bytes) sampled from the held-out test split."""
    samples: list[tuple[str, str, bytes]] = []
    rng = random.Random(seed)
    for class_dir in sorted(p for p in TEST_DIR.iterdir() if p.is_dir()):
        files = sorted(class_dir.glob("*.jpg"))
        rng.shuffle(files)
        for path in files[:per_class]:
            samples.append((class_dir.name, path.name, path.read_bytes()))
    rng.shuffle(samples)
    return samples


def collect_simulated_samples(per_class: int, classes: list[str], seed: int):
    """Synthetic fallback so this script still runs without the dataset.

    The images carry no real cat/dog signal, so accuracy is meaningless here --
    the point is to exercise the deployed path end to end. This is reported
    honestly in the output rather than presented as a real accuracy number.
    """
    rng = random.Random(seed)
    samples = []
    for class_name in classes:
        for i in range(per_class):
            buffer = io.BytesIO()
            color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
            Image.new("RGB", (256, 256), color).save(buffer, "JPEG")
            samples.append((class_name, f"sim_{class_name}_{i}.jpg", buffer.getvalue()))
    rng.shuffle(samples)
    return samples


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def predict_one(base_url: str, filename: str, blob: bytes, timeout: float):
    started = time.perf_counter()
    response = requests.post(
        f"{base_url}/predict",
        files={"file": (filename, blob, "image/jpeg")},
        timeout=timeout,
    )
    round_trip_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    return response.json(), round_trip_ms


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a deployed classifier")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("-n", "--per-class", type=int, default=50,
                        help="Requests per class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--simulate", action="store_true",
                        help="Force simulated requests even if real data exists")
    parser.add_argument("--output", default="artifacts/deployment_evaluation.json")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    # ---- confirm the service is up and learn its class list ----------------
    try:
        health = requests.get(f"{base_url}/health", timeout=10)
        health.raise_for_status()
    except requests.RequestException as exc:
        print(f"ERROR: service not reachable at {base_url}: {exc}")
        return 1

    health_body = health.json()
    classes = health_body.get("classes") or ["cat", "dog"]

    print("=" * 72)
    print(f"POST-DEPLOYMENT EVALUATION  ->  {base_url}")
    print("=" * 72)
    print(f"service classes      : {classes}")
    print(f"deployed mlflow run  : {health_body.get('mlflow_run_id')}")
    print(f"training test_accuracy: {health_body.get('test_accuracy')}")

    # ---- choose the request source ----------------------------------------
    use_real = TEST_DIR.is_dir() and not args.simulate
    if use_real:
        samples = collect_real_samples(args.per_class, args.seed)
        mode = "real"
    else:
        samples = collect_simulated_samples(args.per_class, classes, args.seed)
        mode = "simulated"
        reason = "forced by --simulate" if args.simulate else f"{TEST_DIR} not found"
        print(f"\nNOTE: using SIMULATED requests ({reason}).")
        print("      Accuracy from synthetic images is NOT meaningful; only the")
        print("      request/latency figures below should be interpreted.")

    print(f"\nmode                 : {mode}")
    print(f"requests to send     : {len(samples)}")
    print("-" * 72)

    # ---- send the requests -------------------------------------------------
    confusion = {t: {p: 0 for p in classes} for t in classes}
    latencies: list[float] = []
    server_latencies: list[float] = []
    confidences: list[float] = []
    correct = 0
    failures = 0
    records = []

    started_all = time.perf_counter()
    for index, (true_label, filename, blob) in enumerate(samples, start=1):
        try:
            body, round_trip_ms = predict_one(base_url, filename, blob, args.timeout)
        except requests.RequestException as exc:
            failures += 1
            print(f"  [{index:>4}] {filename:<24} REQUEST FAILED: {exc}")
            continue

        predicted = body["predicted_label"]
        latencies.append(round_trip_ms)
        server_latencies.append(body.get("inference_time_ms", 0.0))
        confidences.append(body["confidence"])

        if true_label in confusion and predicted in confusion[true_label]:
            confusion[true_label][predicted] += 1
        correct += predicted == true_label

        records.append({
            "filename": filename,
            "true_label": true_label,
            "predicted_label": predicted,
            "confidence": round(body["confidence"], 4),
            "round_trip_ms": round(round_trip_ms, 2),
        })

        if index <= 10 or index % 25 == 0:
            flag = "OK  " if predicted == true_label else "MISS"
            print(f"  [{index:>4}] {filename:<24} true={true_label:<4} "
                  f"pred={predicted:<4} conf={body['confidence']:.4f} "
                  f"{round_trip_ms:>7.1f}ms  {flag}")

    total_seconds = time.perf_counter() - started_all
    answered = len(latencies)
    if answered == 0:
        print("\nERROR: every request failed.")
        return 1

    # ---- results -----------------------------------------------------------
    accuracy = correct / answered
    print("-" * 72)
    print(f"requests answered    : {answered}/{len(samples)} (failures: {failures})")
    print(f"wall time            : {total_seconds:.1f}s "
          f"({answered / total_seconds:.2f} req/s)")

    print("\nTRUE vs PREDICTED (confusion matrix)")
    header = "  true \\ pred  " + "".join(f"{c:>8}" for c in classes) + "     total"
    print(header)
    for true_label in classes:
        row = confusion[true_label]
        print(f"  {true_label:<12} " + "".join(f"{row[p]:>8}" for p in classes)
              + f"{sum(row.values()):>10}")

    print("\nPER-CLASS RECALL")
    per_class = {}
    for true_label in classes:
        row = confusion[true_label]
        total = sum(row.values())
        recall = row[true_label] / total if total else 0.0
        per_class[true_label] = {"support": total, "correct": row[true_label],
                                 "recall": round(recall, 4)}
        print(f"  {true_label:<6} {row[true_label]:>4}/{total:<4} = {recall:.2%}")

    print(f"\nOVERALL ACCURACY     : {correct}/{answered} = {accuracy:.2%}"
          + ("" if mode == "real" else "   (meaningless: simulated inputs)"))

    print("\nLATENCY (client-observed round trip)")
    ordered = sorted(latencies)
    def pct(p): return ordered[min(len(ordered) - 1, int(p * (len(ordered) - 1)))]
    print(f"  mean {statistics.mean(latencies):>8.1f} ms")
    print(f"  p50  {pct(0.50):>8.1f} ms")
    print(f"  p95  {pct(0.95):>8.1f} ms")
    print(f"  max  {max(latencies):>8.1f} ms")
    print(f"  server-side inference mean: {statistics.mean(server_latencies):.1f} ms")
    print(f"  mean confidence           : {statistics.mean(confidences):.4f}")

    # ---- the service's own metrics -----------------------------------------
    service_metrics = None
    try:
        response = requests.get(f"{base_url}/metrics", timeout=10)
        if response.status_code == 200:
            service_metrics = response.json()
            print("\nSERVICE /metrics SNAPSHOT")
            print(f"  requests total   : {service_metrics['requests']['total']}")
            print(f"  errors total     : {service_metrics['requests']['errors_total']}")
            print(f"  predictions total: {service_metrics['predictions']['total']}")
            print(f"  by label         : {service_metrics['predictions']['by_label']}")
            print(f"  latency p95      : {service_metrics['latency_ms']['p95']} ms")
    except requests.RequestException:
        print("\n(could not read /metrics)")

    # ---- write the report --------------------------------------------------
    report = {
        "base_url": base_url,
        "mode": mode,
        "evaluated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "deployed_model": {
            "mlflow_run_id": health_body.get("mlflow_run_id"),
            "training_test_accuracy": health_body.get("test_accuracy"),
            "classes": classes,
        },
        "requests": {"sent": len(samples), "answered": answered, "failed": failures},
        "accuracy": round(accuracy, 4),
        "accuracy_is_meaningful": mode == "real",
        "confusion_matrix": confusion,
        "per_class": per_class,
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 2),
            "p50": round(pct(0.50), 2),
            "p95": round(pct(0.95), 2),
            "max": round(max(latencies), 2),
            "server_inference_mean": round(statistics.mean(server_latencies), 2),
        },
        "mean_confidence": round(statistics.mean(confidences), 4),
        "service_metrics": service_metrics,
        "records": records,
    }

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport written -> {output_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
