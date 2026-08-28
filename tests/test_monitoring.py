"""Tests for the monitoring layer (M5): counters, latency stats, logging.

The metrics collector is process-global, so each test resets it first.
"""

from __future__ import annotations

import json

import pytest

from api.monitoring import MetricsCollector, metrics
from tests.conftest import make_image_bytes


@pytest.fixture(autouse=True)
def clean_metrics():
    metrics.reset()
    yield
    metrics.reset()


# --------------------------------------------------------------------------- #
# collector unit tests
# --------------------------------------------------------------------------- #
def test_request_counting():
    collector = MetricsCollector()
    collector.record_request("/predict", 200, 10.0)
    collector.record_request("/predict", 200, 20.0)
    collector.record_request("/health", 200, 1.0)
    collector.record_request("/predict", 400, 5.0)

    snapshot = collector.snapshot()
    assert snapshot["requests"]["total"] == 4
    assert snapshot["requests"]["by_endpoint"] == {"/predict": 3, "/health": 1}
    assert snapshot["requests"]["by_status_class"] == {"2xx": 3, "4xx": 1}
    assert snapshot["requests"]["errors_total"] == 1
    assert snapshot["requests"]["error_rate"] == 0.25


def test_latency_statistics():
    collector = MetricsCollector()
    for value in [10.0, 20.0, 30.0, 40.0, 50.0]:
        collector.record_request("/predict", 200, value)

    latency = collector.snapshot()["latency_ms"]
    assert latency["count"] == 5
    assert latency["average"] == 30.0
    assert latency["min"] == 10.0
    assert latency["max"] == 50.0
    assert latency["p50"] == 30.0
    assert latency["p95"] == 50.0


def test_latency_window_is_bounded():
    """A long-running pod must not accumulate latencies without limit."""
    collector = MetricsCollector()
    for i in range(collector.LATENCY_WINDOW + 500):
        collector.record_request("/predict", 200, float(i))

    snapshot = collector.snapshot()
    assert snapshot["latency_ms"]["window_size"] == collector.LATENCY_WINDOW
    # Totals still count every request, even though the window rolled.
    assert snapshot["latency_ms"]["count"] == collector.LATENCY_WINDOW + 500


def test_prediction_counting():
    collector = MetricsCollector()
    collector.record_prediction("cat", 0.9)
    collector.record_prediction("dog", 0.7)
    collector.record_prediction("cat", 0.8)

    predictions = collector.snapshot()["predictions"]
    assert predictions["total"] == 3
    assert predictions["by_label"] == {"cat": 2, "dog": 1}
    assert predictions["mean_confidence"] == pytest.approx(0.8)


def test_empty_collector_does_not_divide_by_zero():
    snapshot = MetricsCollector().snapshot()
    assert snapshot["requests"]["total"] == 0
    assert snapshot["requests"]["error_rate"] == 0.0
    assert snapshot["latency_ms"]["average"] == 0.0
    assert snapshot["predictions"]["mean_confidence"] == 0.0


def test_prometheus_rendering():
    collector = MetricsCollector()
    collector.record_request("/predict", 200, 12.5)
    collector.record_prediction("dog", 0.95)

    text = collector.prometheus()
    assert "api_requests_total 1" in text
    assert "api_predictions_total 1" in text
    assert 'api_predictions_by_label{label="dog"} 1' in text
    # Every metric must be preceded by HELP/TYPE lines to be scrapeable.
    assert text.count("# HELP") >= 6
    assert text.endswith("\n")


# --------------------------------------------------------------------------- #
# endpoint tests
# --------------------------------------------------------------------------- #
def test_metrics_endpoint_reflects_traffic(client):
    for _ in range(3):
        client.post("/predict", files={"file": ("a.jpg", make_image_bytes(), "image/jpeg")})
    client.post("/predict", files={"file": ("bad.txt", b"nope", "text/plain")})

    body = client.get("/metrics").json()
    assert body["predictions"]["total"] == 3
    assert body["requests"]["errors_total"] == 1
    assert body["requests"]["by_endpoint"]["/predict"] == 4
    assert body["latency_ms"]["count"] >= 4
    assert body["model"]["model_loaded"] is True


def test_prometheus_endpoint_is_plain_text(client):
    client.post("/predict", files={"file": ("a.jpg", make_image_bytes(), "image/jpeg")})
    response = client.get("/metrics/prometheus")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "api_requests_total" in response.text


def test_response_time_header_is_present(client):
    response = client.get("/")
    assert "X-Response-Time-ms" in response.headers
    assert float(response.headers["X-Response-Time-ms"]) >= 0


def test_prediction_is_logged_as_json(client, caplog):
    """Request/response logging must emit parseable JSON, not free text."""
    with caplog.at_level("INFO", logger="api.access"):
        client.post("/predict", files={"file": ("x.jpg", make_image_bytes(), "image/jpeg")})

    events = []
    for record in caplog.records:
        try:
            events.append(json.loads(record.getMessage()))
        except json.JSONDecodeError:
            pass

    prediction_events = [e for e in events if e.get("event") == "prediction"]
    assert prediction_events, f"no prediction event logged; got {events}"

    event = prediction_events[0]
    assert event["filename"] == "x.jpg"
    assert event["predicted_label"] in ("cat", "dog")
    assert "confidence" in event and "probabilities" in event
    assert event["timestamp"].endswith("Z")


def test_health_requests_are_not_logged_but_are_counted(client, caplog):
    """Probe noise must stay out of the log while still being measured."""
    with caplog.at_level("INFO", logger="api.access"):
        client.get("/health")

    logged_paths = []
    for record in caplog.records:
        try:
            logged_paths.append(json.loads(record.getMessage()).get("path"))
        except json.JSONDecodeError:
            pass

    assert "/health" not in logged_paths
    assert client.get("/metrics").json()["requests"]["by_endpoint"]["/health"] >= 1
