"""Request/response logging and in-process metrics for the inference service.

Provides the three things M5 asks for:

  * request/response logging - one structured JSON line per request, written to
    stdout so `kubectl logs` / `docker logs` picks it up with no extra agent
  * request count            - totals, plus breakdowns by endpoint, status class
                               and predicted label
  * latency tracking         - count/sum/min/max and p50/p95/p99 percentiles

Metrics are deliberately in-process and in-memory: no Prometheus server is
required for a student assignment, and the numbers are exposed as JSON on
GET /metrics (plus a Prometheus-compatible text rendering on
GET /metrics/prometheus for anyone who wants to scrape it).

Because the state is per-process, a 2-replica Deployment reports per-pod
numbers. That is called out in the README rather than hidden.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import Counter, deque
from threading import Lock

# Dedicated logger so access logs are machine-readable JSON lines, separate
# from the human-readable application log.
access_logger = logging.getLogger("api.access")
access_logger.setLevel(logging.INFO)
if not access_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))  # the message IS the JSON
    access_logger.addHandler(handler)
    access_logger.propagate = False

    # Optional file sink, useful when demonstrating log collection locally.
    log_file = os.getenv("ACCESS_LOG_FILE")
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        access_logger.addHandler(file_handler)


class MetricsCollector:
    """Thread-safe counters and a bounded latency window."""

    # Keep the most recent N latencies for percentiles. Bounded on purpose:
    # an unbounded list would grow without limit in a long-running pod.
    LATENCY_WINDOW = 1000

    def __init__(self) -> None:
        self._lock = Lock()
        self.started_at = time.time()

        self.requests_total = 0
        self.requests_by_endpoint: Counter = Counter()
        self.requests_by_status: Counter = Counter()
        self.errors_total = 0

        self.predictions_total = 0
        self.predictions_by_label: Counter = Counter()
        self.confidence_sum = 0.0

        self.latencies_ms: deque = deque(maxlen=self.LATENCY_WINDOW)
        self.latency_count = 0
        self.latency_sum_ms = 0.0
        self.latency_min_ms: float | None = None
        self.latency_max_ms: float | None = None

    # ------------------------------------------------------------------ #
    def record_request(
        self, endpoint: str, status_code: int, latency_ms: float
    ) -> None:
        with self._lock:
            self.requests_total += 1
            self.requests_by_endpoint[endpoint] += 1
            self.requests_by_status[f"{status_code // 100}xx"] += 1
            if status_code >= 400:
                self.errors_total += 1

            self.latencies_ms.append(latency_ms)
            self.latency_count += 1
            self.latency_sum_ms += latency_ms
            self.latency_min_ms = (
                latency_ms if self.latency_min_ms is None
                else min(self.latency_min_ms, latency_ms)
            )
            self.latency_max_ms = (
                latency_ms if self.latency_max_ms is None
                else max(self.latency_max_ms, latency_ms)
            )

    def record_prediction(self, label: str, confidence: float) -> None:
        with self._lock:
            self.predictions_total += 1
            self.predictions_by_label[label] += 1
            self.confidence_sum += confidence

    # ------------------------------------------------------------------ #
    @staticmethod
    def _percentile(sorted_values: list[float], fraction: float) -> float:
        if not sorted_values:
            return 0.0
        index = min(
            len(sorted_values) - 1, max(0, int(round(fraction * (len(sorted_values) - 1))))
        )
        return sorted_values[index]

    def snapshot(self) -> dict:
        """Current metrics as a plain dict, ready to serialise as JSON."""
        with self._lock:
            latencies = sorted(self.latencies_ms)
            average = (
                self.latency_sum_ms / self.latency_count if self.latency_count else 0.0
            )
            mean_confidence = (
                self.confidence_sum / self.predictions_total
                if self.predictions_total
                else 0.0
            )
            return {
                "uptime_seconds": round(time.time() - self.started_at, 1),
                "requests": {
                    "total": self.requests_total,
                    "by_endpoint": dict(self.requests_by_endpoint),
                    "by_status_class": dict(self.requests_by_status),
                    "errors_total": self.errors_total,
                    "error_rate": round(
                        self.errors_total / self.requests_total, 4
                    ) if self.requests_total else 0.0,
                },
                "latency_ms": {
                    "count": self.latency_count,
                    "average": round(average, 2),
                    "min": round(self.latency_min_ms, 2) if self.latency_min_ms else 0.0,
                    "max": round(self.latency_max_ms, 2) if self.latency_max_ms else 0.0,
                    "p50": round(self._percentile(latencies, 0.50), 2),
                    "p95": round(self._percentile(latencies, 0.95), 2),
                    "p99": round(self._percentile(latencies, 0.99), 2),
                    "window_size": len(latencies),
                },
                "predictions": {
                    "total": self.predictions_total,
                    "by_label": dict(self.predictions_by_label),
                    "mean_confidence": round(mean_confidence, 4),
                },
            }

    def prometheus(self) -> str:
        """Render the same numbers in Prometheus text exposition format."""
        data = self.snapshot()
        lines = [
            "# HELP api_requests_total Total HTTP requests handled.",
            "# TYPE api_requests_total counter",
            f"api_requests_total {data['requests']['total']}",
            "# HELP api_request_errors_total HTTP responses with status >= 400.",
            "# TYPE api_request_errors_total counter",
            f"api_request_errors_total {data['requests']['errors_total']}",
            "# HELP api_request_latency_ms_average Average request latency.",
            "# TYPE api_request_latency_ms_average gauge",
            f"api_request_latency_ms_average {data['latency_ms']['average']}",
            "# HELP api_request_latency_ms_p95 95th percentile request latency.",
            "# TYPE api_request_latency_ms_p95 gauge",
            f"api_request_latency_ms_p95 {data['latency_ms']['p95']}",
            "# HELP api_predictions_total Predictions served.",
            "# TYPE api_predictions_total counter",
            f"api_predictions_total {data['predictions']['total']}",
            "# HELP api_uptime_seconds Seconds since process start.",
            "# TYPE api_uptime_seconds gauge",
            f"api_uptime_seconds {data['uptime_seconds']}",
        ]
        for endpoint, count in data["requests"]["by_endpoint"].items():
            lines.append(f'api_requests_by_endpoint{{endpoint="{endpoint}"}} {count}')
        for label, count in data["predictions"]["by_label"].items():
            lines.append(f'api_predictions_by_label{{label="{label}"}} {count}')
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        """Used by tests."""
        self.__init__()


# Process-wide collector.
metrics = MetricsCollector()


def log_event(event: str, **fields) -> None:
    """Emit one structured JSON log line."""
    payload = {
        # UTC, marked with Z: container logs from different nodes must be
        # comparable regardless of local timezone.
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event,
        **fields,
    }
    access_logger.info(json.dumps(payload, default=str))
