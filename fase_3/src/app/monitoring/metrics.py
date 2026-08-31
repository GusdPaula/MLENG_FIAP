"""Prometheus instrumentation.

Everything that touches prometheus_client directly lives in this module.
Other layers (api/, services/) call the small helper functions at the
bottom instead of importing Counter/Histogram themselves — if the metrics
backend ever changes, this is the only file that needs to.
"""

import time

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

# --- Raw metric definitions -------------------------------------------------

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests received",
    ["method", "endpoint", "status_code"],
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "Total HTTP request duration in seconds, including preprocessing and inference",
    ["method", "endpoint"],
)

INFERENCE_DURATION = Histogram(
    "inference_duration_seconds",
    "Model-only inference time in seconds (ONNX Runtime session.run call), "
    "isolated from HTTP/preprocessing overhead so baseline vs. ONNX comparisons "
    "aren't skewed by request handling cost.",
)

CLASSIFICATION_COUNT = Counter(
    "classification_by_label_total",
    "Number of predictions returned per class label",
    ["label"],
)

BATCH_SIZE = Histogram(
    "classify_batch_size",
    "Number of texts submitted per /classify/batch request",
    buckets=(1, 2, 4, 8, 16, 32, 64, 128),
)


# --- HTTP middleware ---------------------------------------------------------


class MetricsMiddleware(BaseHTTPMiddleware):
    """Times every request and records it, skipping /metrics itself so
    Prometheus scraping the endpoint doesn't pollute its own histogram.
    """

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)

        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        endpoint = request.url.path
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=response.status_code,
        ).inc()
        REQUEST_DURATION.labels(method=request.method, endpoint=endpoint).observe(duration)

        return response


async def metrics_endpoint() -> Response:
    """Handler for GET /metrics — returns the Prometheus text exposition
    format directly, not JSON.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def setup_metrics(app: FastAPI) -> None:
    """Wires instrumentation into the app. Called once from main.py's
    create_app(), before routers are included.
    """
    app.add_middleware(MetricsMiddleware)
    app.add_api_route("/metrics", metrics_endpoint, methods=["GET"], include_in_schema=False)


# --- Helpers for services/inference.py --------------------------------------
# Keeps InferenceService free of any prometheus_client import.


def observe_inference_duration(seconds: float) -> None:
    INFERENCE_DURATION.observe(seconds)


def observe_batch_size(size: int) -> None:
    BATCH_SIZE.observe(size)


def record_classification(label: str) -> None:
    CLASSIFICATION_COUNT.labels(label=label).inc()
