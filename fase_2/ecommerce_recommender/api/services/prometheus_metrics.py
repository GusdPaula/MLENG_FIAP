"""Prometheus metrics for Grafana integration.

This module provides Prometheus metrics export for application-specific metrics
that can be scraped by Prometheus and visualized in Grafana.
"""

import time
from typing import Callable

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class PrometheusMetrics:
    """Prometheus metrics manager for the application."""

    def __init__(self, port: int = 9090):
        """Initialize Prometheus metrics.

        Args:
            port: Port to expose metrics endpoint on.
        """
        if not PROMETHEUS_AVAILABLE:
            raise ImportError(
                "prometheus_client is not installed. "
                "Install it with: pip install prometheus_client"
            )

        self.port = port

        # Prediction metrics
        self.predictions_total = Counter(
            "predictions_total",
            "Total predictions made",
            ["predictor_type", "model_version"],
        )

        self.prediction_duration = Histogram(
            "prediction_duration_seconds",
            "Prediction duration in seconds",
            ["predictor_type", "model_version"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
        )

        # Drift detection metrics
        self.drift_alerts_total = Counter(
            "drift_alerts_total",
            "Total drift detection alerts",
            ["drift_type", "severity"],
        )

        self.drift_score = Gauge("drift_score", "Current drift score", ["drift_type"])

        # API metrics
        self.api_requests_total = Counter(
            "api_requests_total", "Total API requests", ["endpoint", "method", "status"]
        )

        self.api_request_duration = Histogram(
            "api_request_duration_seconds",
            "API request duration in seconds",
            ["endpoint", "method"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )

        # Model metrics
        self.model_version_info = Gauge(
            "model_version_info",
            "Current model version info",
            ["model_name", "version", "predictor_type"],
        )

        self.active_users = Gauge(
            "active_users", "Number of active users in current window"
        )

        self.active_items = Gauge(
            "active_items", "Number of active items in current window"
        )

        # Error metrics
        self.errors_total = Counter(
            "errors_total", "Total errors", ["error_type", "severity"]
        )

    def start_server(self) -> None:
        """Start the Prometheus metrics HTTP server."""
        start_http_server(self.port)

    def record_prediction(
        self, predictor_type: str, model_version: str, duration: float
    ) -> None:
        """Record a prediction metric.

        Args:
            predictor_type: Type of predictor used.
            model_version: Model version.
            duration: Prediction duration in seconds.
        """
        self.predictions_total.labels(
            predictor_type=predictor_type, model_version=model_version
        ).inc()
        self.prediction_duration.labels(
            predictor_type=predictor_type, model_version=model_version
        ).observe(duration)

    def record_drift_alert(self, drift_type: str, severity: str, score: float) -> None:
        """Record a drift detection alert.

        Args:
            drift_type: Type of drift (data_shift, model_drift).
            severity: Severity level (low, medium, high, critical).
            score: Drift score.
        """
        self.drift_alerts_total.labels(drift_type=drift_type, severity=severity).inc()
        self.drift_score.labels(drift_type=drift_type).set(score)

    def record_api_request(
        self, endpoint: str, method: str, status: int, duration: float
    ) -> None:
        """Record an API request metric.

        Args:
            endpoint: API endpoint.
            method: HTTP method.
            status: HTTP status code.
            duration: Request duration in seconds.
        """
        self.api_requests_total.labels(
            endpoint=endpoint, method=method, status=str(status)
        ).inc()
        self.api_request_duration.labels(endpoint=endpoint, method=method).observe(
            duration
        )

    def set_model_info(
        self, model_name: str, version: str, predictor_type: str
    ) -> None:
        """Set model version info.

        Args:
            model_name: Model name.
            version: Model version.
            predictor_type: Predictor type.
        """
        self.model_version_info.labels(
            model_name=model_name, version=version, predictor_type=predictor_type
        ).set(1)

    def set_active_counts(self, users: int, items: int) -> None:
        """Set active user and item counts.

        Args:
            users: Number of active users.
            items: Number of active items.
        """
        self.active_users.set(users)
        self.active_items.set(items)

    def record_error(self, error_type: str, severity: str) -> None:
        """Record an error metric.

        Args:
            error_type: Type of error.
            severity: Severity level.
        """
        self.errors_total.labels(error_type=error_type, severity=severity).inc()


def timing_decorator(metrics: PrometheusMetrics, endpoint: str, method: str = "GET"):
    """Decorator to measure and record API request timing.

    Args:
        metrics: PrometheusMetrics instance.
        endpoint: API endpoint name.
        method: HTTP method.

    Returns:
        Decorator function.
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                status = 200  # Default success status
                return result
            except Exception as e:
                status = 500
                metrics.record_error(error_type=type(e).__name__, severity="high")
                raise
            finally:
                duration = time.time() - start_time
                metrics.record_api_request(
                    endpoint=endpoint, method=method, status=status, duration=duration
                )

        return wrapper

    return decorator


def prediction_decorator(
    metrics: PrometheusMetrics, predictor_type: str, model_version: str
):
    """Decorator to measure and record prediction timing.

    Args:
        metrics: PrometheusMetrics instance.
        predictor_type: Type of predictor.
        model_version: Model version.

    Returns:
        Decorator function.
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metrics.record_prediction(
                    predictor_type=predictor_type,
                    model_version=model_version,
                    duration=duration,
                )

        return wrapper

    return decorator
