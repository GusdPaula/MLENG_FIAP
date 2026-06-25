"""Monitoring module for model and data shift detection.

This module provides functionality to monitor model performance and detect
data distribution shifts over time. It follows SOLID principles with
separate components for different monitoring concerns.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class MonitoringMetrics:
    """Container for monitoring metrics.

    Attributes:
        timestamp: When the metrics were collected.
        prediction_scores: List of prediction scores.
        user_ids: List of user IDs.
        item_ids: List of item IDs.
        custom_metrics: Additional custom metrics.
    """

    timestamp: datetime
    prediction_scores: list[float]
    user_ids: list[int]
    item_ids: list[int]
    custom_metrics: dict[str, Any] | None = None


@dataclass
class ShiftDetectionResult:
    """Result of shift detection analysis.

    Attributes:
        has_shift: Whether a shift was detected.
        shift_type: Type of shift detected (data, model, or both).
        p_value: Statistical p-value for the test.
        test_statistic: The test statistic value.
        threshold: The threshold used for detection.
        message: Human-readable description of the result.
    """

    has_shift: bool
    shift_type: str
    p_value: float
    test_statistic: float
    threshold: float
    message: str


class BaseShiftDetector(ABC):
    """Abstract base class for shift detectors.

    This follows the Interface Segregation Principle by providing a focused
    interface for shift detection.
    """

    def __init__(self, threshold: float = 0.05):
        """Initialize the shift detector.

        Args:
            threshold: P-value threshold for detecting shifts. Defaults to 0.05.
        """
        self.threshold = threshold
        self._baseline_metrics: MonitoringMetrics | None = None

    @abstractmethod
    def detect_shift(self, current_metrics: MonitoringMetrics) -> ShiftDetectionResult:
        """Detect shift between baseline and current metrics.

        Args:
            current_metrics: The current metrics to compare against baseline.

        Returns:
            A ShiftDetectionResult indicating whether a shift was detected.
        """

    def set_baseline(self, metrics: MonitoringMetrics) -> None:
        """Set the baseline metrics for comparison.

        Args:
            metrics: The baseline metrics to use for future comparisons.
        """
        self._baseline_metrics = metrics
        logger.info(
            "Set baseline metrics with %d predictions from %s",
            len(metrics.prediction_scores),
            metrics.timestamp,
        )

    def has_baseline(self) -> bool:
        """Check if baseline metrics have been set.

        Returns:
            True if baseline metrics are set, False otherwise.
        """
        return self._baseline_metrics is not None


class DataShiftDetector(BaseShiftDetector):
    """Detector for data distribution shifts using statistical tests.

    This detector uses Kolmogorov-Smirnov test to detect shifts in
    prediction score distributions.
    """

    name = "data_shift"

    def detect_shift(self, current_metrics: MonitoringMetrics) -> ShiftDetectionResult:
        """Detect data shift using KS test on prediction scores.

        Args:
            current_metrics: The current metrics to compare against baseline.

        Returns:
            A ShiftDetectionResult indicating whether a data shift was detected.

        Raises:
            RuntimeError: If baseline metrics have not been set.
        """
        if not self.has_baseline():
            logger.error("Cannot detect shift: baseline metrics not set")
            raise RuntimeError("Baseline metrics must be set before detecting shifts.")

        baseline_scores = np.array(self._baseline_metrics.prediction_scores)
        current_scores = np.array(current_metrics.prediction_scores)

        logger.debug(
            "Performing KS test with baseline (n=%d) vs current (n=%d)",
            len(baseline_scores),
            len(current_scores),
        )

        # Perform Kolmogorov-Smirnov test
        statistic, p_value = stats.ks_2samp(baseline_scores, current_scores)

        has_shift = p_value < self.threshold
        shift_type = "data_shift"

        message = (
            f"Data shift {'detected' if has_shift else 'not detected'}: "
            f"KS statistic={statistic:.4f}, p-value={p_value:.4f}, "
            f"threshold={self.threshold}"
        )

        logger.info(message)

        return ShiftDetectionResult(
            has_shift=bool(has_shift),
            shift_type=shift_type,
            p_value=float(p_value),
            test_statistic=float(statistic),
            threshold=float(self.threshold),
            message=message,
        )


class ModelPerformanceMonitor:
    """Monitor model performance metrics over time.

    This class tracks prediction statistics and can detect performance degradation.
    """

    def __init__(self, window_size: int = 1000):
        """Initialize the performance monitor.

        Args:
            window_size: Number of recent predictions to keep in memory.
        """
        self.window_size = window_size
        self._prediction_history: list[float] = []
        self._metrics_history: list[MonitoringMetrics] = []
        self._baseline_mean: float | None = None
        self._baseline_std: float | None = None

    def record_predictions(
        self,
        scores: list[float],
        user_ids: list[int] | None = None,
        item_ids: list[int] | None = None,
        custom_metrics: dict[str, Any] | None = None,
    ) -> MonitoringMetrics:
        """Record prediction scores for monitoring.

        Args:
            scores: List of prediction scores.
            user_ids: List of user IDs (optional).
            item_ids: List of item IDs (optional).
            custom_metrics: Additional custom metrics (optional).

        Returns:
            The recorded MonitoringMetrics object.
        """
        if user_ids is None:
            user_ids = []
        if item_ids is None:
            item_ids = []

        metrics = MonitoringMetrics(
            timestamp=datetime.now(),
            prediction_scores=scores,
            user_ids=user_ids,
            item_ids=item_ids,
            custom_metrics=custom_metrics,
        )

        self._prediction_history.extend(scores)
        self._metrics_history.append(metrics)

        # Keep only the most recent predictions
        if len(self._prediction_history) > self.window_size:
            self._prediction_history = self._prediction_history[-self.window_size :]

        logger.debug(
            "Recorded %d predictions, total history size: %d",
            len(scores),
            len(self._prediction_history),
        )

        return metrics

    def set_baseline(self) -> None:
        """Set current prediction statistics as baseline."""
        if not self._prediction_history:
            logger.warning("Cannot set baseline: no predictions recorded")
            return

        self._baseline_mean = np.mean(self._prediction_history)
        self._baseline_std = np.std(self._prediction_history)

        logger.info(
            "Set baseline: mean=%.4f, std=%.4f (n=%d)",
            self._baseline_mean,
            self._baseline_std,
            len(self._prediction_history),
        )

    def get_current_stats(self) -> dict[str, float]:
        """Get current prediction statistics.

        Returns:
            Dictionary with mean, std, min, max, and count of predictions.
        """
        if not self._prediction_history:
            return {}

        arr = np.array(self._prediction_history)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": len(arr),
        }

    def detect_performance_drift(self, threshold: float = 2.0) -> ShiftDetectionResult:
        """Detect performance drift using z-score.

        Args:
            threshold: Number of standard deviations for drift detection.

        Returns:
            A ShiftDetectionResult indicating whether performance drift was detected.

        Raises:
            RuntimeError: If baseline has not been set.
        """
        if self._baseline_mean is None or self._baseline_std is None:
            logger.error("Cannot detect drift: baseline not set")
            raise RuntimeError(
                "Baseline must be set before detecting performance drift."
            )

        current_stats = self.get_current_stats()
        current_mean = current_stats["mean"]

        # Calculate z-score
        z_score = abs(current_mean - self._baseline_mean) / self._baseline_std
        has_drift = z_score > threshold

        shift_type = "model_drift"
        message = (
            f"Performance drift {'detected' if has_drift else 'not detected'}: "
            f"z-score={z_score:.4f}, threshold={threshold}, "
            f"baseline_mean={self._baseline_mean:.4f}, current_mean={current_mean:.4f}"
        )

        logger.info(message)

        return ShiftDetectionResult(
            has_shift=bool(has_drift),
            shift_type=shift_type,
            p_value=float(1.0 - stats.norm.cdf(z_score)),  # Convert z-score to p-value
            test_statistic=float(z_score),
            threshold=float(threshold),
            message=message,
        )


class MonitoringService:
    """High-level service for model and data shift monitoring.

    This service orchestrates monitoring operations and provides a unified
    interface for shift detection.
    """

    def __init__(
        self,
        shift_threshold: float = 0.05,
        drift_threshold: float = 2.0,
        window_size: int = 1000,
    ):
        """Initialize the monitoring service.

        Args:
            shift_threshold: P-value threshold for data shift detection.
            drift_threshold: Z-score threshold for performance drift detection.
            window_size: Number of predictions to keep in memory.
        """
        self.data_shift_detector = DataShiftDetector(threshold=shift_threshold)
        self.performance_monitor = ModelPerformanceMonitor(window_size=window_size)
        self.drift_threshold = drift_threshold

        logger.info(
            "Initialized MonitoringService with shift_threshold=%.3f, drift_threshold=%.2f",
            shift_threshold,
            drift_threshold,
        )

    def record_predictions(
        self,
        scores: list[float],
        user_ids: list[int] | None = None,
        item_ids: list[int] | None = None,
        custom_metrics: dict[str, Any] | None = None,
    ) -> MonitoringMetrics:
        """Record predictions for monitoring.

        Args:
            scores: List of prediction scores.
            user_ids: List of user IDs (optional).
            item_ids: List of item IDs (optional).
            custom_metrics: Additional custom metrics (optional).

        Returns:
            The recorded MonitoringMetrics object.
        """
        return self.performance_monitor.record_predictions(
            scores, user_ids, item_ids, custom_metrics
        )

    def set_baselines(self) -> None:
        """Set baselines for both data shift and performance monitoring."""
        self.performance_monitor.set_baseline()

        if self.performance_monitor._metrics_history:
            # Use the most recent metrics as baseline for data shift
            latest_metrics = self.performance_monitor._metrics_history[-1]
            self.data_shift_detector.set_baseline(latest_metrics)

        logger.info("Baselines set for monitoring")

    def check_shifts(self) -> dict[str, ShiftDetectionResult]:
        """Check for both data shift and performance drift.

        Returns:
            Dictionary with results from both detectors.
        """
        results = {}

        # Check performance drift
        if self.performance_monitor._baseline_mean is not None:
            try:
                results["performance_drift"] = (
                    self.performance_monitor.detect_performance_drift(
                        self.drift_threshold
                    )
                )
            except RuntimeError:
                logger.warning("Performance drift check skipped: baseline not set")

        # Check data shift
        if self.data_shift_detector.has_baseline():
            latest_metrics = self.performance_monitor._metrics_history[-1]
            results["data_shift"] = self.data_shift_detector.detect_shift(
                latest_metrics
            )

        return results

    def get_monitoring_summary(self) -> dict[str, Any]:
        """Get a summary of monitoring status.

        Returns:
            Dictionary with monitoring statistics and status.
        """
        return {
            "performance_stats": self.performance_monitor.get_current_stats(),
            "has_baseline": self.data_shift_detector.has_baseline(),
            "window_size": self.performance_monitor.window_size,
            "history_size": len(self.performance_monitor._prediction_history),
            "metrics_history_size": len(self.performance_monitor._metrics_history),
        }
