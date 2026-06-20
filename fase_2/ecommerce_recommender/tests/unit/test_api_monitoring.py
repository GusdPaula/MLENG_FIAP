"""Unit tests for API monitoring.

Tests the monitoring module for model and data shift detection.
"""

from datetime import datetime

import pytest
from api.services.monitoring_service import (
    DataShiftDetector,
    ModelPerformanceMonitor,
    MonitoringMetrics,
    MonitoringService,
    ShiftDetectionResult,
)


class TestMonitoringMetrics:
    """Tests for MonitoringMetrics dataclass."""

    def test_monitoring_metrics_creation(self):
        """Test creating MonitoringMetrics."""
        metrics = MonitoringMetrics(
            timestamp=datetime.now(),
            prediction_scores=[0.8, 0.9, 0.7],
            user_ids=[1, 2, 3],
            item_ids=[10, 20, 30],
        )

        assert len(metrics.prediction_scores) == 3
        assert len(metrics.user_ids) == 3
        assert len(metrics.item_ids) == 3
        assert metrics.custom_metrics is None

    def test_monitoring_metrics_with_custom_metrics(self):
        """Test creating MonitoringMetrics with custom metrics."""
        custom = {"accuracy": 0.85, "precision": 0.82}
        metrics = MonitoringMetrics(
            timestamp=datetime.now(),
            prediction_scores=[0.8, 0.9],
            user_ids=[1, 2],
            item_ids=[10, 20],
            custom_metrics=custom,
        )

        assert metrics.custom_metrics == custom


class TestShiftDetectionResult:
    """Tests for ShiftDetectionResult dataclass."""

    def test_shift_detection_result_creation(self):
        """Test creating ShiftDetectionResult."""
        result = ShiftDetectionResult(
            has_shift=False,
            shift_type="data_shift",
            p_value=0.8,
            test_statistic=0.1,
            threshold=0.05,
            message="No shift detected",
        )

        assert result.has_shift is False
        assert result.shift_type == "data_shift"
        assert result.p_value == 0.8
        assert result.test_statistic == 0.1


class TestDataShiftDetector:
    """Tests for DataShiftDetector."""

    def test_data_shift_detector_initialization(self):
        """Test DataShiftDetector initialization."""
        detector = DataShiftDetector(threshold=0.05)
        assert detector.threshold == 0.05
        assert detector._baseline_metrics is None

    def test_set_baseline(self):
        """Test setting baseline metrics."""
        detector = DataShiftDetector(threshold=0.05)
        baseline = MonitoringMetrics(
            timestamp=datetime.now(),
            prediction_scores=[0.8, 0.9, 0.7],
            user_ids=[1, 2, 3],
            item_ids=[10, 20, 30],
        )

        detector.set_baseline(baseline)
        assert detector._baseline_metrics == baseline

    def test_detect_shift_without_baseline(self):
        """Test detecting shift without baseline raises error."""
        detector = DataShiftDetector(threshold=0.05)
        current = MonitoringMetrics(
            timestamp=datetime.now(),
            prediction_scores=[0.8, 0.9],
            user_ids=[1, 2],
            item_ids=[10, 20],
        )

        with pytest.raises(RuntimeError):
            detector.detect_shift(current)

    def test_detect_shift_with_baseline(self):
        """Test detecting shift with baseline set."""
        detector = DataShiftDetector(threshold=0.05)
        baseline = MonitoringMetrics(
            timestamp=datetime.now(),
            prediction_scores=[0.8, 0.9, 0.7],
            user_ids=[1, 2, 3],
            item_ids=[10, 20, 30],
        )
        current = MonitoringMetrics(
            timestamp=datetime.now(),
            prediction_scores=[0.75, 0.85, 0.65],
            user_ids=[4, 5, 6],
            item_ids=[40, 50, 60],
        )

        detector.set_baseline(baseline)
        result = detector.detect_shift(current)

        assert isinstance(result, ShiftDetectionResult)
        assert result.shift_type == "data_shift"
        assert isinstance(result.has_shift, bool)
        assert isinstance(result.p_value, float)


class TestModelPerformanceMonitor:
    """Tests for ModelPerformanceMonitor."""

    def test_performance_monitor_initialization(self):
        """Test ModelPerformanceMonitor initialization."""
        monitor = ModelPerformanceMonitor(window_size=100)
        assert monitor.window_size == 100
        assert len(monitor._prediction_history) == 0
        assert monitor._baseline_mean is None
        assert monitor._baseline_std is None

    def test_record_predictions(self):
        """Test recording predictions."""
        monitor = ModelPerformanceMonitor(window_size=100)
        metrics = monitor.record_predictions(
            scores=[0.8, 0.9, 0.7],
            user_ids=[1, 2, 3],
            item_ids=[10, 20, 30],
        )

        assert len(monitor._prediction_history) == 3
        assert len(monitor._metrics_history) == 1
        assert isinstance(metrics, MonitoringMetrics)

    def test_window_size_limit(self):
        """Test that window size limits prediction history."""
        monitor = ModelPerformanceMonitor(window_size=5)

        # Record more predictions than window size
        for i in range(10):
            monitor.record_predictions(
                scores=[0.8], user_ids=[i], item_ids=[i * 10]
            )

        assert len(monitor._prediction_history) == 5

    def test_set_baseline(self):
        """Test setting baseline from current predictions."""
        monitor = ModelPerformanceMonitor(window_size=100)
        monitor.record_predictions(
            scores=[0.8, 0.9, 0.7, 0.85],
            user_ids=[1, 2, 3, 4],
            item_ids=[10, 20, 30, 40],
        )

        monitor.set_baseline()

        assert monitor._baseline_mean is not None
        assert monitor._baseline_std is not None

    def test_set_baseline_without_predictions(self):
        """Test setting baseline without predictions doesn't crash."""
        monitor = ModelPerformanceMonitor(window_size=100)
        monitor.set_baseline()  # Should not crash

        assert monitor._baseline_mean is None
        assert monitor._baseline_std is None

    def test_get_current_stats(self):
        """Test getting current statistics."""
        monitor = ModelPerformanceMonitor(window_size=100)
        monitor.record_predictions(
            scores=[0.8, 0.9, 0.7],
            user_ids=[1, 2, 3],
            item_ids=[10, 20, 30],
        )

        stats = monitor.get_current_stats()

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "count" in stats
        assert stats["count"] == 3

    def test_get_current_stats_without_predictions(self):
        """Test getting stats without predictions returns empty dict."""
        monitor = ModelPerformanceMonitor(window_size=100)
        stats = monitor.get_current_stats()

        assert stats == {}

    def test_detect_performance_drift_without_baseline(self):
        """Test detecting drift without baseline raises error."""
        monitor = ModelPerformanceMonitor(window_size=100)

        with pytest.raises(RuntimeError):
            monitor.detect_performance_drift(threshold=2.0)

    def test_detect_performance_drift_with_baseline(self):
        """Test detecting performance drift with baseline."""
        monitor = ModelPerformanceMonitor(window_size=100)

        # Set baseline
        monitor.record_predictions(
            scores=[0.8, 0.9, 0.7, 0.85],
            user_ids=[1, 2, 3, 4],
            item_ids=[10, 20, 30, 40],
        )
        monitor.set_baseline()

        # Record new predictions
        monitor.record_predictions(
            scores=[0.75, 0.88],
            user_ids=[5, 6],
            item_ids=[50, 60],
        )

        result = monitor.detect_performance_drift(threshold=2.0)

        assert isinstance(result, ShiftDetectionResult)
        assert result.shift_type == "model_drift"
        assert isinstance(result.has_shift, bool)


class TestMonitoringService:
    """Tests for MonitoringService."""

    def test_monitoring_service_initialization(self):
        """Test MonitoringService initialization."""
        service = MonitoringService(
            shift_threshold=0.05,
            drift_threshold=2.0,
            window_size=100,
        )

        assert service.data_shift_detector.threshold == 0.05
        assert service.performance_monitor.window_size == 100
        assert service.drift_threshold == 2.0

    def test_record_predictions(self):
        """Test recording predictions through service."""
        service = MonitoringService(window_size=100)
        metrics = service.record_predictions(
            scores=[0.8, 0.9, 0.7],
            user_ids=[1, 2, 3],
            item_ids=[10, 20, 30],
        )

        assert isinstance(metrics, MonitoringMetrics)
        assert len(metrics.prediction_scores) == 3

    def test_set_baselines(self):
        """Test setting baselines through service."""
        service = MonitoringService(window_size=100)

        # Record predictions
        service.record_predictions(
            scores=[0.8, 0.9, 0.7],
            user_ids=[1, 2, 3],
            item_ids=[10, 20, 30],
        )

        # Set baselines
        service.set_baselines()

        # Baselines should be set in performance monitor
        assert service.performance_monitor._baseline_mean is not None

    def test_detect_data_shift(self):
        """Test detecting data shift through service."""
        service = MonitoringService(shift_threshold=0.05)

        # Set baseline
        baseline = service.record_predictions(
            scores=[0.8, 0.9, 0.7],
            user_ids=[1, 2, 3],
            item_ids=[10, 20, 30],
        )
        service.data_shift_detector.set_baseline(baseline)

        # Record current predictions
        current = service.record_predictions(
            scores=[0.75, 0.85, 0.65],
            user_ids=[4, 5, 6],
            item_ids=[40, 50, 60],
        )

        result = service.data_shift_detector.detect_shift(current)

        assert isinstance(result, ShiftDetectionResult)
        assert result.shift_type == "data_shift"

    def test_detect_performance_drift(self):
        """Test detecting performance drift through service."""
        service = MonitoringService(drift_threshold=2.0)

        # Record predictions and set baseline
        service.record_predictions(
            scores=[0.8, 0.9, 0.7],
            user_ids=[1, 2, 3],
            item_ids=[10, 20, 30],
        )
        service.set_baselines()

        # Record new predictions
        service.record_predictions(
            scores=[0.75, 0.88],
            user_ids=[4, 5],
            item_ids=[40, 50],
        )

        # Get latest metrics to detect performance drift
        result = service.performance_monitor.detect_performance_drift(threshold=2.0)

        assert isinstance(result, ShiftDetectionResult)
        assert result.shift_type == "model_drift"

    def test_get_performance_stats(self):
        """Test getting performance stats through service."""
        service = MonitoringService(window_size=100)

        service.record_predictions(
            scores=[0.8, 0.9, 0.7],
            user_ids=[1, 2, 3],
            item_ids=[10, 20, 30],
        )

        # Check that predictions were recorded
        assert len(service.performance_monitor._metrics_history) == 1
        assert service.performance_monitor._metrics_history[0].prediction_scores == [0.8, 0.9, 0.7]
