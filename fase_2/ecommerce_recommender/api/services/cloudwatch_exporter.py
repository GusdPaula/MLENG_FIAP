"""CloudWatch Metrics exporter for model monitoring.

This module exports model shift and performance metrics to AWS CloudWatch
for visualization in Grafana dashboards.
"""

import logging
from datetime import datetime
from typing import Any

import boto3

logger = logging.getLogger(__name__)


class CloudWatchMetricsExporter:
    """Export monitoring metrics to AWS CloudWatch."""

    def __init__(self, region: str = "us-east-1", namespace: str = "ML-Recommender"):
        """Initialize CloudWatch metrics exporter.

        Args:
            region: AWS region for CloudWatch.
            namespace: CloudWatch metrics namespace.
        """
        import boto3.session

        session = boto3.session.Session()
        self.cloudwatch = session.client("cloudwatch", region_name=region)
        self.namespace = namespace

    def put_metric(
        self,
        metric_name: str,
        value: float,
        dimensions: list[dict[str, str]] | None = None,
        unit: str = "None",
    ) -> None:
        """Put a single metric to CloudWatch.

        Args:
            metric_name: Name of the metric.
            value: Metric value.
            dimensions: List of dimension key-value pairs.
            unit: Unit of the metric (Count, None, Seconds, etc.).
        """
        if dimensions is None:
            dimensions = []

        try:
            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=[
                    {
                        "MetricName": metric_name,
                        "Value": value,
                        "Unit": unit,
                        "Dimensions": dimensions,
                        "Timestamp": datetime.utcnow(),
                    }
                ],
            )
            logger.debug(f"Exported metric {metric_name}={value}")
        except Exception as e:
            logger.error(f"Failed to export metric {metric_name}: {e}")

    def export_shift_detection_result(
        self, result: Any, metric_type: str = "DataShift"
    ) -> None:
        """Export shift detection results to CloudWatch.

        Args:
            result: ShiftDetectionResult object.
            metric_type: Type of shift (DataShift or ModelDrift).
        """
        dimensions = [{"Name": "ShiftType", "Value": metric_type}]

        # Export p-value (lower = more significant shift)
        self.put_metric(
            metric_name="ShiftPValue",
            value=result.p_value,
            dimensions=dimensions,
            unit="None",
        )

        # Export test statistic
        self.put_metric(
            metric_name="ShiftTestStatistic",
            value=result.test_statistic,
            dimensions=dimensions,
            unit="None",
        )

        # Export shift detection as binary (1 = shift detected, 0 = no shift)
        self.put_metric(
            metric_name="ShiftDetected",
            value=1.0 if result.has_shift else 0.0,
            dimensions=dimensions,
            unit="Count",
        )

        logger.info(f"Exported {metric_type} metrics to CloudWatch")

    def export_performance_stats(
        self, stats: dict[str, float], model_version: str = "unknown"
    ) -> None:
        """Export performance statistics to CloudWatch.

        Args:
            stats: Dictionary with mean, std, min, max, count.
            model_version: Model version identifier.
        """
        dimensions = [{"Name": "ModelVersion", "Value": model_version}]

        if "mean" in stats:
            self.put_metric(
                metric_name="PredictionScoreMean",
                value=stats["mean"],
                dimensions=dimensions,
                unit="None",
            )

        if "std" in stats:
            self.put_metric(
                metric_name="PredictionScoreStd",
                value=stats["std"],
                dimensions=dimensions,
                unit="None",
            )

        if "min" in stats:
            self.put_metric(
                metric_name="PredictionScoreMin",
                value=stats["min"],
                dimensions=dimensions,
                unit="None",
            )

        if "max" in stats:
            self.put_metric(
                metric_name="PredictionScoreMax",
                value=stats["max"],
                dimensions=dimensions,
                unit="None",
            )

        if "count" in stats:
            self.put_metric(
                metric_name="PredictionCount",
                value=float(stats["count"]),
                dimensions=dimensions,
                unit="Count",
            )

        logger.info(f"Exported performance stats to CloudWatch")

    def export_api_metrics(
        self,
        request_count: int,
        error_count: int,
        latency_ms: float,
        endpoint: str = "predict",
    ) -> None:
        """Export API performance metrics to CloudWatch.

        Args:
            request_count: Number of requests processed.
            error_count: Number of errors encountered.
            latency_ms: Average request latency in milliseconds.
            endpoint: API endpoint name.
        """
        dimensions = [{"Name": "Endpoint", "Value": endpoint}]

        # Request count
        self.put_metric(
            metric_name="RequestCount",
            value=float(request_count),
            dimensions=dimensions,
            unit="Count",
        )

        # Error count
        self.put_metric(
            metric_name="ErrorCount",
            value=float(error_count),
            dimensions=dimensions,
            unit="Count",
        )

        # Calculate error rate
        if request_count > 0:
            error_rate = (error_count / request_count) * 100
            self.put_metric(
                metric_name="ErrorRate",
                value=error_rate,
                dimensions=dimensions,
                unit="Percent",
            )

        # Latency
        self.put_metric(
            metric_name="RequestLatency",
            value=latency_ms,
            dimensions=dimensions,
            unit="Milliseconds",
        )

        logger.info(f"Exported API metrics to CloudWatch")
