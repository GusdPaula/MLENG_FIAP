"""Structured logging configuration for Grafana integration.

This module provides a structured JSON formatter for CloudWatch Logs integration
with Grafana, enabling better log search and visualization.
"""

import json
import logging
from datetime import datetime, timezone


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging.

    This formatter outputs logs in JSON format with structured fields for
    better integration with CloudWatch Logs and Grafana.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON.

        Args:
            record: The log record to format.

        Returns:
            JSON-formatted log string.
        """
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add custom fields if present
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        if hasattr(record, "item_ids"):
            log_data["item_ids"] = record.item_ids
        if hasattr(record, "prediction_time"):
            log_data["prediction_time"] = record.prediction_time
        if hasattr(record, "model_version"):
            log_data["model_version"] = record.model_version
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "endpoint"):
            log_data["endpoint"] = record.endpoint
        if hasattr(record, "status_code"):
            log_data["status_code"] = record.status_code
        if hasattr(record, "latency_ms"):
            log_data["latency_ms"] = record.latency_ms

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def configure_structured_logging(
    log_level: str = "INFO",
    log_group: str = "/ecs/ml-recommender-api",
    stream_name: str = "api-logs",
) -> None:
    """Configure structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_group: CloudWatch log group name.
        stream_name: CloudWatch log stream name.
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler with structured formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(StructuredFormatter())
    logger.addHandler(console_handler)

    # Try to add CloudWatch handler if available
    try:
        import watchtower

        cloudwatch_handler = watchtower.CloudWatchLogHandler(
            log_group=log_group,
            stream_name=stream_name,
            formatter=StructuredFormatter(),
        )
        logger.addHandler(cloudwatch_handler)
    except ImportError:
        # watchtower not available, skip CloudWatch logging
        pass

    logging.info("Structured logging configured", extra={"log_group": log_group})
