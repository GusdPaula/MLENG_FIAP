"""MLflow logging callbacks for training."""

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def create_mlflow_logger(mlflow_toolkit: Any) -> Callable[[Any], None]:
    """Create an MLflow logging callback for training epochs.

    Args:
        mlflow_toolkit: MLflowToolkit instance for logging metrics.

    Returns:
        Callback function that logs metrics after each epoch.
    """

    def mlflow_logger(epoch_result: Any) -> None:
        """Log training metrics to MLflow.

        Args:
            epoch_result: EpochResult object with training metrics.
        """
        metrics = {
            "train_loss": float(epoch_result.train_loss),
            "learning_rate": float(epoch_result.learning_rate),
            **{
                k: float(v)
                for k, v in epoch_result.eval_metrics.items()
            },
        }
        mlflow_toolkit.log_metrics(metrics, step=epoch_result.epoch)
        logger.debug(f"Logged metrics for epoch {epoch_result.epoch}")

    return mlflow_logger
