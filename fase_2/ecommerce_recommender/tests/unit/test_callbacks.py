"""Unit tests for MLflow callbacks module."""

from dataclasses import dataclass

from src.recommender.mlflow_toolkit.callbacks import create_mlflow_logger


@dataclass
class MockEpochResult:
    """Mock epoch result for testing."""
    epoch: int
    train_loss: float
    learning_rate: float
    eval_metrics: dict


class MockMLflowToolkit:
    """Mock MLflow toolkit for testing."""

    def __init__(self):
        self.logged_metrics = []

    def log_metrics(self, metrics: dict, step: int):
        """Mock log metrics method."""
        self.logged_metrics.append((metrics, step))


def test_create_mlflow_logger_basic():
    """Test basic MLflow logger creation and execution."""
    mlflow_toolkit = MockMLflowToolkit()
    logger = create_mlflow_logger(mlflow_toolkit)

    epoch_result = MockEpochResult(
        epoch=1,
        train_loss=0.5,
        learning_rate=0.001,
        eval_metrics={"auc_roc": 0.85, "ndcg_at_10": 0.75},
    )

    logger(epoch_result)

    assert len(mlflow_toolkit.logged_metrics) == 1
    logged_metrics, step = mlflow_toolkit.logged_metrics[0]
    assert step == 1
    assert logged_metrics["train_loss"] == 0.5
    assert logged_metrics["learning_rate"] == 0.001
    assert logged_metrics["auc_roc"] == 0.85
    assert logged_metrics["ndcg_at_10"] == 0.75


def test_create_mlflow_logger_empty_metrics():
    """Test MLflow logger with empty evaluation metrics."""
    mlflow_toolkit = MockMLflowToolkit()
    logger = create_mlflow_logger(mlflow_toolkit)

    epoch_result = MockEpochResult(
        epoch=1,
        train_loss=0.5,
        learning_rate=0.001,
        eval_metrics={},
    )

    logger(epoch_result)

    assert len(mlflow_toolkit.logged_metrics) == 1
    logged_metrics, step = mlflow_toolkit.logged_metrics[0]
    assert step == 1
    assert logged_metrics["train_loss"] == 0.5
    assert logged_metrics["learning_rate"] == 0.001
    assert len(logged_metrics) == 2  # Only train_loss and learning_rate


def test_create_mlflow_logger_multiple_epochs():
    """Test MLflow logger logging multiple epochs."""
    mlflow_toolkit = MockMLflowToolkit()
    logger = create_mlflow_logger(mlflow_toolkit)

    for epoch in range(1, 4):
        epoch_result = MockEpochResult(
            epoch=epoch,
            train_loss=0.5 / epoch,
            learning_rate=0.001,
            eval_metrics={"auc_roc": 0.85 + epoch * 0.01},
        )
        logger(epoch_result)

    assert len(mlflow_toolkit.logged_metrics) == 3
    for i, (logged_metrics, step) in enumerate(mlflow_toolkit.logged_metrics, 1):
        assert step == i
        assert logged_metrics["train_loss"] == 0.5 / i


def test_create_mlflow_logger_metric_types():
    """Test that metrics are converted to float."""
    mlflow_toolkit = MockMLflowToolkit()
    logger = create_mlflow_logger(mlflow_toolkit)

    epoch_result = MockEpochResult(
        epoch=1,
        train_loss=0.5,
        learning_rate=0.001,
        eval_metrics={"auc_roc": 0.85, "ndcg_at_10": 0.75},
    )

    logger(epoch_result)

    logged_metrics, _ = mlflow_toolkit.logged_metrics[0]
    assert isinstance(logged_metrics["train_loss"], float)
    assert isinstance(logged_metrics["learning_rate"], float)
    assert isinstance(logged_metrics["auc_roc"], float)
    assert isinstance(logged_metrics["ndcg_at_10"], float)
