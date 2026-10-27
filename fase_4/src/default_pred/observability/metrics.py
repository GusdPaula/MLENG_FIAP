from collections.abc import Mapping, Sequence
from typing import Any


def training_params(
    *,
    max_iter: int,
    class_weight: Any,
    test_size: float,
    random_state: int,
    n_train: int,
    n_features: int,
) -> dict[str, Any]:
    """Parameters describing a training run."""
    return {
        "model": "LogisticRegression",
        "max_iter": max_iter,
        "class_weight": str(class_weight),
        "test_size": test_size,
        "random_state": random_state,
        "n_train": n_train,
        "n_features": n_features,
    }


def predicted_default_rate(predictions: Sequence[float]) -> float:
    """Share of positive (default) predictions; 0.0 for empty input."""
    if len(predictions) == 0:
        return 0.0
    return float(sum(1 for p in predictions if p == 1) / len(predictions))


def inference_metrics(
    predictions: Sequence[float], latency_ms: float
) -> dict[str, float]:
    """Operational metrics of an inference run."""
    return {
        "rows_in": float(len(predictions)),
        "latency_ms": float(latency_ms),
        "default_rate_predicted": predicted_default_rate(predictions),
    }


def as_float_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Cast metric values (e.g. numpy floats) to plain floats."""
    return {name: float(value) for name, value in metrics.items()}
