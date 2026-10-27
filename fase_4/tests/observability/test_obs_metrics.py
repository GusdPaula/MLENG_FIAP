import numpy as np
import pytest

from default_pred.observability.metrics import (
    as_float_metrics,
    inference_metrics,
    predicted_default_rate,
    training_params,
)


def test_predicted_default_rate():
    assert predicted_default_rate([1, 0, 1, 0]) == 0.5


def test_predicted_default_rate_empty_is_zero():
    assert predicted_default_rate([]) == 0.0


def test_inference_metrics_with_numpy_predictions():
    metrics = inference_metrics(np.array([1, 0, 0, 0]), latency_ms=12.5)

    assert metrics == {
        "rows_in": 4.0,
        "latency_ms": 12.5,
        "default_rate_predicted": 0.25,
    }


def test_inference_metrics_empty():
    assert inference_metrics([], 1.0)["rows_in"] == 0.0


def test_training_params_content():
    params = training_params(
        max_iter=100,
        class_weight=None,
        test_size=0.2,
        random_state=42,
        n_train=80,
        n_features=5,
    )

    assert params["model"] == "LogisticRegression"
    assert params["class_weight"] == "None"
    assert params["n_features"] == 5


def test_as_float_metrics_casts_numpy_values():
    result = as_float_metrics({"a": np.float64(0.5), "b": 1})

    assert result == {"a": 0.5, "b": 1.0}
    assert all(type(v) is float for v in result.values())
    assert result["a"] == pytest.approx(0.5)
