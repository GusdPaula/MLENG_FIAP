import numpy as np
import pandas as pd
import pytest

from default_pred.observability.instrumentation import (
    track_inference,
    track_training,
)
from default_pred.observability.null_tracker import NullTracker


class SpyTracker(NullTracker):
    def __init__(self):
        self.experiment = None
        self.params = {}
        self.metrics = {}
        self.models = []
        self.status = None

    def start_run(self, experiment, run_name):
        self.experiment = experiment

    def log_params(self, params):
        self.params.update(params)

    def log_metrics(self, metrics):
        self.metrics.update(metrics)

    def log_model(self, model, name):
        self.models.append((model, name))

    def end_run(self, status):
        self.status = status


def _fake_train(X_train, X_test, y_train, y_test, max_iter, class_weight):
    metrics = {
        "accuracy": np.float64(0.9),
        "precision": 0.8,
        "recall": 0.7,
        "f1_score": 0.75,
        "roc_auc": 0.95,
    }
    return "model-object", metrics, "report", "matrix"


@pytest.fixture
def split():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.0, 1.0, 0.0, 1.0]})
    y = pd.Series([0, 1, 0, 1])
    return X, X.head(2), y, y.head(2)


def test_track_training_records_params_metrics_and_model(split):
    tracker = SpyTracker()
    X_train, X_test, y_train, y_test = split

    model = track_training(
        tracker,
        _fake_train,
        X_train,
        X_test,
        y_train,
        y_test,
        max_iter=500,
        class_weight=None,
        test_size=0.2,
        random_state=42,
    )

    assert model == "model-object"
    assert tracker.experiment == "credit_training"
    assert tracker.params["max_iter"] == 500
    assert tracker.params["n_features"] == 2
    assert set(tracker.metrics) >= {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "duration_ms",
    }
    assert type(tracker.metrics["accuracy"]) is float
    assert tracker.models == [("model-object", "model")]
    assert tracker.status == "success"
    assert tracker.metrics["rows_processed"] == 4.0


def test_track_training_failure_is_recorded_and_reraised(split):
    tracker = SpyTracker()
    X_train, X_test, y_train, y_test = split

    def failing(*args, **kwargs):
        raise RuntimeError("no convergence")

    with pytest.raises(RuntimeError):
        track_training(
            tracker,
            failing,
            X_train,
            X_test,
            y_train,
            y_test,
            max_iter=1,
            class_weight=None,
            test_size=0.2,
            random_state=1,
        )

    assert tracker.status == "failed"
    assert tracker.models == []


def test_track_inference_records_operational_metrics():
    tracker = SpyTracker()

    predictions = track_inference(tracker, lambda: np.array([1, 0, 0, 0]))

    assert list(predictions) == [1, 0, 0, 0]
    assert tracker.experiment == "credit_inference"
    assert tracker.metrics["rows_in"] == 4.0
    assert tracker.metrics["default_rate_predicted"] == 0.25
    assert tracker.metrics["latency_ms"] >= 0
    assert tracker.metrics["rows_processed"] == 4.0
    assert tracker.status == "success"


def test_track_inference_failure_is_recorded_and_reraised():
    tracker = SpyTracker()

    def failing():
        raise FileNotFoundError("payload missing")

    with pytest.raises(FileNotFoundError):
        track_inference(tracker, failing)

    assert tracker.status == "failed"
    assert tracker.metrics["run_success"] == 0.0
    assert "latency_ms" not in tracker.metrics
