import logging

import numpy as np
import pandas as pd
import pytest

from default_pred.observability.instrumentation import (
    track_inference,
    track_training,
)
from default_pred.observability.null_tracker import NullTracker
from default_pred.pipelines.pipeline_drift import run as run_drift
from default_pred.utils.logging import setup_logging


@pytest.fixture
def log_file(tmp_path):
    root = logging.getLogger()
    previous = list(root.handlers)
    setup_logging(tmp_path)
    yield tmp_path / "loan_prediction.log"
    for handler in list(root.handlers):
        if handler not in previous:
            handler.close()
            root.removeHandler(handler)


def _read(log_file):
    for handler in logging.getLogger().handlers:
        handler.flush()
    return log_file.read_text(encoding="utf-8")


def _assert_lifecycle(text, experiment):
    assert f"Run started: experiment={experiment}" in text
    assert f"Run finished: experiment={experiment}" in text
    assert "status=success" in text
    assert "duration_ms=" in text
    assert "rows=" in text


def test_training_writes_lifecycle_events_to_log_file(log_file):
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    y = pd.Series([0, 1])

    def fake_train(*args, **kwargs):
        return object(), {"accuracy": 1.0}, "", ""

    track_training(
        NullTracker(),
        fake_train,
        X,
        X,
        y,
        y,
        max_iter=1,
        class_weight=None,
        test_size=0.2,
        random_state=1,
    )

    _assert_lifecycle(_read(log_file), "credit_training")


def test_inference_writes_lifecycle_events_to_log_file(log_file):
    track_inference(NullTracker(), lambda: np.array([0, 1]))

    _assert_lifecycle(_read(log_file), "credit_inference")


def test_drift_writes_lifecycle_events_to_log_file(log_file, tmp_path):
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})

    class Snapshot:
        def dict(self):
            return {"metrics": []}

    run_drift(
        tracker=NullTracker(),
        reference=frame,
        production=frame,
        detect_fn=lambda ref, cur, path: Snapshot(),
        report_path=tmp_path / "r.html",
        alert_threshold=0.5,
    )

    _assert_lifecycle(_read(log_file), "credit_drift_monitoring")


def test_failure_is_logged_with_failed_status(log_file):
    def failing():
        raise ValueError("bad payload")

    with pytest.raises(ValueError):
        track_inference(NullTracker(), failing)

    text = _read(log_file)
    assert "Run failed" in text
    assert "status=failed" in text
    assert "ValueError: bad payload" in text
