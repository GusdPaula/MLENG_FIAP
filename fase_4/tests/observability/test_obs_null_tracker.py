from pathlib import Path

from default_pred.observability.null_tracker import NullTracker
from default_pred.observability.tracker import ExperimentTracker


def test_null_tracker_satisfies_protocol():
    assert isinstance(NullTracker(), ExperimentTracker)


def test_null_tracker_methods_are_noops():
    tracker = NullTracker()

    assert tracker.start_run("exp", "run") is None
    assert tracker.log_params({"a": 1}) is None
    assert tracker.log_metrics({"m": 1.0}) is None
    assert tracker.log_artifact(Path("x")) is None
    assert tracker.log_model(object(), "model") is None
    assert tracker.end_run("success") is None
