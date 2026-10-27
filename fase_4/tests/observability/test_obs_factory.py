import pytest

from default_pred.observability.factory import build_tracker
from default_pred.observability.mlflow_tracker import MlflowTracker
from default_pred.observability.null_tracker import NullTracker


def test_build_tracker_null():
    assert isinstance(build_tracker("null"), NullTracker)


def test_build_tracker_mlflow():
    assert isinstance(build_tracker("mlflow"), MlflowTracker)


def test_build_tracker_reads_environment(monkeypatch):
    monkeypatch.setenv("TRACKER", "null")
    assert isinstance(build_tracker(), NullTracker)


def test_build_tracker_defaults_to_mlflow(monkeypatch):
    monkeypatch.delenv("TRACKER", raising=False)
    assert isinstance(build_tracker(), MlflowTracker)


def test_build_tracker_rejects_unknown_kind():
    with pytest.raises(ValueError, match="Unknown tracker"):
        build_tracker("datadog")
