import copy
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from default_pred.observability.null_tracker import NullTracker
from default_pred.pipelines.pipeline_drift import (
    prepare_production_features,
    run,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "evidently_snapshot.json"


class FakeSnapshot:
    def __init__(self, payload):
        self._payload = payload

    def dict(self):
        return self._payload


class SpyTracker(NullTracker):
    def __init__(self):
        self.params = {}
        self.metrics = {}
        self.artifacts = []
        self.status = None

    def log_params(self, params):
        self.params.update(params)

    def log_metrics(self, metrics):
        self.metrics.update(metrics)

    def log_artifact(self, path):
        self.artifacts.append(path)

    def end_run(self, status):
        self.status = status


@pytest.fixture
def payload():
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def frames():
    reference = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
    production = pd.DataFrame({"a": [9.0, 8.0, 7.0], "b": [1.0, 2.0, 3.0]})
    return reference, production


def _detector(snapshot, write=True):
    def detect(reference, production, path):
        if write:
            path.write_text("<html>report</html>")
        return snapshot

    return detect


def test_run_alerts_and_logs_artifact(frames, payload, tmp_path, caplog):
    tracker = SpyTracker()
    report = tmp_path / "report.html"
    reference, production = frames

    with caplog.at_level(logging.WARNING):
        summary = run(
            tracker=tracker,
            reference=reference,
            production=production,
            detect_fn=_detector(FakeSnapshot(payload)),
            report_path=report,
            alert_threshold=0.3,
        )

    assert summary.alert is True
    assert tracker.metrics["alert_triggered"] == 1.0
    assert tracker.metrics["n_drifted_features"] == 1.0
    assert tracker.artifacts == [report]
    assert tracker.status == "success"
    assert tracker.params["production_rows"] == 3
    assert any("DRIFT ALERT" in r.message for r in caplog.records)


def test_run_without_alert(frames, payload, tmp_path, caplog):
    tracker = SpyTracker()
    reference, production = frames

    with caplog.at_level(logging.WARNING):
        summary = run(
            tracker=tracker,
            reference=reference,
            production=production,
            detect_fn=_detector(FakeSnapshot(payload)),
            report_path=tmp_path / "r.html",
            alert_threshold=0.5,
        )

    assert summary.alert is False
    assert tracker.metrics["alert_triggered"] == 0.0
    assert not any("DRIFT ALERT" in r.message for r in caplog.records)


def test_run_does_not_attach_missing_report(frames, payload, tmp_path):
    tracker = SpyTracker()
    reference, production = frames

    run(
        tracker=tracker,
        reference=reference,
        production=production,
        detect_fn=_detector(FakeSnapshot(payload), write=False),
        report_path=tmp_path / "missing.html",
        alert_threshold=0.5,
    )

    assert tracker.artifacts == []


def test_run_skips_when_sample_is_too_small(frames, tmp_path, caplog):
    tracker = SpyTracker()
    reference, production = frames
    calls = []

    def detect(*args):
        calls.append(args)

    with caplog.at_level(logging.WARNING):
        summary = run(
            tracker=tracker,
            reference=reference,
            production=production.head(1),
            detect_fn=detect,
            report_path=tmp_path / "r.html",
            alert_threshold=0.5,
        )

    assert summary is None
    assert calls == []
    assert tracker.status == "skipped"
    assert not (tmp_path / "r.html").exists()
    assert any("skipped" in r.message for r in caplog.records)


def test_run_skips_when_detector_returns_nothing(frames, tmp_path):
    tracker = SpyTracker()
    reference, production = frames

    summary = run(
        tracker=tracker,
        reference=reference,
        production=production,
        detect_fn=_detector(None),
        report_path=tmp_path / "r.html",
        alert_threshold=0.5,
    )

    assert summary is None
    assert tracker.status == "skipped"


def test_run_marks_failure_and_reraises(frames, tmp_path):
    tracker = SpyTracker()
    reference, production = frames

    def detect(*args):
        raise RuntimeError("evidently exploded")

    with pytest.raises(RuntimeError, match="evidently exploded"):
        run(
            tracker=tracker,
            reference=reference,
            production=production,
            detect_fn=detect,
            report_path=tmp_path / "r.html",
            alert_threshold=0.5,
        )

    assert tracker.status == "failed"
    assert tracker.metrics["run_success"] == 0.0


def test_run_fails_on_malformed_snapshot(frames, payload, tmp_path):
    tracker = SpyTracker()
    reference, production = frames
    broken = copy.deepcopy(payload)
    del broken["metrics"][1]["config"]["method"]

    with pytest.raises(ValueError):
        run(
            tracker=tracker,
            reference=reference,
            production=production,
            detect_fn=_detector(FakeSnapshot(broken)),
            report_path=tmp_path / "r.html",
            alert_threshold=0.5,
        )

    assert tracker.status == "failed"


def test_prepare_production_features_scales_and_aligns():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
    scaler = StandardScaler().fit(train[["x"]])
    feature_columns = ["x", "y", "dummy"]
    raw = pd.DataFrame({"x": [2.0, 4.0], "y": [5.0, 6.0]})

    result = prepare_production_features(raw, scaler, feature_columns, lambda d: d)

    assert list(result.columns) == feature_columns
    assert result["dummy"].tolist() == [0, 0]
    assert result["y"].tolist() == [5.0, 6.0]
    expected_x = scaler.transform(raw[["x"]]).ravel()
    assert np.allclose(result["x"].to_numpy(), expected_x)


def test_prepare_production_features_without_scaler_columns():
    class NoColumnsScaler:
        pass

    raw = pd.DataFrame({"x": [1.0, 2.0]})

    result = prepare_production_features(raw, NoColumnsScaler(), ["x"], lambda d: d)

    assert result["x"].tolist() == [1.0, 2.0]
