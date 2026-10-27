import logging

import pytest

from default_pred.observability.run import tracked_run


class RecordingTracker:
    def __init__(self):
        self.calls = []

    def start_run(self, experiment, run_name):
        self.calls.append(("start", experiment, run_name))

    def log_params(self, params):
        self.calls.append(("params", dict(params)))

    def log_metrics(self, metrics):
        self.calls.append(("metrics", dict(metrics)))

    def log_artifact(self, path):
        self.calls.append(("artifact", path))

    def log_model(self, model, name):
        self.calls.append(("model", name))

    def end_run(self, status):
        self.calls.append(("end", status))


def _metrics(tracker):
    return next(c[1] for c in tracker.calls if c[0] == "metrics")


def test_tracked_run_success_logs_metrics_and_ends_with_success(caplog):
    tracker = RecordingTracker()

    with caplog.at_level(logging.INFO):
        with tracked_run(tracker, "exp", "run") as outcome:
            outcome.rows = 7

    metrics = _metrics(tracker)
    assert tracker.calls[0] == ("start", "exp", "run")
    assert tracker.calls[-1] == ("end", "success")
    assert metrics["rows_processed"] == 7.0
    assert metrics["run_success"] == 1.0
    assert metrics["duration_ms"] >= 0
    assert any("Run finished" in r.message for r in caplog.records)


def test_tracked_run_failure_marks_failed_and_reraises(caplog):
    tracker = RecordingTracker()

    with caplog.at_level(logging.INFO):
        with pytest.raises(ValueError):
            with tracked_run(tracker, "exp", "run"):
                raise ValueError("boom")

    assert tracker.calls[-1] == ("end", "failed")
    assert _metrics(tracker)["run_success"] == 0.0
    assert any("Run failed" in r.message for r in caplog.records)


def test_tracked_run_skipped_status_is_propagated():
    tracker = RecordingTracker()

    with tracked_run(tracker, "exp", "run") as outcome:
        outcome.status = "skipped"

    assert tracker.calls[-1] == ("end", "skipped")
    assert _metrics(tracker)["run_success"] == 1.0
