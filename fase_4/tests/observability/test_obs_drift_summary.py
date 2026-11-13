import copy
import json
import logging
from pathlib import Path

import pytest

from default_pred.observability.drift_summary import summarize_drift

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "evidently_snapshot.json"


@pytest.fixture
def snapshot():
    return json.loads(FIXTURE.read_text())


def test_fixture_has_drifted_and_stable_features(snapshot):
    summary = summarize_drift(snapshot, alert_threshold=0.5)

    drifted = {f.name for f in summary.features if f.drifted}
    stable = {f.name for f in summary.features if not f.drifted}
    assert drifted == {"a"}
    assert stable == {"b", "c"}


def test_no_alert_below_threshold(snapshot):
    summary = summarize_drift(snapshot, alert_threshold=0.5)

    assert summary.n_features == 3
    assert summary.n_drifted == 1
    assert summary.share_drifted == pytest.approx(1 / 3)
    assert summary.alert is False
    assert summary.to_metrics()["alert_triggered"] == 0.0


def test_alert_above_threshold(snapshot):
    summary = summarize_drift(snapshot, alert_threshold=0.3)

    assert summary.alert is True
    assert summary.to_metrics()["alert_triggered"] == 1.0


def test_alert_at_exact_threshold(snapshot):
    summary = summarize_drift(snapshot, alert_threshold=1 / 3)

    assert summary.alert is True


def test_method_and_score_per_feature(snapshot):
    summary = summarize_drift(snapshot, alert_threshold=0.5)
    by_name = {f.name: f for f in summary.features}

    assert by_name["a"].method == "K-S p_value"
    assert by_name["c"].method == "Z-test p_value"
    assert by_name["b"].score == pytest.approx(0.5089169658042532)
    assert by_name["a"].threshold == 0.05


def test_distance_method_drifts_at_or_above_threshold(snapshot):
    data = copy.deepcopy(snapshot)
    data["metrics"][2]["config"]["method"] = "Wasserstein distance (normed)"
    data["metrics"][2]["config"]["threshold"] = 0.1
    data["metrics"][2]["value"] = 0.2

    summary = summarize_drift(data, alert_threshold=0.5)
    by_name = {f.name: f for f in summary.features}

    assert by_name["b"].drifted is True
    assert summary.alert is True


def test_to_metrics_contains_scores_per_feature(snapshot):
    metrics = summarize_drift(snapshot, 0.5).to_metrics()

    assert metrics["n_features"] == 3.0
    assert metrics["n_drifted_features"] == 1.0
    assert "drift_score.a" in metrics
    assert "drift_score.c" in metrics


def test_empty_features_has_no_alert():
    summary = summarize_drift({"metrics": []}, alert_threshold=0.0)

    assert summary.n_features == 0
    assert summary.share_drifted == 0.0
    assert summary.alert is False


def test_missing_metrics_raises():
    with pytest.raises(ValueError, match="no 'metrics' list"):
        summarize_drift({}, 0.5)


def test_malformed_metric_raises_clear_error(snapshot):
    data = copy.deepcopy(snapshot)
    del data["metrics"][1]["config"]["method"]

    with pytest.raises(ValueError, match="Malformed ValueDrift metric"):
        summarize_drift(data, 0.5)


def test_summary_is_logged(snapshot, caplog):
    with caplog.at_level(logging.INFO):
        summarize_drift(snapshot, 0.5)

    assert any("Drift summary" in r.message for r in caplog.records)
