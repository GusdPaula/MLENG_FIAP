import numpy as np
import pytest
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression

from default_pred.observability.mlflow_tracker import MlflowTracker
from default_pred.observability.run import tracked_run


@pytest.fixture
def tracker(tmp_path):
    return MlflowTracker(
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        artifact_root=tmp_path / "mlruns",
    )


def _only_run(tracker, experiment):
    client = MlflowClient(tracking_uri=tracker._tracking_uri)
    exp = client.get_experiment_by_name(experiment)
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1
    return client, runs[0]


def test_mlflow_tracker_persists_params_metrics_and_artifacts(tracker, tmp_path):
    artifact = tmp_path / "report.html"
    artifact.write_text("<html></html>")

    with tracked_run(tracker, "exp_a", "run_a") as outcome:
        outcome.rows = 3
        tracker.log_params({"alpha": 0.5})
        tracker.log_metrics({"score": 0.9})
        tracker.log_artifact(artifact)

    client, run = _only_run(tracker, "exp_a")
    assert run.data.params["alpha"] == "0.5"
    assert run.data.metrics["score"] == pytest.approx(0.9)
    assert run.data.metrics["rows_processed"] == 3.0
    assert run.data.tags["pipeline_status"] == "success"
    assert run.info.status == "FINISHED"
    assert [a.path for a in client.list_artifacts(run.info.run_id)] == ["report.html"]


def test_mlflow_tracker_marks_failed_run(tracker):
    with pytest.raises(RuntimeError):
        with tracked_run(tracker, "exp_f", "run_f"):
            raise RuntimeError("boom")

    _, run = _only_run(tracker, "exp_f")
    assert run.info.status == "FAILED"
    assert run.data.tags["pipeline_status"] == "failed"


def test_mlflow_tracker_reuses_existing_experiment(tracker):
    for _ in range(2):
        with tracked_run(tracker, "exp_r", "run"):
            pass

    client = MlflowClient(tracking_uri=tracker._tracking_uri)
    exp = client.get_experiment_by_name("exp_r")
    assert len(client.search_runs([exp.experiment_id])) == 2


def test_mlflow_tracker_logs_model(tracker):
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    model = LogisticRegression().fit(features, [0, 0, 1, 1])

    with tracked_run(tracker, "exp_m", "run_m"):
        tracker.log_model(model, "model")

    client = MlflowClient(tracking_uri=tracker._tracking_uri)
    exp = client.get_experiment_by_name("exp_m")
    assert client.search_logged_models([exp.experiment_id])
