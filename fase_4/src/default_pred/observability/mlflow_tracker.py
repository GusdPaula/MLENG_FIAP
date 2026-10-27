import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn

from default_pred.observability.tracker import RunStatus

logger = logging.getLogger(__name__)


class MlflowTracker:
    """ExperimentTracker adapter backed by a local MLflow store."""

    def __init__(self, tracking_uri: str, artifact_root: Path) -> None:
        self._tracking_uri = tracking_uri
        self._artifact_root = Path(artifact_root)

    def start_run(self, experiment: str, run_name: str) -> None:
        mlflow.set_tracking_uri(self._tracking_uri)
        existing = mlflow.get_experiment_by_name(experiment)
        if existing is None:
            experiment_id = mlflow.create_experiment(
                experiment,
                artifact_location=(self._artifact_root / experiment).as_uri(),
            )
        else:
            experiment_id = existing.experiment_id
        mlflow.start_run(experiment_id=experiment_id, run_name=run_name)

    def log_params(self, params: Mapping[str, Any]) -> None:
        mlflow.log_params(dict(params))

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

    def log_artifact(self, path: Path) -> None:
        mlflow.log_artifact(str(path))

    def log_model(self, model: Any, name: str) -> None:
        mlflow.sklearn.log_model(model, name=name)

    def end_run(self, status: RunStatus) -> None:
        mlflow.set_tag("pipeline_status", status)
        mlflow_status = "FAILED" if status == "failed" else "FINISHED"
        mlflow.end_run(status=mlflow_status)
