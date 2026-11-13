import os

from default_pred.config import MLFLOW_ARTIFACT_ROOT, MLFLOW_TRACKING_URI
from default_pred.observability.mlflow_tracker import MlflowTracker
from default_pred.observability.null_tracker import NullTracker
from default_pred.observability.tracker import ExperimentTracker

TRACKER_ENV_VAR = "TRACKER"


def build_tracker(kind: str | None = None) -> ExperimentTracker:
    """Build a tracker from `kind` or the TRACKER env var (null|mlflow)."""
    selected = (kind or os.environ.get(TRACKER_ENV_VAR, "mlflow")).strip().lower()
    if selected == "mlflow":
        return MlflowTracker(MLFLOW_TRACKING_URI, MLFLOW_ARTIFACT_ROOT)
    if selected == "null":
        return NullTracker()
    raise ValueError(f"Unknown tracker '{selected}'. Use 'mlflow' or 'null'.")
