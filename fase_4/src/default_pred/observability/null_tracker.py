from collections.abc import Mapping
from pathlib import Path
from typing import Any

from default_pred.observability.tracker import RunStatus


class NullTracker:
    """No-op tracker for tests and offline execution."""

    def start_run(self, experiment: str, run_name: str) -> None:
        return None

    def log_params(self, params: Mapping[str, Any]) -> None:
        return None

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        return None

    def log_artifact(self, path: Path) -> None:
        return None

    def log_model(self, model: Any, name: str) -> None:
        return None

    def end_run(self, status: RunStatus) -> None:
        return None
