from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

RunStatus = Literal["success", "failed", "skipped"]


@runtime_checkable
class ExperimentTracker(Protocol):
    """Port used by pipelines to record runs, independent of the backend."""

    def start_run(self, experiment: str, run_name: str) -> None: ...

    def log_params(self, params: Mapping[str, Any]) -> None: ...

    def log_metrics(self, metrics: Mapping[str, float]) -> None: ...

    def log_artifact(self, path: Path) -> None: ...

    def log_model(self, model: Any, name: str) -> None: ...

    def end_run(self, status: RunStatus) -> None: ...
