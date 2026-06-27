"""Small MLflow wrapper used to standardize experiment tracking.

The toolkit keeps the MLflow-specific code in one place so the training
pipeline can stay focused on data preparation, model training, and
evaluation.
"""

from __future__ import annotations

import logging
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MLflowToolkit:
    """Utility class for MLflow setup and logging."""

    tracking_uri: str | None = None
    experiment_name: str | None = None
    registry_uri: str | None = None
    offline_tracking_db: str = "mlflow.db"
    allow_offline_fallback: bool = True
    _is_offline: bool = False
    _mlflow_module: Any | None = None

    def _require_mlflow(self) -> Any:
        if self._mlflow_module is not None:
            return self._mlflow_module
        try:
            import mlflow  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency issue
            raise ImportError(
                "mlflow is not installed. Add it to the environment before using "
                "MLflowToolkit."
            ) from exc
        self._mlflow_module = mlflow
        return mlflow

    def _offline_tracking_uri(self) -> str:
        return f"sqlite:///{self.offline_tracking_db}"

    def _apply_tracking_uri(self, mlflow: Any, tracking_uri: str | None) -> None:
        from ..config import get_settings

        settings = get_settings()
        env_uri = settings.mlflow_tracking_uri
        if env_uri and env_uri != "sqlite:///mlflow.db":
            mlflow.set_tracking_uri(env_uri)
        elif tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            mlflow.set_tracking_uri(self._offline_tracking_uri())

    def setup(self, allow_fallback: bool | None = None) -> str | None:
        """Configure tracking and create/select the experiment.

        If the remote server cannot be reached and fallback is enabled,
        the toolkit switches to a local file-based MLflow store so the
        notebook can keep running offline.
        """
        mlflow = self._require_mlflow()
        allow_fallback = (
            self.allow_offline_fallback if allow_fallback is None else allow_fallback
        )

        try:
            self._apply_tracking_uri(mlflow, self.tracking_uri)
            if self.registry_uri:
                mlflow.set_registry_uri(self.registry_uri)
            if self.experiment_name:
                mlflow.set_experiment(self.experiment_name)
            self._is_offline = False
        except Exception:
            if not allow_fallback:
                raise
            self._apply_tracking_uri(mlflow, self._offline_tracking_uri())
            if self.experiment_name:
                mlflow.set_experiment(self.experiment_name)
            self._is_offline = True
        return self.experiment_name

    def get_experiment_id(self) -> str | None:
        """Return the configured experiment id, creating it if necessary."""
        mlflow = self._require_mlflow()
        if not self.experiment_name:
            return None

        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is not None:
            return experiment.experiment_id

        return mlflow.create_experiment(self.experiment_name)

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, Any] | None = None,
        nested: bool = False,
    ) -> Iterator[Any]:
        """Open an MLflow run and yield it as a context manager.

        Args:
            run_name: Name for the MLflow run.
            tags: Dictionary of tags to add to the run.
            nested: Whether the run is nested inside a parent run.

        Yields:
            The MLflow run object.
        """
        mlflow = self._require_mlflow()
        self.setup()
        run_tags = dict(tags or {})
        if self._is_offline:
            run_tags.setdefault("mlflow.mode", "offline")
        with mlflow.start_run(run_name=run_name, tags=run_tags, nested=nested) as run:
            yield run

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a parameter dictionary to the active MLflow run.

        Args:
            params: Dictionary of parameter names to values.
        """
        mlflow = self._require_mlflow()
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log a metric dictionary to the active MLflow run.

        Args:
            metrics: Dictionary of metric names to values.
            step: Step number for the metrics.
        """
        mlflow = self._require_mlflow()
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, artifact_path: str | Path) -> None:
        """Log an artifact file or folder to the active MLflow run.

        Args:
            artifact_path: Path to the artifact file or folder.
        """
        mlflow = self._require_mlflow()
        mlflow.log_artifact(str(artifact_path))

    def log_dataset(
        self,
        dataset: pd.DataFrame,
        name: str,
        source: str | None = None,
        context: str = "training",
    ) -> None:
        """Log a dataset to MLflow.

        The method prefers the native MLflow dataset API when available,
        and falls back to logging a CSV artifact plus metadata tags.

        Args:
            dataset: Pandas DataFrame to log.
            name: Name for the dataset.
            source: Source path or description of the dataset.
            context: Context in which the dataset is used. Defaults to "training".
        """
        mlflow = self._require_mlflow()
        self.setup()

        if hasattr(mlflow, "data") and hasattr(mlflow, "log_input"):
            try:
                ml_dataset = mlflow.data.from_pandas(dataset, name=name)
                mlflow.log_input(ml_dataset, context=context)
                logger.info(f"Logged dataset {name} using mlflow.data.from_pandas")
                return
            except Exception as e:
                logger.warning(
                    f"mlflow.data.from_pandas failed for {name}: {e}, falling back to artifact logging"
                )

        with tempfile.NamedTemporaryFile(
            prefix=f"{name}_", suffix=".csv", delete=False
        ) as tmp:
            csv_path = Path(tmp.name)
        dataset.to_csv(csv_path, index=False)
        mlflow.log_artifact(str(csv_path))
        mlflow.set_tag(f"dataset.{name}.context", context)
        if source:
            mlflow.set_tag(f"dataset.{name}.source", source)
        if self._is_offline:
            mlflow.set_tag("mlflow.mode", "offline")

    def register_model(
        self,
        model_uri: str,
        registered_model_name: str,
    ) -> Any:
        """Register a model artifact with the MLflow Model Registry."""
        mlflow = self._require_mlflow()
        return mlflow.register_model(model_uri=model_uri, name=registered_model_name)

    def log_pytorch_model(
        self,
        model: Any,
        name: str,
        registered_model_name: str | None = None,
        input_example: Any | None = None,
        signature: Any | None = None,
    ) -> str:
        """Log a PyTorch model and optionally register it."""
        mlflow = self._require_mlflow()
        import torch  # local import to avoid hard dependency at module load time

        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module instance")

        mlflow.pytorch.log_model(
            pytorch_model=model,
            name=name,
            registered_model_name=registered_model_name,
            input_example=input_example,
            signature=signature,
        )
        return name

    def log_sklearn_model(
        self,
        model: Any,
        name: str,
        registered_model_name: str | None = None,
        input_example: Any | None = None,
        signature: Any | None = None,
    ) -> str:
        """Log a scikit-learn model and optionally register it."""
        mlflow = self._require_mlflow()
        mlflow.sklearn.log_model(
            sk_model=model,
            name=name,
            registered_model_name=registered_model_name,
            input_example=input_example,
            signature=signature,
        )
        return name

    @property
    def is_offline(self) -> bool:
        """Return True when the toolkit is using the local file store."""
        return self._is_offline

    def get_model_version_by_run_id(self, model_name: str, run_id: str) -> str | None:
        """Retrieve the registered model version associated with a specific run_id."""
        if self._is_offline:
            return None

        self._require_mlflow()
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        try:
            filter_string = f"name='{model_name}'"
            model_versions = client.search_model_versions(filter_string)
            for mv in model_versions:
                if mv.run_id == run_id:
                    return mv.version
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(
                "Error finding model version for run %s: %s", run_id, e
            )
        return None

    def set_model_version_alias(
        self, model_name: str, version: str, alias: str
    ) -> None:
        """Assign an alias to a registered model version."""
        if self._is_offline:
            return

        self._require_mlflow()
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        client.set_registered_model_alias(name=model_name, alias=alias, version=version)

    def get_version_by_alias(self, model_name: str, alias: str) -> Any | None:
        """Retrieve the model version details for a specific alias."""
        if self._is_offline:
            return None

        self._require_mlflow()
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        try:
            return client.get_model_version_by_alias(model_name, alias)
        except Exception:
            return None

    def promote_best_to_staging(
        self,
        model_name: str,
        run_id: str,
        metric_name: str,
        higher_is_better: bool = True,
    ) -> bool:
        """Compare the metric of the new run against the current model with 'staging' alias.

        Promotes the new model by assigning the 'staging' alias if it is better,
        or if no model has the 'staging' alias.
        Returns True if promoted, False otherwise.
        """
        if self._is_offline:
            import logging

            logging.getLogger(__name__).info(
                "Offline mode active. Skipping promotion to staging."
            )
            return False

        self._require_mlflow()
        from mlflow.tracking import MlflowClient

        client = MlflowClient()

        # Get the model version for the new run
        new_version = self.get_model_version_by_run_id(model_name, run_id)
        if not new_version:
            import logging

            logging.getLogger(__name__).warning(
                "Could not find a registered model version for run %s. Skipping promotion.",
                run_id,
            )
            return False

        # Get current model version with staging alias
        staging_version_obj = self.get_version_by_alias(model_name, "staging")

        import logging

        logger = logging.getLogger(__name__)

        if not staging_version_obj:
            logger.info(
                "No model currently has the 'staging' alias. Assigning 'staging' to version %s unconditionally.",
                new_version,
            )
            self.set_model_version_alias(model_name, new_version, "staging")
            return True

        # Fetch metrics for both runs to compare
        try:
            new_run = client.get_run(run_id)
            staging_run = client.get_run(staging_version_obj.run_id)

            new_metric = new_run.data.metrics.get(metric_name)
            staging_metric = staging_run.data.metrics.get(metric_name)

            if new_metric is None:
                logger.warning(
                    "New run %s does not have metric %s. Skipping promotion.",
                    run_id,
                    metric_name,
                )
                return False

            if staging_metric is None:
                logger.info(
                    "Staging run %s does not have metric %s. Promoting new version %s.",
                    staging_version_obj.run_id,
                    metric_name,
                    new_version,
                )
                self.set_model_version_alias(model_name, new_version, "staging")
                return True

            is_better = (
                (new_metric > staging_metric)
                if higher_is_better
                else (new_metric < staging_metric)
            )

            if is_better:
                logger.info(
                    "New model version %s is better than staging version %s (%s: %s vs %s). Assigning 'staging' alias.",
                    new_version,
                    staging_version_obj.version,
                    metric_name,
                    new_metric,
                    staging_metric,
                )
                self.set_model_version_alias(model_name, new_version, "staging")
                return True
            else:
                logger.info(
                    "New model version %s is NOT better than staging version %s (%s: %s vs %s). Keeping current staging.",
                    new_version,
                    staging_version_obj.version,
                    metric_name,
                    new_metric,
                    staging_metric,
                )
                return False
        except Exception as e:
            logger.error("Failed to compare metrics and promote model: %s", e)
            return False
