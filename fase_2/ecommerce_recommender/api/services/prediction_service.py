"""Prediction service for orchestrating predictions.

This module provides a high-level service class that orchestrates predictions
using the predictor factory. It follows the Single Responsibility Principle
by focusing only on orchestration, and the Dependency Inversion Principle
by depending on abstractions (BasePredictor) rather than concrete implementations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from ..domain.base_predictor import BasePredictor
from ..domain.predictor_factory import PredictorFactory
from ..exceptions import ModelLoadError
from ..models.schemas import (
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
    RecommendationResponse,
)
from .monitoring_service import MonitoringService

logger = logging.getLogger(__name__)


class PredictionService:
    """High-level service for prediction orchestration.

    This service provides a unified interface for loading models and generating
    predictions. It handles model loading, predictor creation, and prediction
    execution, following the Single Responsibility Principle.
    """

    def __init__(
        self,
        model_path: str | Path,
        predictor_type: str = "single_user",
        device: str = "cpu",
        enable_monitoring: bool = True,
        shift_threshold: float = 0.05,
        drift_threshold: float = 2.0,
        monitoring_window_size: int = 1000,
        mlflow_tracking_uri: str | None = None,
        mlflow_model_name: str | None = None,
        mlflow_model_version: str | None = None,
        mlflow_model_alias: str | None = None,
    ):
        """Initialize the prediction service.

        Args:
            model_path: Path to the saved model artifact (.pt file) as fallback.
            predictor_type: Type of predictor to use. Defaults to "single_user".
            device: Device to run predictions on ("cpu" or "cuda").
            enable_monitoring: Whether to enable model/data shift monitoring. Defaults to True.
            shift_threshold: P-value threshold for data shift detection. Defaults to 0.05.
            drift_threshold: Z-score threshold for performance drift detection. Defaults to 2.0.
            monitoring_window_size: Number of predictions to keep in memory for monitoring. Defaults to 1000.
            mlflow_tracking_uri: MLflow tracking URI for remote model loading.
            mlflow_model_name: MLflow model name for remote model loading.
            mlflow_model_version: MLflow model version for remote model loading.
            mlflow_model_alias: MLflow model alias (e.g., "champion") for remote model loading.

        Raises:
            ModelLoadError: If the model cannot be loaded.
        """
        self.model_path = Path(model_path)
        self.predictor_type = predictor_type
        self.device = device
        self._predictor: BasePredictor | None = None
        self._model_metadata: dict[str, Any] = {}
        self.enable_monitoring = enable_monitoring
        self._monitoring_service: MonitoringService | None = None
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_model_name = mlflow_model_name
        self.mlflow_model_version = mlflow_model_version
        self.mlflow_model_alias = mlflow_model_alias

        logger.info(
            "Initializing PredictionService with model_path=%s, predictor_type=%s, device=%s, monitoring=%s",
            self.model_path,
            self.predictor_type,
            self.device,
            enable_monitoring,
        )

        if self.enable_monitoring:
            self._monitoring_service = MonitoringService(
                shift_threshold=shift_threshold,
                drift_threshold=drift_threshold,
                window_size=monitoring_window_size,
            )
            logger.info("Monitoring service enabled")

        self._load_model()

    def _load_model(self) -> None:
        """Load the model and create the predictor instance.

        Tries to load from MLflow first if configured, otherwise falls back to local path.

        Raises:
            ModelLoadError: If the model cannot be loaded.
        """
        # Try loading from MLflow first if configured
        if self.mlflow_tracking_uri and (self.mlflow_model_name or self.mlflow_model_alias):
            logger.info(
                "Attempting to load model from MLflow: URI=%s, Model=%s, Version=%s, Alias=%s",
                self.mlflow_tracking_uri,
                self.mlflow_model_name,
                self.mlflow_model_version,
                self.mlflow_model_alias,
            )
            try:
                checkpoint = self._load_from_mlflow()
                logger.info("Successfully loaded model from MLflow")
                self._initialize_from_checkpoint(checkpoint)
                return
            except Exception as e:
                logger.warning(
                    "Failed to load model from MLflow: %s. Falling back to local path.",
                    e
                )

        # Fall back to local path
        logger.info("Loading model from local path: %s", self.model_path)

        if not self.model_path.exists():
            logger.error("Model file not found at %s", self.model_path)
            raise ModelLoadError(f"Model file not found at {self.model_path}")

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self._model_metadata = checkpoint.get("metadata", {})
            logger.debug("Model checkpoint loaded successfully")
            self._initialize_from_checkpoint(checkpoint)
        except Exception as e:
            logger.error("Failed to load model from local path: %s", e)
            raise ModelLoadError(f"Failed to load model: {e}") from e

    def _load_from_mlflow(self) -> dict[str, Any]:
        """Load model checkpoint from MLflow.

        Returns:
            Model checkpoint dictionary.

        Raises:
            Exception: If MLflow loading fails.
        """
        import mlflow

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        # Build model URI based on alias, version, or name
        if self.mlflow_model_alias:
            # Search for model with the specified alias
            model_uri = self._find_model_by_alias(self.mlflow_model_alias)
        elif self.mlflow_model_version:
            model_uri = f"models:/{self.mlflow_model_name}/{self.mlflow_model_version}"
        else:
            model_uri = f"models:/{self.mlflow_model_name}/latest"

        logger.info("Loading model from MLflow URI: %s", model_uri)

        # Download model to temporary location
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Try PyTorch model loading first
                model_path = mlflow.pytorch.load_model(model_uri, dst_path=temp_dir)
                checkpoint = torch.load(model_path, map_location=self.device)
            except Exception:
                # Fallback to artifact downloading
                logger.info("PyTorch model loading failed, trying artifact download")
                client = mlflow.tracking.MlflowClient()

                # Get the model version info
                if self.mlflow_model_alias:
                    # Get model name from alias
                    model_name = self._find_model_name_by_alias(self.mlflow_model_alias)
                    model_version = client.get_model_version_by_alias(model_name, self.mlflow_model_alias)
                elif self.mlflow_model_version:
                    model_version = client.get_model_version(self.mlflow_model_name, self.mlflow_model_version)
                else:
                    model_version = client.get_latest_versions(self.mlflow_model_name)[0]

                # Download artifacts
                artifacts_dir = client.download_artifacts(
                    model_version.run_id,
                    model_name if self.mlflow_model_alias else self.mlflow_model_name,
                    temp_dir
                )

                # Find the .pt file
                import os
                pt_files = [f for f in os.listdir(artifacts_dir) if f.endswith('.pt')]
                if not pt_files:
                    raise Exception("No .pt file found in artifacts") from None

                model_path = os.path.join(artifacts_dir, pt_files[0])
                checkpoint = torch.load(model_path, map_location=self.device)

        return checkpoint

    def _find_model_name_by_alias(self, alias: str) -> str:
        """Find model name by searching for the specified alias.

        Args:
            alias: The model alias to search for.

        Returns:
            Model name with the specified alias.

        Raises:
            Exception: If no model with the alias is found.
        """
        import mlflow

        client = mlflow.tracking.MlflowClient()
        registered_models = client.search_registered_models()

        for model in registered_models:
            try:
                client.get_model_version_by_alias(model.name, alias)
                return model.name
            except Exception:
                continue

        raise Exception(f"No model found with alias '{alias}'")

    def _find_model_by_alias(self, alias: str) -> str:
        """Find a model in MLflow by searching for the specified alias.

        Args:
            alias: The model alias to search for (e.g., "champion").

        Returns:
            Model URI for the model with the specified alias.

        Raises:
            Exception: If no model with the alias is found.
        """
        import mlflow

        client = mlflow.tracking.MlflowClient()

        # Get all registered models
        registered_models = client.search_registered_models()

        logger.info(f"Searching for model with alias '{alias}' across {len(registered_models)} registered models")

        # Search for the alias across all models
        for model in registered_models:
            model_name = model.name
            try:
                # Get latest version with the specified alias
                model_version = client.get_model_version_by_alias(model_name, alias)
                model_uri = f"models:/{model_name}@{alias}"
                logger.info(f"Found model '{model_name}' with alias '{alias}' (version {model_version.version})")
                return model_uri
            except Exception:
                # This model doesn't have the alias, continue searching
                continue

        raise Exception(f"No model found with alias '{alias}'")

    def _initialize_from_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Initialize model and predictor from checkpoint.

        Args:
            checkpoint: Model checkpoint dictionary.

        Raises:
            ModelLoadError: If initialization fails.
        """
        self._model_metadata = checkpoint.get("metadata", {})

        logger.debug("Model checkpoint loaded successfully")

        # Reconstruct the model
        model_type = checkpoint.get("model_type")
        if not model_type:
            logger.error("Model type not found in checkpoint")
            raise ModelLoadError("Model type not found in checkpoint.")

        from src.recommender.models.factory import ModelFactory

        # Handle different checkpoint structures
        # MLflow experiments save user2idx/item2idx instead of num_users/num_items
        user2idx = checkpoint.get("user2idx")
        item2idx = checkpoint.get("item2idx")

        if user2idx is not None and item2idx is not None:
            num_users = len(user2idx)
            num_items = len(item2idx)
            hyperparams = checkpoint.get("config", {}).get("hyperparams", {})
            logger.info("Derived num_users=%d, num_items=%d from user2idx/item2idx", num_users, num_items)
        else:
            num_users = checkpoint.get("num_users")
            num_items = checkpoint.get("num_items")
            hyperparams = checkpoint.get("hyperparams", {})

        if num_users is None or num_items is None:
            logger.error("num_users or num_items not found in checkpoint")
            raise ModelLoadError(
                "num_users or num_items not found in checkpoint."
            )

        logger.info(
            "Reconstructing model of type %s with %d users and %d items",
            model_type,
            num_users,
            num_items,
        )

        model = ModelFactory.create(
            model_type=model_type,
            num_users=num_users,
            num_items=num_items,
            **hyperparams,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        logger.debug("Model state loaded and set to eval mode")

        # Create the predictor
        user2idx = checkpoint.get("user2idx", {})
        item2idx = checkpoint.get("item2idx", {})

        logger.info(
            "Creating predictor of type '%s' with %d users and %d items",
            self.predictor_type,
            len(user2idx),
            len(item2idx),
        )

        self._predictor = PredictorFactory.create(
            predictor_type=self.predictor_type,
            model=model,
            user2idx=user2idx,
            item2idx=item2idx,
        )

        logger.info("PredictionService initialized successfully")

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Generate predictions for a single user.

        Args:
            request: The prediction request.

        Returns:
            A prediction response.

        Raises:
            RuntimeError: If the predictor is not initialized.
        """
        if self._predictor is None:
            logger.error("Predictor not initialized")
            raise RuntimeError("Predictor not initialized.")

        logger.debug("Processing prediction request for user %d", request.user_id)
        response = self._predictor.predict(request)

        if self.enable_monitoring and self._monitoring_service:
            scores = list(response.item_scores.values())
            self._monitoring_service.record_predictions(
                scores=scores,
                user_ids=[response.user_id],
                item_ids=list(response.item_scores.keys()),
            )

        return response

    def predict_batch(
        self, requests: list[PredictionRequest]
    ) -> BatchPredictionResponse:
        """Generate predictions for multiple users.

        Args:
            requests: List of prediction requests.

        Returns:
            A batch prediction response.

        Raises:
            RuntimeError: If the predictor is not initialized.
        """
        if self._predictor is None:
            logger.error("Predictor not initialized")
            raise RuntimeError("Predictor not initialized.")

        logger.info("Processing batch prediction for %d requests", len(requests))
        predictions = self._predictor.predict_batch(requests)

        if self.enable_monitoring and self._monitoring_service:
            all_scores = []
            all_user_ids = []
            all_item_ids = []

            for pred in predictions:
                all_scores.extend(pred.item_scores.values())
                all_user_ids.extend([pred.user_id] * len(pred.item_scores))
                all_item_ids.extend(pred.item_scores.keys())

            self._monitoring_service.record_predictions(
                scores=all_scores,
                user_ids=all_user_ids,
                item_ids=all_item_ids,
            )

        return BatchPredictionResponse(
            predictions=predictions,
            metadata={
                "model_type": self._model_metadata.get("model_type"),
                "predictor_type": self.predictor_type,
                "num_requests": len(requests),
            },
        )

    def recommend(self, user_id: int, k: int = 10) -> RecommendationResponse:
        """Generate top-k recommendations for a user.

        Args:
            user_id: The user ID to generate recommendations for.
            k: Number of recommendations to return.

        Returns:
            A recommendation response.

        Raises:
            RuntimeError: If the predictor is not initialized.
            InvalidInputError: If the predictor does not support recommendations.
        """
        if self._predictor is None:
            logger.error("Predictor not initialized")
            raise RuntimeError("Predictor not initialized.")

        if not hasattr(self._predictor, "recommend"):
            from ..exceptions import InvalidInputError

            logger.error(
                "Predictor type '%s' does not support recommendations",
                self.predictor_type,
            )
            raise InvalidInputError(
                f"Predictor type '{self.predictor_type}' does not support recommendations."
            )

        logger.info("Generating top-%d recommendations for user %d", k, user_id)
        response = self._predictor.recommend(user_id, k)

        if self.enable_monitoring and self._monitoring_service:
            scores = [score for _, score in response.recommendations]
            item_ids = [item_id for item_id, _ in response.recommendations]
            self._monitoring_service.record_predictions(
                scores=scores,
                user_ids=[user_id],
                item_ids=item_ids,
            )

        return response

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary containing model metadata.
        """
        info = {
            "model_path": str(self.model_path),
            "predictor_type": self.predictor_type,
            "device": self.device,
            "metadata": self._model_metadata,
        }
        logger.debug("Model info requested: %s", info)
        return info

    def reload_predictor(self, predictor_type: str) -> None:
        """Reload the service with a different predictor type.

        Args:
            predictor_type: The new predictor type to use.

        Raises:
            PredictorNotFoundError: If the predictor type is not available.
        """
        logger.info("Reloading predictor from '%s' to '%s'", self.predictor_type, predictor_type)
        self.predictor_type = predictor_type
        self._load_model()

        if self.enable_monitoring and self._monitoring_service:
            logger.info("Resetting monitoring service after predictor reload")
            self._monitoring_service = MonitoringService(
                shift_threshold=self._monitoring_service.data_shift_detector.threshold,
                drift_threshold=self._monitoring_service.drift_threshold,
                window_size=self._monitoring_service.performance_monitor.window_size,
            )

    def set_monitoring_baselines(self) -> None:
        """Set baselines for monitoring based on current prediction history.

        Raises:
            RuntimeError: If monitoring is not enabled.
        """
        if not self.enable_monitoring or self._monitoring_service is None:
            logger.error("Cannot set baselines: monitoring is not enabled")
            raise RuntimeError("Monitoring is not enabled for this service.")

        self._monitoring_service.set_baselines()
        logger.info("Monitoring baselines set successfully")

    def check_shifts(self) -> dict[str, Any]:
        """Check for model and data shifts.

        Returns:
            Dictionary with shift detection results.

        Raises:
            RuntimeError: If monitoring is not enabled.
        """
        if not self.enable_monitoring or self._monitoring_service is None:
            logger.error("Cannot check shifts: monitoring is not enabled")
            raise RuntimeError("Monitoring is not enabled for this service.")

        results = self._monitoring_service.check_shifts()
        logger.info("Shift check completed: %d results", len(results))
        return results

    def get_monitoring_summary(self) -> dict[str, Any]:
        """Get monitoring status summary.

        Returns:
            Dictionary with monitoring statistics.

        Raises:
            RuntimeError: If monitoring is not enabled.
        """
        if not self.enable_monitoring or self._monitoring_service is None:
            logger.error("Cannot get monitoring summary: monitoring is not enabled")
            raise RuntimeError("Monitoring is not enabled for this service.")

        summary = self._monitoring_service.get_monitoring_summary()
        logger.debug("Monitoring summary retrieved")
        return summary
