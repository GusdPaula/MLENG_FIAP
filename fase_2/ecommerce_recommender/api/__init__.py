"""API module for e-commerce recommendation system.

This module provides a REST API for generating recommendations using trained
recommender models. It follows the MVC architecture pattern with:
- Controllers: API endpoints and request handling
- Models: Data transfer objects and schemas
- Services: Business logic and orchestration
- Domain: Core prediction logic and strategies
"""

from .controllers.routes import app
from .exceptions import (
    InvalidInputError,
    ModelLoadError,
    PredictionError,
    PredictorNotFoundError,
)
from .models.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
    RecommendationResponse,
)
from .services.monitoring_service import MonitoringService
from .services.prediction_service import PredictionService

__all__ = [
    # FastAPI app
    "app",
    # Services
    "PredictionService",
    "MonitoringService",
    # Data models
    "PredictionRequest",
    "PredictionResponse",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "RecommendationResponse",
    # Exceptions
    "PredictionError",
    "ModelLoadError",
    "InvalidInputError",
    "PredictorNotFoundError",
]
