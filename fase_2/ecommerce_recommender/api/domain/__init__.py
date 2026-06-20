"""Domain module for core prediction logic.

This module contains the predictor interfaces and implementations.
"""

from .base_predictor import BasePredictor
from .predictor_factory import PredictorFactory
from .predictors import (
    BatchPredictor,
    SingleUserPredictor,
    TopKRecommendationPredictor,
)

__all__ = [
    "BasePredictor",
    "PredictorFactory",
    "SingleUserPredictor",
    "TopKRecommendationPredictor",
    "BatchPredictor",
]
