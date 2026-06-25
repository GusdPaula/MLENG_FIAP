"""Predictor factory following the Factory pattern.

The PredictorFactory is a registry-based factory that creates predictor instances
based on a string identifier. This follows the Open/Closed Principle by allowing
new predictors to be added without modifying existing code, and the Dependency
Inversion Principle by decoupling the service layer from concrete predictor implementations.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch

from ..exceptions import PredictorNotFoundError
from .base_predictor import BasePredictor
from .predictors import BatchPredictor, SingleUserPredictor, TopKRecommendationPredictor

logger = logging.getLogger(__name__)


class PredictorFactory:
    """Registry-based factory for predictor instances.

    This factory manages the creation of predictor instances based on a string name.
    It follows the Factory pattern to decouple the service layer from concrete
    predictor implementations.
    """

    _registry: dict[str, type[BasePredictor]] = {}

    @classmethod
    def register(
        cls, name: str
    ) -> Callable[[type[BasePredictor]], type[BasePredictor]]:
        """Class decorator that registers a predictor under ``name``.

        Args:
            name: The name to register the predictor under.

        Returns:
            A decorator function that registers the predictor class.
        """

        def decorator(
            predictor_cls: type[BasePredictor],
        ) -> type[BasePredictor]:
            cls._registry[name] = predictor_cls
            logger.info("Registered predictor '%s' as %s", predictor_cls.__name__, name)
            return predictor_cls

        return decorator

    @classmethod
    def create(
        cls,
        predictor_type: str,
        model: torch.nn.Module,
        user2idx: dict[int, int],
        item2idx: dict[int, int],
        **kwargs: Any,
    ) -> BasePredictor:
        """Instantiate a predictor registered under ``predictor_type``.

        Args:
            predictor_type: The type of predictor to create.
            model: The trained recommender model.
            user2idx: Mapping from user IDs to internal indices.
            item2idx: Mapping from item IDs to internal indices.
            **kwargs: Additional arguments for predictor initialization.

        Returns:
            An instance of the requested predictor.

        Raises:
            PredictorNotFoundError: If the predictor type is not registered.
        """
        logger.info("Creating predictor of type '%s'", predictor_type)

        if predictor_type not in cls._registry:
            available = ", ".join(sorted(cls._registry)) or "<empty>"
            logger.error(
                "Unknown predictor type '%s'. Available: %s", predictor_type, available
            )
            raise PredictorNotFoundError(
                f"Unknown predictor type '{predictor_type}'. "
                f"Available predictors: {available}."
            )

        predictor_cls = cls._registry[predictor_type]
        predictor = predictor_cls(
            model=model, user2idx=user2idx, item2idx=item2idx, **kwargs
        )
        logger.debug(
            "Successfully created predictor instance of type '%s'", predictor_type
        )
        return predictor

    @classmethod
    def available_predictors(cls) -> list[str]:
        """Return the list of registered predictor identifiers.

        Returns:
            List of available predictor names.
        """
        predictors = sorted(cls._registry)
        logger.debug("Available predictors: %s", predictors)
        return predictors


# Built-in registrations. New predictors can also self-register via the
# ``@PredictorFactory.register("name")`` decorator on their class.
PredictorFactory._registry.update(
    {
        "single_user": SingleUserPredictor,
        "top_k": TopKRecommendationPredictor,
        "batch": BatchPredictor,
    }
)
