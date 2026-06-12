"""Model factory.

The :class:`ModelFactory` is a small registry of model classes keyed by
a string identifier (the ``type`` field in ``configs/model.yaml``). The
training pipeline asks the factory for an instance instead of importing
each model class directly, which decouples the pipeline from concrete
implementations and makes it trivial to add new models or swap them via
configuration.

To add a new model:

1. Create a subclass of :class:`BaseRecommenderModel`.
2. Decorate it with ``@ModelFactory.register("my_model")`` or call
   ``ModelFactory.register("my_model", MyModel)``.
3. Reference it from config as ``model.type: my_model``.
"""
from __future__ import annotations

from typing import Any, Callable

from .base import BaseRecommenderModel
from .gmf import GMFModel
from .matrix_factorization import MatrixFactorizationModel
from .ncf import NCFModel


class ModelFactory:
    """Registry-based factory for recommender models."""

    _registry: dict[str, type[BaseRecommenderModel]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[BaseRecommenderModel]], type[BaseRecommenderModel]]:
        """Class decorator that registers a model under ``name``."""

        def decorator(model_cls: type[BaseRecommenderModel]) -> type[BaseRecommenderModel]:
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(
        cls, model_type: str, num_users: int, num_items: int, **hyperparams: Any
    ) -> BaseRecommenderModel:
        """Instantiate a model registered under ``model_type``."""
        if model_type not in cls._registry:
            available = ", ".join(sorted(cls._registry)) or "<empty>"
            raise ValueError(
                f"Unknown model type '{model_type}'. "
                f"Available models: {available}."
            )
        model_cls = cls._registry[model_type]
        return model_cls(
            num_users=num_users,
            num_items=num_items,
            **hyperparams,
        )

    @classmethod
    def available_models(cls) -> list[str]:
        """Return the list of registered model identifiers."""
        return sorted(cls._registry)


# Built-in registrations. New models can also self-register via the
# ``@ModelFactory.register("name")`` decorator on their class.
ModelFactory._registry.update(
    {
        "ncf": NCFModel,
        "gmf": GMFModel,
        "matrix_factorization": MatrixFactorizationModel,
    }
)
