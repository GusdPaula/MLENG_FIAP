"""Public model API.

Importing :mod:`recommender.models` registers the built-in models in
:data:`ModelFactory._registry` and exposes the abstract base class.
"""

from .base import BaseRecommenderModel
from .factory import ModelFactory
from .gmf import GMFModel
from .matrix_factorization import MatrixFactorizationModel
from .ncf import NCFModel

__all__ = [
    "BaseRecommenderModel",
    "ModelFactory",
    "NCFModel",
    "GMFModel",
    "MatrixFactorizationModel",
]
