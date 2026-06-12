"""Base model interface for all recommender models.

This abstract class is the contract that the ModelFactory relies on.
Every concrete model (NCF, GMF, MatrixFactorization, ...) must implement
``forward`` taking ``(user_ids, item_ids)`` and returning a score tensor.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseRecommenderModel(nn.Module, ABC):
    """Abstract base class for all recommender models."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

    @abstractmethod
    def forward(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor
    ) -> torch.Tensor:
        """Return a score tensor of shape ``(batch,)`` for each (user, item) pair."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Short identifier used by the factory and MLflow tags."""
