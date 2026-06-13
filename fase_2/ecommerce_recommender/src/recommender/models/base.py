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
        """Initialize the base recommender model.

        Args:
            num_users: Number of unique users in the dataset.
            num_items: Number of unique items in the dataset.
            embedding_dim: Dimension of the embedding vectors. Defaults to 64.
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

    def _init_embeddings(self, init_type: str = "xavier_uniform") -> None:
        """Initialize embedding layers using specified initialization method.

        Args:
            init_type: Initialization method ('xavier_uniform', 'normal', 'zeros').
        """
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                if init_type == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                elif init_type == "normal":
                    nn.init.normal_(module.weight, std=0.01)
                elif init_type == "zeros":
                    nn.init.zeros_(module.weight)

    def _init_linear_layers(self, init_type: str = "xavier_uniform", nonlinearity: str = "relu") -> None:
        """Initialize linear layers using specified initialization method.

        Args:
            init_type: Initialization method ('xavier_uniform', 'kaiming_uniform', 'zeros').
            nonlinearity: Nonlinearity function for Kaiming initialization. Defaults to "relu".
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init_type == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                elif init_type == "kaiming_uniform":
                    nn.init.kaiming_uniform_(module.weight, nonlinearity=nonlinearity)
                elif init_type == "zeros":
                    nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @abstractmethod
    def forward(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor
    ) -> torch.Tensor:
        """Return a score tensor of shape ``(batch,)`` for each (user, item) pair."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Short identifier used by the factory and MLflow tags."""
