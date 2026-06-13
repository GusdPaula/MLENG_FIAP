"""Generalized Matrix Factorization (GMF).

GMF is the element-wise product branch of the original NCF paper. It
learns separate embeddings for users and items and combines them with a
hadamard product, optionally projected by a linear layer.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseRecommenderModel


class GMFModel(BaseRecommenderModel):
    """Generalized Matrix Factorization."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        projection_dim: int | None = None,
        dropout: float = 0.0,
    ):
        """Initialize the GMF model.

        Args:
            num_users: Number of unique users in the dataset.
            num_items: Number of unique items in the dataset.
            embedding_dim: Dimension of the embedding vectors. Defaults to 64.
            projection_dim: Dimension of the projection layer. If None, no projection is used.
            dropout: Dropout rate for regularization. Defaults to 0.0.
        """
        super().__init__(num_users, num_items, embedding_dim)

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        if projection_dim is not None:
            self.projection = nn.Linear(embedding_dim, projection_dim)
            out_dim = projection_dim
        else:
            self.projection = None
            out_dim = embedding_dim

        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(out_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights using Xavier uniform initialization."""
        self._init_embeddings("xavier_uniform")
        self._init_linear_layers("xavier_uniform")

    def forward(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor
    ) -> torch.Tensor:
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        element_product = user_emb * item_emb
        if self.projection is not None:
            element_product = self.projection(element_product)
        element_product = self.dropout(element_product)
        score = self.output(element_product)
        return self.sigmoid(score).squeeze(-1)

    @property
    def model_name(self) -> str:
        return "gmf"
