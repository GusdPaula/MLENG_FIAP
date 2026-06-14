"""Classic Matrix Factorization with bias terms.

This is a non-neural, non-MLP baseline: it learns a user embedding, an
item embedding, global/user/item biases, and scores pairs with the
classic Funk-SVD formula::

    score(u, i) = mu + b_u + b_i + p_u . q_i
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseRecommenderModel


class MatrixFactorizationModel(BaseRecommenderModel):
    """Bias-aware matrix factorization."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        global_bias: float = 0.0,
    ):
        """Initialize the Matrix Factorization model.

        Args:
            num_users: Number of unique users in the dataset.
            num_items: Number of unique items in the dataset.
            embedding_dim: Dimension of the embedding vectors. Defaults to 64.
            global_bias: Global bias term. Defaults to 0.0.
        """
        super().__init__(num_users, num_items, embedding_dim)

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        self.global_bias = nn.Parameter(torch.tensor(float(global_bias)))

        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights using normal distribution for embeddings and zeros for biases."""
        self._init_embeddings("normal")
        self._init_linear_layers("zeros")

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        dot = (user_emb * item_emb).sum(dim=-1)
        score = (
            self.global_bias
            + self.user_bias(user_ids).squeeze(-1)
            + self.item_bias(item_ids).squeeze(-1)
            + dot
        )
        return self.sigmoid(score)

    @property
    def model_name(self) -> str:
        return "matrix_factorization"
