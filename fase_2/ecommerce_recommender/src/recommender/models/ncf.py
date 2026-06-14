"""Neural Collaborative Filtering (He et al., 2017).

The model concatenates user and item embeddings and feeds them into an
MLP whose output is squashed through a Sigmoid to produce a
probability-like score in ``[0, 1]``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseRecommenderModel


class NCFModel(BaseRecommenderModel):
    """Neural Collaborative Filtering (MLP variant)."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        hidden_layers: list[int] | tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.2,
    ):
        """Initialize the NCF model.

        Args:
            num_users: Number of unique users in the dataset.
            num_items: Number of unique items in the dataset.
            embedding_dim: Dimension of the embedding vectors. Defaults to 64.
            hidden_layers: List of hidden layer sizes for the MLP. Defaults to (128, 64, 32).
            dropout: Dropout rate for regularization. Defaults to 0.2.
        """
        super().__init__(num_users, num_items, embedding_dim)

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        layers: list[nn.Module] = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights using Xavier uniform for embeddings and Kaiming uniform for linear layers."""
        self._init_embeddings("xavier_uniform")
        self._init_linear_layers("kaiming_uniform", nonlinearity="relu")

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        x = torch.cat([user_emb, item_emb], dim=-1)
        return self.mlp(x).squeeze(-1)

    @property
    def model_name(self) -> str:
        return "ncf"
