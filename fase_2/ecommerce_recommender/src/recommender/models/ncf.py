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
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor
    ) -> torch.Tensor:
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        x = torch.cat([user_emb, item_emb], dim=-1)
        return self.mlp(x).squeeze(-1)

    @property
    def model_name(self) -> str:
        return "ncf"
