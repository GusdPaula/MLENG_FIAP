"""Unit tests for ranking metrics."""

import numpy as np
import torch
import torch.nn as nn
from src.recommender.training.metrics import hit_rate_at_k, ndcg_at_k


class MockModel(nn.Module):
    """Simple mock model for testing."""

    def __init__(self, num_users: int, num_items: int) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = 8
        self.user_embeddings = nn.Embedding(num_users, self.embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, self.embedding_dim)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)
        return (user_emb * item_emb).sum(dim=1)


def test_hit_rate_at_k_basic() -> None:
    """Test basic hit rate calculation."""
    model = MockModel(num_users=10, num_items=5)
    test_interactions = np.array(
        [
            [0, 0],  # user 0, item 0
            [1, 1],  # user 1, item 1
            [2, 2],  # user 2, item 2
        ]
    )

    hr = hit_rate_at_k(model, test_interactions, num_items=5, k=3, device="cpu")

    assert 0 <= hr <= 1
    assert isinstance(hr, float)


def test_hit_rate_at_k_empty() -> None:
    """Test hit rate with empty interactions."""
    model = MockModel(num_users=10, num_items=5)
    test_interactions = np.array([])

    hr = hit_rate_at_k(model, test_interactions, num_items=5, k=3, device="cpu")

    assert hr == 0.0


def test_hit_rate_at_k_single_user() -> None:
    """Test hit rate with single user."""
    model = MockModel(num_users=10, num_items=5)
    test_interactions = np.array([[0, 0]])

    hr = hit_rate_at_k(model, test_interactions, num_items=5, k=3, device="cpu")

    assert 0 <= hr <= 1


def test_ndcg_at_k_basic() -> None:
    """Test basic NDCG calculation."""
    model = MockModel(num_users=10, num_items=5)
    test_interactions = np.array(
        [
            [0, 0],
            [1, 1],
            [2, 2],
        ]
    )

    ndcg = ndcg_at_k(model, test_interactions, num_items=5, k=3, device="cpu")

    assert 0 <= ndcg <= 1
    assert isinstance(ndcg, float)


def test_ndcg_at_k_empty() -> None:
    """Test NDCG with empty interactions."""
    model = MockModel(num_users=10, num_items=5)
    test_interactions = np.array([])

    ndcg = ndcg_at_k(model, test_interactions, num_items=5, k=3, device="cpu")

    assert ndcg == 0.0


def test_ndcg_at_k_single_user() -> None:
    """Test NDCG with single user."""
    model = MockModel(num_users=10, num_items=5)
    test_interactions = np.array([[0, 0]])

    ndcg = ndcg_at_k(model, test_interactions, num_items=5, k=3, device="cpu")

    assert 0 <= ndcg <= 1


def test_ndcg_at_k_multiple_items_per_user() -> None:
    """Test NDCG with user having multiple true items."""
    model = MockModel(num_users=10, num_items=5)
    test_interactions = np.array(
        [
            [0, 0],
            [0, 1],
            [0, 2],
        ]
    )

    ndcg = ndcg_at_k(model, test_interactions, num_items=5, k=3, device="cpu")

    assert 0 <= ndcg <= 1


def test_ndcg_at_k_k_parameter() -> None:
    """Test NDCG with different k values."""
    model = MockModel(num_users=10, num_items=5)
    test_interactions = np.array([[0, 0], [1, 1]])

    ndcg_k3 = ndcg_at_k(model, test_interactions, num_items=5, k=3, device="cpu")
    ndcg_k5 = ndcg_at_k(model, test_interactions, num_items=5, k=5, device="cpu")

    assert 0 <= ndcg_k3 <= 1
    assert 0 <= ndcg_k5 <= 1
