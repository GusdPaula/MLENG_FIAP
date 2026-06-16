"""Unit tests for evaluator module."""

import pandas as pd
import torch
import torch.nn as nn
from src.recommender.data import RecommenderDataset
from src.recommender.training.evaluator import compute_ranking_metrics
from torch.utils.data import Subset


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


def test_compute_ranking_metrics_basic() -> None:
    """Test basic ranking metrics computation."""
    model = MockModel(num_users=10, num_items=5)

    # Create simple dataset with proper format
    interactions = pd.DataFrame(
        {
            "user_idx": [0, 1, 2, 0, 1],
            "item_idx": [0, 1, 2, 1, 2],
        }
    )
    dataset = RecommenderDataset(interactions, num_items=5, num_negatives=2)

    # Create validation subset
    val_indices = [0, 1, 2]  # positive samples
    val_dataset = Subset(dataset, val_indices)

    hr, ndcg = compute_ranking_metrics(
        model=model,
        val_dataset=val_dataset,
        dataset=dataset,
        num_items=5,
        device="cpu",
        k=3,
        sample_limit=10,
        positive_limit=5,
    )

    assert 0 <= hr <= 1
    assert 0 <= ndcg <= 1
    assert isinstance(hr, float)
    assert isinstance(ndcg, float)


def test_compute_ranking_metrics_k_parameter() -> None:
    """Test ranking metrics with different k values."""
    model = MockModel(num_users=10, num_items=5)
    interactions = pd.DataFrame({"user_idx": [0, 1], "item_idx": [0, 1]})
    dataset = RecommenderDataset(interactions, num_items=5, num_negatives=2)
    val_dataset = Subset(dataset, [0, 1])

    hr_k3, ndcg_k3 = compute_ranking_metrics(
        model=model,
        val_dataset=val_dataset,
        dataset=dataset,
        num_items=5,
        device="cpu",
        k=3,
        sample_limit=10,
        positive_limit=5,
    )

    hr_k5, ndcg_k5 = compute_ranking_metrics(
        model=model,
        val_dataset=val_dataset,
        dataset=dataset,
        num_items=5,
        device="cpu",
        k=5,
        sample_limit=10,
        positive_limit=5,
    )

    assert 0 <= hr_k3 <= 1
    assert 0 <= hr_k5 <= 1
    assert 0 <= ndcg_k3 <= 1
    assert 0 <= ndcg_k5 <= 1


def test_compute_ranking_metrics_sample_limits() -> None:
    """Test ranking metrics with sample limits."""
    model = MockModel(num_users=10, num_items=5)
    interactions = pd.DataFrame({"user_idx": [0, 1, 2], "item_idx": [0, 1, 2]})
    dataset = RecommenderDataset(interactions, num_items=5, num_negatives=2)
    val_dataset = Subset(dataset, [0, 1, 2])

    hr, ndcg = compute_ranking_metrics(
        model=model,
        val_dataset=val_dataset,
        dataset=dataset,
        num_items=5,
        device="cpu",
        k=3,
        sample_limit=2,  # Limit to 2 samples
        positive_limit=2,
    )

    assert 0 <= hr <= 1
    assert 0 <= ndcg <= 1
