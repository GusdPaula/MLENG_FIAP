"""Ranking metrics for recommender systems evaluation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def hit_rate_at_k(
    model: nn.Module,
    test_interactions: np.ndarray,
    num_items: int,
    k: int = 10,
    device: str = "cpu",
) -> float:
    """Calculate Hit Rate@K for the model.

    For each user in the test set, generates top-K recommendations
    and checks if the actual item is in that list.

    Args:
        model: Trained recommender model.
        test_interactions: Array of (user, item) pairs for testing.
        num_items: Total number of items in the catalog.
        k: Number of top items to consider. Defaults to 10.
        device: Device to run computations on. Defaults to "cpu".

    Returns:
        Hit rate at K (proportion of users where the true item is in top-K).
    """
    model.eval()
    hits = 0
    total = 0

    users_items: dict[int, list[int]] = {}
    for user, item in test_interactions:
        users_items.setdefault(int(user), []).append(int(item))

    with torch.no_grad():
        for user_idx, true_items in users_items.items():
            user_tensor = torch.full((num_items,), user_idx, dtype=torch.long).to(
                device
            )
            item_tensor = torch.arange(num_items, dtype=torch.long).to(device)

            scores = model(user_tensor, item_tensor)
            _, top_k_indices = torch.topk(scores, k)
            top_k_set = set(top_k_indices.cpu().numpy())

            for item in true_items:
                if item in top_k_set:
                    hits += 1
                total += 1

    return hits / total if total > 0 else 0.0


def ndcg_at_k(
    model: nn.Module,
    test_interactions: np.ndarray,
    num_items: int,
    k: int = 10,
    device: str = "cpu",
) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K (NDCG@K).

    Measures not only if the item is in the top-K, but also its position.
    Higher-ranked items contribute more to the score.

    Args:
        model: Trained recommender model.
        test_interactions: Array of (user, item) pairs for testing.
        num_items: Total number of items in the catalog.
        k: Number of top items to consider. Defaults to 10.
        device: Device to run computations on. Defaults to "cpu".

    Returns:
        NDCG at K score (normalized discounted cumulative gain).
    """
    model.eval()
    ndcg_scores = []

    users_items: dict[int, list[int]] = {}
    for user, item in test_interactions:
        users_items.setdefault(int(user), []).append(int(item))

    with torch.no_grad():
        for user_idx, true_items in users_items.items():
            user_tensor = torch.full((num_items,), user_idx, dtype=torch.long).to(
                device
            )
            item_tensor = torch.arange(num_items, dtype=torch.long).to(device)

            scores = model(user_tensor, item_tensor)
            _, top_k_indices = torch.topk(scores, k)
            top_k_list = top_k_indices.cpu().numpy()

            dcg = 0.0
            for rank, item_id in enumerate(top_k_list):
                if item_id in true_items:
                    dcg += 1.0 / np.log2(rank + 2)

            ideal_dcg = sum(
                1.0 / np.log2(i + 2) for i in range(min(len(true_items), k))
            )
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
            ndcg_scores.append(ndcg)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def _get_user_top_k(
    model: nn.Module,
    user_idx: int,
    num_items: int,
    k: int,
    device: str,
) -> np.ndarray:
    """Return top-K item indices for a single user."""
    user_tensor = torch.full((num_items,), user_idx, dtype=torch.long, device=device)
    item_tensor = torch.arange(num_items, dtype=torch.long, device=device)
    scores = model(user_tensor, item_tensor)
    _, top_k_indices = torch.topk(scores, k)
    return top_k_indices.cpu().numpy()


def precision_at_k(
    model: nn.Module,
    test_interactions: np.ndarray,
    num_items: int,
    k: int = 10,
    device: str = "cpu",
) -> float:
    """Calculate Precision@K for the model.

    Measures the proportion of recommended items in the top-K that are relevant.

    Args:
        model: Trained recommender model.
        test_interactions: Array of (user, item) pairs for testing.
        num_items: Total number of items in the catalog.
        k: Number of top items to consider. Defaults to 10.
        device: Device to run computations on. Defaults to "cpu".

    Returns:
        Average precision at K across all users.
    """
    model.eval()
    precisions: list[float] = []

    users_items: dict[int, list[int]] = {}
    for user, item in test_interactions:
        users_items.setdefault(int(user), []).append(int(item))

    with torch.no_grad():
        for user_idx, true_items in users_items.items():
            top_k = _get_user_top_k(model, user_idx, num_items, k, device)
            hits = sum(1 for item_id in top_k if item_id in true_items)
            precisions.append(hits / k)

    return float(np.mean(precisions)) if precisions else 0.0


def recall_at_k(
    model: nn.Module,
    test_interactions: np.ndarray,
    num_items: int,
    k: int = 10,
    device: str = "cpu",
) -> float:
    """Calculate Recall@K for the model.

    Measures the proportion of relevant items that appear in the top-K recommendations.

    Args:
        model: Trained recommender model.
        test_interactions: Array of (user, item) pairs for testing.
        num_items: Total number of items in the catalog.
        k: Number of top items to consider. Defaults to 10.
        device: Device to run computations on. Defaults to "cpu".

    Returns:
        Average recall at K across all users.
    """
    model.eval()
    recalls: list[float] = []

    users_items: dict[int, list[int]] = {}
    for user, item in test_interactions:
        users_items.setdefault(int(user), []).append(int(item))

    with torch.no_grad():
        for user_idx, true_items in users_items.items():
            top_k = _get_user_top_k(model, user_idx, num_items, k, device)
            hits = sum(1 for item_id in top_k if item_id in true_items)
            recalls.append(hits / len(true_items))

    return float(np.mean(recalls)) if recalls else 0.0


def mrr(
    model: nn.Module,
    test_interactions: np.ndarray,
    num_items: int,
    k: int = 10,
    device: str = "cpu",
) -> float:
    """Calculate Mean Reciprocal Rank (MRR@K) for the model.

    For each user, finds the rank of the first relevant item in the top-K list
    and averages the reciprocal of those ranks.

    Args:
        model: Trained recommender model.
        test_interactions: Array of (user, item) pairs for testing.
        num_items: Total number of items in the catalog.
        k: Number of top items to consider. Defaults to 10.
        device: Device to run computations on. Defaults to "cpu".

    Returns:
        Mean reciprocal rank across all users.
    """
    model.eval()
    rr_scores: list[float] = []

    users_items: dict[int, list[int]] = {}
    for user, item in test_interactions:
        users_items.setdefault(int(user), []).append(int(item))

    with torch.no_grad():
        for user_idx, true_items in users_items.items():
            top_k = _get_user_top_k(model, user_idx, num_items, k, device)
            rr = 0.0
            for rank, item_id in enumerate(top_k):
                if item_id in true_items:
                    rr = 1.0 / (rank + 1)
                    break
            rr_scores.append(rr)

    return float(np.mean(rr_scores)) if rr_scores else 0.0
