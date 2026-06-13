"""Ranking metrics for recommender systems evaluation."""

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
