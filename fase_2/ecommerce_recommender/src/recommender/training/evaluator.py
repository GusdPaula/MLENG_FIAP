"""Ranking metrics evaluation for recommender models."""

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .metrics import hit_rate_at_k, ndcg_at_k

logger = logging.getLogger(__name__)


def compute_ranking_metrics(
    model: nn.Module,
    val_dataset: torch.utils.data.Subset,
    dataset: torch.utils.data.Dataset,
    num_items: int,
    device: str,
    k: int = 10,
    sample_limit: int = 10000,
    positive_limit: int = 1000,
) -> Tuple[float, float]:
    """Compute ranking metrics (Hit Rate@K and NDCG@K) on validation set.

    Args:
        model: Trained recommender model.
        val_dataset: Validation dataset subset.
        dataset: Full dataset with samples attribute.
        num_items: Total number of items in the catalog.
        device: Device to run computations on.
        k: Rank position for metrics. Defaults to 10.
        sample_limit: Maximum number of validation samples to evaluate. Defaults to 10000.
        positive_limit: Maximum number of positive samples for ranking metrics. Defaults to 1000.

    Returns:
        Tuple of (hit_rate_at_k, ndcg_at_k) scores.
    """
    logger.info("Computing ranking metrics on validation set")

    # Handle both Subset objects (from random_split) and direct Dataset objects
    if hasattr(val_dataset, "indices"):
        # Old approach: random_split creates Subset objects
        val_indices = val_dataset.indices
        val_samples = np.array(
            [
                dataset.samples[i]
                for i in val_indices[: min(sample_limit, len(val_indices))]
            ]
        )
    else:
        # New approach: separate train/val datasets
        val_samples = np.array(
            val_dataset.samples[: min(sample_limit, len(val_dataset))]
        )

    positive_only = val_samples[val_samples[:, 2] == 1.0][:, :2].astype(np.int64)

    hr = hit_rate_at_k(
        model,
        positive_only[:positive_limit],
        num_items,
        k=k,
        device=device,
    )

    ndcg = ndcg_at_k(
        model,
        positive_only[:positive_limit],
        num_items,
        k=k,
        device=device,
    )

    logger.info(f"Hit Rate@{k}: {hr:.4f}, NDCG@{k}: {ndcg:.4f}")

    return hr, ndcg
