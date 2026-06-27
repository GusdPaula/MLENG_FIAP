"""Ranking metrics evaluation for recommender models."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .metrics import hit_rate_at_k, mrr, ndcg_at_k, precision_at_k, recall_at_k

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RankingMetrics:
    """Container for all ranking metrics."""

    hit_rate: float
    ndcg: float
    precision: float
    recall: float
    mrr: float

    def to_dict(self, k: int = 10) -> dict[str, float]:
        """Return metrics as a dictionary suitable for MLflow logging."""
        return {
            f"hit_rate_{k}": self.hit_rate,
            f"ndcg_{k}": self.ndcg,
            f"precision_{k}": self.precision,
            f"recall_{k}": self.recall,
            f"mrr_{k}": self.mrr,
        }


def compute_ranking_metrics(
    model: nn.Module,
    val_dataset: torch.utils.data.Subset,
    dataset: torch.utils.data.Dataset,
    num_items: int,
    device: str,
    k: int = 10,
    sample_limit: int = 10000,
    positive_limit: int = 1000,
) -> RankingMetrics:
    """Compute ranking metrics on validation set.

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
        RankingMetrics with all computed scores.
    """
    logger.info("Computing ranking metrics on validation set")

    val_indices = val_dataset.indices

    val_samples = np.array(
        [dataset.samples[i] for i in val_indices[: min(sample_limit, len(val_indices))]]
    )

    positive_only = val_samples[val_samples[:, 2] == 1.0][:, :2].astype(np.int64)
    limited = positive_only[:positive_limit]

    hr = hit_rate_at_k(model, limited, num_items, k=k, device=device)
    ndcg = ndcg_at_k(model, limited, num_items, k=k, device=device)
    prec = precision_at_k(model, limited, num_items, k=k, device=device)
    rec = recall_at_k(model, limited, num_items, k=k, device=device)
    mean_rr = mrr(model, limited, num_items, k=k, device=device)

    logger.info(
        f"Hit Rate@{k}: {hr:.4f}, NDCG@{k}: {ndcg:.4f}, "
        f"Precision@{k}: {prec:.4f}, Recall@{k}: {rec:.4f}, MRR@{k}: {mean_rr:.4f}"
    )

    return RankingMetrics(
        hit_rate=hr,
        ndcg=ndcg,
        precision=prec,
        recall=rec,
        mrr=mean_rr,
    )
