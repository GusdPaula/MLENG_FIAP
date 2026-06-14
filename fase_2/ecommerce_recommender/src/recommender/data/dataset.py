"""Dataset and batch-processing utilities for the recommender.

The :class:`RecommenderDataset` turns a user/item interaction table
into ``(user, item, label)`` triples suitable for PyTorch training.
It supports two flavours of negative sampling:

* **Eager** (default, backwards compatible) - all positives and
  their negative samples are pre-computed up-front and stored in
  ``self.samples``. Simple, fast, but uses ``O(N * num_negatives)``
  memory.
* **Streaming / batch processing** - negatives are sampled lazily
  on every ``__getitem__`` call. This is the "batch processing"
  mode: it keeps memory bounded, lets the DataLoader iterate
  one batch at a time, and avoids the upfront materialization of
  the full sample table.

Use :class:`BatchCollator` to control how the underlying triples
are packed into PyTorch tensors and :func:`make_batches` for
explicit batch iteration without a DataLoader.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_events(path: str) -> pd.DataFrame:
    """Load and filter relevant events from CSV file.

    Args:
        path: Path to the events CSV file.

    Returns:
        DataFrame with events and computed weights (view=1, addtocart=2, transaction=3).
    """
    df = pd.read_csv(path)
    event_weights = {"view": 1, "addtocart": 2, "transaction": 3}
    df["weight"] = df["event"].map(event_weights)
    return df


def create_interaction_matrix(
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, int], dict[int, int]]:
    """Create user/item to index mappings and positive interaction pairs.

    Args:
        events: DataFrame containing user and item interactions.

    Returns:
        Tuple of (events with indices, user2idx mapping, item2idx mapping).
    """
    user_ids = events["visitorid"].unique()
    item_ids = events["itemid"].unique()

    user2idx = {uid: idx for idx, uid in enumerate(user_ids)}
    item2idx = {iid: idx for idx, iid in enumerate(item_ids)}

    events["user_idx"] = events["visitorid"].map(user2idx)
    events["item_idx"] = events["itemid"].map(item2idx)

    return events, user2idx, item2idx


class BatchCollator:
    """Collate a list of ``(user, item, label)`` triples into a batch.

    This is the explicit "batch" object the trainer and DataLoader
    consume. Using a small dedicated class (instead of letting the
    default collator do it implicitly) makes it easy to:

    * Move the batch to the right device up-front.
    * Reject tiny trailing batches.
    * Inspect or log the batch shape from the training loop.
    """

    def __init__(self, device: str | torch.device = "cpu"):
        self.device = torch.device(device)

    def __call__(
        self, batch: list[tuple[np.int64, np.int64, np.float32]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not batch:
            raise ValueError("Cannot collate an empty list of samples")
        users = torch.as_tensor(np.stack([b[0] for b in batch]), dtype=torch.long)
        items = torch.as_tensor(np.stack([b[1] for b in batch]), dtype=torch.long)
        labels = torch.as_tensor(np.stack([b[2] for b in batch]), dtype=torch.float32)
        return users.to(self.device), items.to(self.device), labels.to(self.device)


def make_batches(
    dataset: "RecommenderDataset",
    batch_size: int,
    drop_last: bool = False,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Iterate over ``dataset`` in fixed-size batches.

    Convenience helper for callers that want explicit batch
    iteration (e.g. notebooks, debugging) without instantiating a
    :class:`torch.utils.data.DataLoader`.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    n = len(dataset)
    for start in range(0, n, batch_size):
        end = start + batch_size
        if drop_last and end > n:
            break
        users = []
        items = []
        labels = []
        for idx in range(start, min(end, n)):
            u, i, l = dataset[idx]  # noqa: E741
            users.append(u)
            items.append(i)
            labels.append(l)
        yield (
            torch.as_tensor(np.stack(users), dtype=torch.long),
            torch.as_tensor(np.stack(items), dtype=torch.long),
            torch.as_tensor(np.stack(labels), dtype=torch.float32),
        )


class RecommenderDataset(Dataset):
    """Dataset for training with negative sampling.

    Parameters
    ----------
    interactions:
        DataFrame with at least the columns ``user_idx`` and
        ``item_idx``.
    num_items:
        Total number of items - needed as the upper bound for
        negative sampling.
    num_negatives:
        Number of negatives to draw per positive interaction.
    streaming:
        When ``True`` negatives are sampled lazily on every
        ``__getitem__`` call (memory-efficient batch processing).
        When ``False`` (the default) all samples are materialized
        up-front for backwards compatibility.
    seed:
        Optional seed used by the streaming mode so successive
        epochs see a deterministic but different set of negatives.
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        num_items: int,
        num_negatives: int = 4,
        streaming: bool = False,
        seed: int | None = None,
    ):
        self.interactions = interactions[["user_idx", "item_idx"]].values
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.streaming = streaming

        self.positive_set = set(map(tuple, self.interactions))
        self._rng = np.random.default_rng(seed)

        if not streaming:
            self.samples: list[tuple[int, int, float]] = self._generate_samples()

    # ------------------------------------------------------------------
    # Eager mode: build the whole sample table up-front
    # ------------------------------------------------------------------

    def _generate_samples(self) -> list[tuple[int, int, float]]:
        samples: list[tuple[int, int, float]] = []
        for user_idx, item_idx in self.interactions:
            samples.append((int(user_idx), int(item_idx), 1.0))
            for _ in range(self.num_negatives):
                neg_item = int(self._rng.integers(0, self.num_items))
                while (int(user_idx), neg_item) in self.positive_set:
                    neg_item = int(self._rng.integers(0, self.num_items))
                samples.append((int(user_idx), neg_item, 0.0))
        return samples

    # ------------------------------------------------------------------
    # Streaming / batch-processing helpers
    # ------------------------------------------------------------------

    def stream_batches(
        self,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int | None = None,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Yield batches of ``(users, items, labels)`` lazily.

        Each call produces a fresh negative sample for every
        positive, which is what makes it memory-efficient batch
        processing.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        rng = np.random.default_rng(seed)
        n = len(self.interactions)
        order = rng.permutation(n) if shuffle else np.arange(n)

        batch_users: list[int] = []
        batch_items: list[int] = []
        batch_labels: list[float] = []

        def _flush() -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            if not batch_users:
                return
            yield (
                torch.as_tensor(batch_users, dtype=torch.long),
                torch.as_tensor(batch_items, dtype=torch.long),
                torch.as_tensor(batch_labels, dtype=torch.float32),
            )

        for idx in order:
            user_idx, item_idx = self.interactions[idx]
            user_idx = int(user_idx)
            item_idx = int(item_idx)
            batch_users.append(user_idx)
            batch_items.append(item_idx)
            batch_labels.append(1.0)

            for _ in range(self.num_negatives):
                neg_item = int(rng.integers(0, self.num_items))
                while (user_idx, neg_item) in self.positive_set:
                    neg_item = int(rng.integers(0, self.num_items))
                batch_users.append(user_idx)
                batch_items.append(neg_item)
                batch_labels.append(0.0)

            if len(batch_users) >= batch_size * (1 + self.num_negatives):
                yield from _flush()
                batch_users = []
                batch_items = []
                batch_labels = []

        if not drop_last:
            yield from _flush()

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if self.streaming:
            return len(self.interactions) * (1 + self.num_negatives)
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[np.int64, np.int64, np.float32]:
        if not self.streaming:
            user, item, label = self.samples[idx]
            return (
                np.int64(user),
                np.int64(item),
                np.float32(label),
            )

        # Streaming / batch processing mode: lazily generate the
        # (idx // (1 + num_negatives))-th positive plus its negatives.
        num_per_positive = 1 + self.num_negatives
        positive_idx = idx // num_per_positive
        slot = idx % num_per_positive

        if positive_idx >= len(self.interactions):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        user_idx, item_idx = self.interactions[positive_idx]
        user_idx = int(user_idx)
        item_idx = int(item_idx)

        if slot == 0:
            return np.int64(user_idx), np.int64(item_idx), np.float32(1.0)

        neg_item = int(self._rng.integers(0, self.num_items))
        while (user_idx, neg_item) in self.positive_set:
            neg_item = int(self._rng.integers(0, self.num_items))
        return np.int64(user_idx), np.int64(neg_item), np.float32(0.0)
