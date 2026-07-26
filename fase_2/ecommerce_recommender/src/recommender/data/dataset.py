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
    """Create user/item index mappings and positive interaction pairs."""
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

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)

    def __call__(self, batch: list[tuple[np.int64, np.int64, np.float32]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not batch:
            raise ValueError("Cannot collate an empty list of samples")
        users = torch.as_tensor(np.stack([b[0] for b in batch]), dtype=torch.long)
        items = torch.as_tensor(np.stack([b[1] for b in batch]), dtype=torch.long)
        labels = torch.as_tensor(np.stack([b[2] for b in batch]), dtype=torch.float32)
        return users.to(self.device), items.to(self.device), labels.to(self.device)


def _collect_batch(
    dataset: "RecommenderDataset",
    start: int,
    end: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect samples from *start* to *end* into a tensor triple."""
    users, items, labels = [], [], []
    for idx in range(start, end):
        u, i, l = dataset[idx]  # noqa: E741
        users.append(u)
        items.append(i)
        labels.append(l)
    return (
        torch.as_tensor(np.stack(users), dtype=torch.long),
        torch.as_tensor(np.stack(items), dtype=torch.long),
        torch.as_tensor(np.stack(labels), dtype=torch.float32),
    )


def make_batches(dataset: "RecommenderDataset", batch_size: int, drop_last: bool = False) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Iterate over ``dataset`` in fixed-size batches."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    n = len(dataset)
    for start in range(0, n, batch_size):
        end = start + batch_size
        if drop_last and end > n:
            break
        yield _collect_batch(dataset, start, min(end, n))


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
    ) -> None:
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
            self._append_negatives(samples, int(user_idx), self._rng)
        return samples

    def _append_negatives(
        self,
        samples: list[tuple[int, int, float]],
        user_idx: int,
        rng: np.random.Generator,
    ) -> None:
        """Sample *num_negatives* negative items and append to *samples*."""
        for _ in range(self.num_negatives):
            neg_item = int(rng.integers(0, self.num_items))
            while (user_idx, neg_item) in self.positive_set:
                neg_item = int(rng.integers(0, self.num_items))
            samples.append((user_idx, neg_item, 0.0))

    # ------------------------------------------------------------------
    # Streaming / batch-processing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_batch_tensors(
        users: list[int],
        items: list[int],
        labels: list[float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert accumulator lists into a tensor triple."""
        return (
            torch.as_tensor(users, dtype=torch.long),
            torch.as_tensor(items, dtype=torch.long),
            torch.as_tensor(labels, dtype=torch.float32),
        )

    def stream_batches(
        self,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int | None = None,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Yield batches of ``(users, items, labels)`` lazily."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self.interactions)) if shuffle else np.arange(len(self.interactions))
        threshold = batch_size * (1 + self.num_negatives)
        users, items, labels = [], [], []
        for idx in order:
            self._stream_one(idx, users, items, labels, rng)
            if len(users) >= threshold:
                yield self._build_batch_tensors(users, items, labels)
                users, items, labels = [], [], []
        if not drop_last and users:
            yield self._build_batch_tensors(users, items, labels)

    def _stream_one(
        self,
        idx: int,
        users: list[int],
        items: list[int],
        labels: list[float],
        rng: np.random.Generator,
    ) -> None:
        """Append one positive and its negatives to the accumulators."""
        user_idx, item_idx = self.interactions[idx]
        user_idx, item_idx = int(user_idx), int(item_idx)
        users.append(user_idx)
        items.append(item_idx)
        labels.append(1.0)

        for _ in range(self.num_negatives):
            neg = int(rng.integers(0, self.num_items))
            while (user_idx, neg) in self.positive_set:
                neg = int(rng.integers(0, self.num_items))
            users.append(user_idx)
            items.append(neg)
            labels.append(0.0)

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
            return np.int64(user), np.int64(item), np.float32(label)
        return self._getitem_streaming(idx)

    def _getitem_streaming(self, idx: int) -> tuple[np.int64, np.int64, np.float32]:
        """Lazy sample generation for streaming mode."""
        num_per_positive = 1 + self.num_negatives
        positive_idx = idx // num_per_positive
        slot = idx % num_per_positive

        if positive_idx >= len(self.interactions):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        user_idx, item_idx = self.interactions[positive_idx]
        user_idx, item_idx = int(user_idx), int(item_idx)

        if slot == 0:
            return np.int64(user_idx), np.int64(item_idx), np.float32(1.0)

        neg_item = int(self._rng.integers(0, self.num_items))
        while (user_idx, neg_item) in self.positive_set:
            neg_item = int(self._rng.integers(0, self.num_items))
        return np.int64(user_idx), np.int64(neg_item), np.float32(0.0)
