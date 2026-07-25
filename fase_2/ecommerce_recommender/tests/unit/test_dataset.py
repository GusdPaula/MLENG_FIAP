from pathlib import Path

import numpy as np
import pandas as pd
import torch
from src.recommender.data.dataset import (
    RecommenderDataset,
    create_interaction_matrix,
    load_events,
)


def test_load_events_adds_weight(tmp_path: Path) -> None:
    csv_path = tmp_path / "events.csv"
    df = pd.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "visitorid": [1, 1, 2],
            "event": ["view", "addtocart", "transaction"],
            "itemid": [10, 20, 10],
            "transactionid": [None, None, 100],
        }
    )
    df.to_csv(csv_path, index=False)

    result = load_events(str(csv_path))

    assert "weight" in result.columns
    assert result["weight"].tolist() == [1, 2, 3]


def test_create_interaction_matrix() -> None:
    events = pd.DataFrame(
        {
            "visitorid": [1, 1, 2, 3],
            "itemid": [10, 20, 10, 30],
            "event": ["view", "view", "addtocart", "transaction"],
        }
    )

    result, user2idx, item2idx = create_interaction_matrix(events)

    assert len(user2idx) == 3
    assert len(item2idx) == 3
    assert "user_idx" in result.columns
    assert "item_idx" in result.columns


def test_recommender_dataset_size() -> None:
    events = pd.DataFrame(
        {
            "visitorid": [1, 1, 2],
            "itemid": [10, 20, 10],
            "event": ["view", "view", "view"],
        }
    )
    events["user_idx"] = [0, 0, 1]
    events["item_idx"] = [0, 1, 0]

    num_negatives = 2
    dataset = RecommenderDataset(events, num_items=3, num_negatives=num_negatives)

    num_positives = 3
    expected_size = num_positives * (1 + num_negatives)
    assert len(dataset) == expected_size


def test_recommender_dataset_item_returns_correct_types() -> None:
    events = pd.DataFrame(
        {
            "visitorid": [1, 2],
            "itemid": [10, 20],
            "event": ["view", "view"],
        }
    )
    events["user_idx"] = [0, 1]
    events["item_idx"] = [0, 1]

    dataset = RecommenderDataset(events, num_items=3, num_negatives=1)

    user, item, label = dataset[0]

    assert isinstance(user, np.int64)
    assert isinstance(item, np.int64)
    assert isinstance(label, np.float32)
    assert label in (0.0, 1.0)


def test_collect_batch() -> None:
    events = pd.DataFrame({"user_idx": [0, 1], "item_idx": [0, 1]})
    dataset = RecommenderDataset(events, num_items=3, num_negatives=1)
    from src.recommender.data.dataset import _collect_batch
    users, items, labels = _collect_batch(dataset, 0, 2)
    assert users.shape == (2,)
    assert items.shape == (2,)
    assert labels.shape == (2,)


def test_append_negatives() -> None:
    events = pd.DataFrame({"user_idx": [0], "item_idx": [0]})
    dataset = RecommenderDataset(events, num_items=5, num_negatives=2)
    rng = np.random.default_rng(42)
    samples = [(0, 0, 1.0)]
    dataset._append_negatives(samples, 0, rng)
    assert len(samples) == 3
    assert all(label == 0.0 for _, _, label in samples[1:])
    assert all((0, neg) not in dataset.positive_set for _, neg, _ in samples[1:])


def test_build_batch_tensors() -> None:
    users, items, labels = RecommenderDataset._build_batch_tensors(
        [0, 1], [1, 2], [1.0, 0.0]
    )
    assert isinstance(users, torch.Tensor)
    assert isinstance(items, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert users.tolist() == [0, 1]
    assert items.tolist() == [1, 2]
    assert labels.tolist() == [1.0, 0.0]


def test_stream_one() -> None:
    events = pd.DataFrame({"user_idx": [0], "item_idx": [1]})
    dataset = RecommenderDataset(events, num_items=5, num_negatives=2, streaming=True)
    rng = np.random.default_rng(42)
    users, items, labels = [], [], []
    dataset._stream_one(0, users, items, labels, rng)
    assert len(users) == 3
    assert labels == [1.0, 0.0, 0.0]
    assert users == [0, 0, 0]


def test_getitem_streaming() -> None:
    events = pd.DataFrame({"user_idx": [0], "item_idx": [1]})
    dataset = RecommenderDataset(events, num_items=5, num_negatives=2, streaming=True)
    user, item, label = dataset._getitem_streaming(0)
    assert user == 0
    assert item == 1
    assert label == 1.0
    u_neg, i_neg, l_neg = dataset._getitem_streaming(1)
    assert u_neg == 0
    assert l_neg == 0.0

