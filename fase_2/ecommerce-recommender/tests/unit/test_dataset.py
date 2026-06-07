import pandas as pd
import numpy as np
from recommender.data.dataset import load_events, create_interaction_matrix, RecommenderDataset


def test_load_events_adds_weight(tmp_path):
    csv_path = tmp_path / "events.csv"
    df = pd.DataFrame({
        "timestamp": [1, 2, 3],
        "visitorid": [1, 1, 2],
        "event": ["view", "addtocart", "transaction"],
        "itemid": [10, 20, 10],
        "transactionid": [None, None, 100],
    })
    df.to_csv(csv_path, index=False)

    result = load_events(str(csv_path))

    assert "weight" in result.columns
    assert result["weight"].tolist() == [1, 2, 3]


def test_create_interaction_matrix():
    events = pd.DataFrame({
        "visitorid": [1, 1, 2, 3],
        "itemid": [10, 20, 10, 30],
        "event": ["view", "view", "addtocart", "transaction"],
    })

    result, user2idx, item2idx = create_interaction_matrix(events)

    assert len(user2idx) == 3
    assert len(item2idx) == 3
    assert "user_idx" in result.columns
    assert "item_idx" in result.columns


def test_recommender_dataset_size():
    events = pd.DataFrame({
        "visitorid": [1, 1, 2],
        "itemid": [10, 20, 10],
        "event": ["view", "view", "view"],
    })
    events["user_idx"] = [0, 0, 1]
    events["item_idx"] = [0, 1, 0]

    num_negatives = 2
    dataset = RecommenderDataset(events, num_items=3, num_negatives=num_negatives)

    num_positives = 3
    expected_size = num_positives * (1 + num_negatives)
    assert len(dataset) == expected_size


def test_recommender_dataset_item_returns_correct_types():
    events = pd.DataFrame({
        "visitorid": [1, 2],
        "itemid": [10, 20],
        "event": ["view", "view"],
    })
    events["user_idx"] = [0, 1]
    events["item_idx"] = [0, 1]

    dataset = RecommenderDataset(events, num_items=3, num_negatives=1)

    user, item, label = dataset[0]

    assert isinstance(user, np.int64)
    assert isinstance(item, np.int64)
    assert isinstance(label, np.float32)
    assert label in (0.0, 1.0)