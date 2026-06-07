import pandas as pd
import numpy as np
from torch.utils.data import Dataset


def load_events(path: str) -> pd.DataFrame:
    """Carrega e filtra eventos relevantes."""
    df = pd.read_csv(path)
    event_weights = {"view": 1, "addtocart": 2, "transaction": 3}
    df["weight"] = df["event"].map(event_weights)
    return df


def create_interaction_matrix(events: pd.DataFrame):
    """
    Cria mapeamentos user/item -> índices inteiros e
    gera pares positivos (user_idx, item_idx).
    """
    user_ids = events["visitorid"].unique()
    item_ids = events["itemid"].unique()

    user2idx = {uid: idx for idx, uid in enumerate(user_ids)}
    item2idx = {iid: idx for idx, iid in enumerate(item_ids)}

    events["user_idx"] = events["visitorid"].map(user2idx)
    events["item_idx"] = events["itemid"].map(item2idx)

    return events, user2idx, item2idx


class RecommenderDataset(Dataset):
    """Dataset para treino com negative sampling."""

    def __init__(self, interactions: pd.DataFrame, num_items: int, num_negatives: int = 4):
        self.interactions = interactions[["user_idx", "item_idx"]].values
        self.num_items = num_items
        self.num_negatives = num_negatives

        self.positive_set = set(map(tuple, self.interactions))
        self.samples = self._generate_samples()

    def _generate_samples(self):
        samples = []
        for user_idx, item_idx in self.interactions:
            samples.append((user_idx, item_idx, 1.0))

            for _ in range(self.num_negatives):
                neg_item = np.random.randint(0, self.num_items)
                while (user_idx, neg_item) in self.positive_set:
                    neg_item = np.random.randint(0, self.num_items)
                samples.append((user_idx, neg_item, 0.0))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user, item, label = self.samples[idx]
        return (
            np.int64(user),
            np.int64(item),
            np.float32(label),
        )