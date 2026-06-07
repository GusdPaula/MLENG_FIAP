import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score


class Trainer:
    def __init__(self, model: nn.Module, config: dict, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.config = config

        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config["learning_rate"]
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for users, items, labels in dataloader:
            users = users.to(self.device)
            items = items.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(users, items)
            loss = self.criterion(predictions, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(users)

        return total_loss / len(dataloader.dataset)

    def evaluate(self, dataloader: DataLoader) -> dict:
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for users, items, labels in dataloader:
                users = users.to(self.device)
                items = items.to(self.device)

                predictions = self.model(users, items)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())

        auc = roc_auc_score(all_labels, all_preds)
        ap = average_precision_score(all_labels, all_preds)

        return {"auc_roc": auc, "avg_precision": ap}
