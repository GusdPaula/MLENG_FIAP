import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(self, model: nn.Module, config: dict, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.config = config

        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config["learning_rate"]
        )

    def train_epoch(
        self,
        dataloader: DataLoader,
        show_progress: bool = False,
        description: str = "Training",
    ) -> float:
        self.model.train()
        total_loss = 0.0

        batches = tqdm(dataloader, desc=description, leave=False) if show_progress else dataloader
        for users, items, labels in batches:
            users = users.to(self.device)
            items = items.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(users, items)
            loss = self.criterion(predictions, labels)
            loss.backward()
            self.optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss * len(users)
            if show_progress:
                batches.set_postfix(loss=f"{batch_loss:.4f}")

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
