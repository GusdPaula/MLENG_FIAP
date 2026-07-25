"""Training loop for recommender models.

The :class:`Trainer` is a thin, single-purpose component: it knows
how to run one epoch of training and one epoch of evaluation. The
training pipeline composes the trainer with the optimizer, loss
function, and any extra concerns (early stopping, checkpointing,
progress bars, ...).

For convenience, :meth:`Trainer.fit` and
:meth:`Trainer.fit_with_early_stopping` provide a slightly higher
level of orchestration that the pipeline and notebooks can use to
avoid repeating the same boilerplate.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Callable

import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .early_stopping import EarlyStopping

logger = getLogger(__name__)


@dataclass
class EpochResult:
    """Container for the metrics of a single epoch.

    Attributes:
        epoch: 1-indexed epoch number.
        train_loss: Average training loss for the epoch.
        eval_metrics: Dictionary of evaluation metrics (e.g. ``auc_roc``,
            ``avg_precision``) returned by :meth:`Trainer.evaluate`.
        learning_rate: Learning rate used by the optimizer during the epoch.
    """

    epoch: int
    train_loss: float
    eval_metrics: dict[str, float] = field(default_factory=dict)
    learning_rate: float = 0.0


class Trainer:
    """Encapsulates a single-epoch train/evaluate cycle."""

    def __init__(self, model: nn.Module, config: dict[str, Any], device: str = "cpu") -> None:
        """Initialize the trainer.

        Args:
            model: PyTorch model to train.
            config: Training configuration dictionary.
            device: Device to run computations on. Defaults to "cpu".
        """
        self.model = model.to(device)
        self.device = device
        self.config = config

        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config["learning_rate"]
        )

    # ------------------------------------------------------------------
    # batch-level primitives
    # ------------------------------------------------------------------

    def train_batch(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Run a single gradient step on one ``(users, items, labels)`` batch.

        This is the smallest unit of "batch processing" the trainer
        exposes. The caller (a DataLoader, a custom loop, or the
        :meth:`train_epoch` helper below) is responsible for moving
        tensors to ``self.device`` and for any progress reporting.

        Args:
            users: User IDs tensor.
            items: Item IDs tensor.
            labels: Label tensor.

        Returns:
            The loss value for this batch (post-backprop, pre-step).
        """
        self.model.train()
        self.optimizer.zero_grad()

        users = users.to(self.device)
        items = items.to(self.device)
        labels = labels.to(self.device)

        predictions = self.model(users, items)
        loss = self.criterion(predictions, labels)
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def train_epoch(
        self,
        dataloader: DataLoader,
        show_progress: bool = False,
        description: str = "Training",
    ) -> float:
        """Train one full epoch by iterating ``dataloader`` batch-by-batch.

        Args:
            dataloader: DataLoader providing training batches.
            show_progress: Whether to show progress bar. Defaults to False.
            description: Description for progress bar. Defaults to "Training".

        Returns:
            The average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_samples = 0

        batches = (
            tqdm(dataloader, desc=description, leave=False)
            if show_progress
            else dataloader
        )
        for users, items, labels in batches:
            batch_size = users.shape[0]
            batch_loss = self.train_batch(users, items, labels)
            total_loss += batch_loss * batch_size
            num_samples += batch_size
            if show_progress:
                batches.set_postfix(loss=f"{batch_loss:.4f}")

        if num_samples == 0:
            return 0.0
        return total_loss / num_samples

    def evaluate(
        self,
        dataloader: DataLoader,
        metrics: tuple[str, ...] = ("auc_roc", "avg_precision"),
        num_items: int | None = None,
        k: int = 10,
    ) -> dict[str, float]:
        """Evaluate the model on ``dataloader``.

        Args:
            dataloader: DataLoader providing validation batches.
            metrics: Tuple of metric names to compute. Supported metrics:
                - ``"auc_roc"`` - area under the ROC curve
                - ``"avg_precision"`` - average precision (AP)
                - ``"loss"`` - binary cross-entropy over the predictions
                - ``"ndcg_at_k"`` - NDCG@K (requires num_items parameter, uses sampling for efficiency)
            num_items: Total number of items in the catalog (required for ndcg_at_k).
            k: K value for NDCG@K metric. Defaults to 10.

        Returns:
            Dictionary of metric names to computed values.
        """
        self.model.eval()
        collect_ndcg = "ndcg_at_k" in metrics and num_items is not None
        all_preds, all_labels, positive_samples = self._collect_predictions(
            dataloader, collect_ndcg
        )
        return {
            metric: self._compute_metric(
                metric, all_preds, all_labels, positive_samples, num_items, k
            )
            for metric in metrics
        }

    def _collect_predictions(
        self,
        dataloader: DataLoader,
        collect_ndcg: bool,
    ) -> tuple[list[float], list[float], list[tuple[int, int]]]:
        """Run inference and collect predictions, labels, and positive samples."""
        all_preds: list[float] = []
        all_labels: list[float] = []
        positive_samples: list[tuple[int, int]] = []

        with torch.no_grad():
            for users, items, labels in dataloader:
                users, items = users.to(self.device), items.to(self.device)
                predictions = self.model(users, items)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
                if collect_ndcg:
                    self._append_positive_samples(
                        positive_samples, users, items, labels
                    )
        return all_preds, all_labels, positive_samples

    @staticmethod
    def _append_positive_samples(
        positive_samples: list[tuple[int, int]],
        users: torch.Tensor,
        items: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Append positive (user, item) pairs to the accumulator."""
        for user, item, label in zip(
            users.cpu().numpy(), items.cpu().numpy(), labels.numpy(), strict=True
        ):
            if label == 1.0:
                positive_samples.append((int(user), int(item)))

    def _compute_metric(
        self,
        metric: str,
        all_preds: list[float],
        all_labels: list[float],
        positive_samples: list[tuple[int, int]],
        num_items: int | None,
        k: int,
    ) -> float:
        """Dispatch computation for a single evaluation metric."""
        if metric == "auc_roc":
            return float(roc_auc_score(all_labels, all_preds))
        if metric == "avg_precision":
            return float(average_precision_score(all_labels, all_preds))
        if metric == "loss":
            preds_t = torch.tensor(all_preds, dtype=torch.float32)
            labels_t = torch.tensor(all_labels, dtype=torch.float32)
            return float(self.criterion(preds_t, labels_t))
        if metric == "ndcg_at_k":
            if num_items is None:
                raise ValueError("num_items must be provided for ndcg_at_k metric")
            return self._compute_ndcg_at_k_sampled(positive_samples, num_items, k)
        raise ValueError(f"Unknown evaluation metric: {metric!r}")

    # ------------------------------------------------------------------
    # higher-level orchestration
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        show_progress: bool = False,
        metric_for_best: str | None = None,
        mode: str = "min",
        log_callback: Callable[[EpochResult], None] | None = None,
    ) -> list[EpochResult]:
        """Run the train/eval loop for ``epochs`` epochs.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            epochs: Number of epochs to train.
            show_progress: Whether to show progress bars. Defaults to False.
            metric_for_best: If provided, the trainer keeps a deep-copy of the model
                state dict with the best value of that metric.
            mode: ``"min"`` if lower metric is better, ``"max"`` if higher is better.
                Defaults to "min".
            log_callback: Optional callable invoked with the :class:`EpochResult`
                of every epoch, useful for MLflow / progress logging.

        Returns:
            List of EpochResult objects for each epoch.
        """
        if metric_for_best is not None and mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

        results: list[EpochResult] = []
        best_value: float | None = None
        best_state: dict | None = None

        for epoch in range(epochs):
            result = self._run_epoch(
                train_loader, val_loader, epoch, epochs, show_progress
            )
            results.append(result)
            best_value, best_state = self._update_best_model(
                result, metric_for_best, mode, best_value, best_state
            )
            if log_callback is not None:
                log_callback(result)

        if metric_for_best is not None and best_state is not None:
            self.model.load_state_dict(best_state)
        return results

    def _run_epoch(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epoch: int,
        total_epochs: int,
        show_progress: bool,
    ) -> EpochResult:
        """Execute one train + eval cycle and return the result."""
        description = f"Epoch {epoch + 1}/{total_epochs}"
        train_loss = self.train_epoch(
            train_loader, show_progress=show_progress, description=description
        )
        eval_metrics = self.evaluate(val_loader)
        return EpochResult(
            epoch=epoch + 1,
            train_loss=train_loss,
            eval_metrics=eval_metrics,
            learning_rate=self.optimizer.param_groups[0]["lr"],
        )

    def _update_best_model(
        self,
        result: EpochResult,
        metric_for_best: str | None,
        mode: str,
        best_value: float | None,
        best_state: dict | None,
    ) -> tuple[float | None, dict | None]:
        """Conditionally deep-copy model state if a new best metric is found."""
        if metric_for_best is None or metric_for_best not in result.eval_metrics:
            return best_value, best_state
        current = result.eval_metrics[metric_for_best]
        if best_value is None or self._is_better(current, best_value, mode):
            return current, deepcopy(self.model.state_dict())
        return best_value, best_state

    def fit_with_early_stopping(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        early_stopping: EarlyStopping,
        monitor: str = "val_loss",
        show_progress: bool = False,
        log_callback: Callable[[EpochResult], None] | None = None,
        num_items: int | None = None,
        ranking_k: int = 10,
    ) -> tuple[list[EpochResult], dict]:
        """Train with early stopping.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            epochs: Maximum number of epochs to train.
            early_stopping: EarlyStopping instance.
            monitor: Metric to monitor for early stopping. Defaults to "val_loss".
            show_progress: Whether to show progress bars. Defaults to False.
            log_callback: Optional callable invoked with the :class:`EpochResult`
                of every epoch, useful for MLflow / progress logging.
            num_items: Total number of items (required for ranking metrics).
            ranking_k: K value for ranking metrics. Defaults to 10.

        Returns:
            Tuple of (history, best) where:
            - history: List of EpochResult objects for every executed epoch.
            - best: Dictionary containing value, epoch, and state_dict of the best model.
        """
        history: list[EpochResult] = []
        best: dict[str, Any] = {"value": None, "epoch": None, "state_dict": None}

        for epoch in range(epochs):
            result = self._run_early_stopping_epoch(
                train_loader, val_loader, epoch, epochs,
                show_progress, monitor, num_items, ranking_k,
            )
            history.append(result)
            if log_callback is not None:
                log_callback(result)

            monitored_value = self._resolve_monitor(monitor, result)
            best = self._check_and_update_best(best, monitored_value, result.epoch, early_stopping.mode)

            if early_stopping(monitored_value, epoch=result.epoch):
                self._log_early_stopping(result.epoch, monitor, best, epochs)
                break

        if best["state_dict"] is not None:
            self.model.load_state_dict(best["state_dict"])
        return history, best

    def _run_early_stopping_epoch(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epoch: int,
        total_epochs: int,
        show_progress: bool,
        monitor: str,
        num_items: int | None,
        ranking_k: int,
    ) -> EpochResult:
        """Execute one train + eval cycle for early-stopping training."""
        description = f"Epoch {epoch + 1}/{total_epochs}"
        train_loss = self.train_epoch(
            train_loader, show_progress=show_progress, description=description
        )
        eval_metrics_tuple, eval_kwargs = self._build_eval_kwargs(
            monitor, num_items, ranking_k
        )
        eval_metrics = self.evaluate(val_loader, metrics=eval_metrics_tuple, **eval_kwargs)
        result = EpochResult(
            epoch=epoch + 1,
            train_loss=train_loss,
            eval_metrics=eval_metrics,
            learning_rate=self.optimizer.param_groups[0]["lr"],
        )
        logger.info(
            f"Epoch {result.epoch:02d}/{total_epochs} | "
            f"loss={result.train_loss:.4f} | "
            f"auc={result.eval_metrics['auc_roc']:.4f} | "
            f"ap={result.eval_metrics['avg_precision']:.4f}"
        )
        return result

    @staticmethod
    def _build_eval_kwargs(
        monitor: str, num_items: int | None, ranking_k: int
    ) -> tuple[tuple[str, ...], dict]:
        """Build evaluation metrics tuple and kwargs based on the monitored metric."""
        base_metrics = ("auc_roc", "avg_precision")
        if num_items is None:
            return base_metrics, {}
        if monitor == "ndcg_at_k":
            return (*base_metrics, "ndcg_at_k"), {"num_items": num_items, "k": ranking_k}
        if monitor.startswith("ndcg_at"):
            return (*base_metrics, monitor), {"num_items": num_items, "k": ranking_k}
        return base_metrics, {}

    def _check_and_update_best(
        self,
        best: dict[str, Any],
        monitored_value: float,
        epoch: int,
        mode: str,
    ) -> dict[str, Any]:
        """Update best model state if monitored value improves."""
        if best["value"] is None or self._is_better(monitored_value, best["value"], mode):
            return {
                "value": monitored_value,
                "epoch": epoch,
                "state_dict": deepcopy(self.model.state_dict()),
            }
        return best

    @staticmethod
    def _log_early_stopping(
        epoch: int, monitor: str, best: dict[str, Any], total_epochs: int
    ) -> None:
        """Log early stopping trigger information."""
        logger.info(
            f"Early stopping triggered at epoch {epoch}. "
            f"Best {monitor}: {best['value']:.4f} "
            f"(epoch {best['epoch']})"
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _compute_ndcg_at_k_sampled(
        self,
        positive_samples: list[tuple[int, int]],
        num_items: int,
        k: int = 10,
        sample_limit: int = 100,
    ) -> float:
        """Compute NDCG@K using sampled positive samples for efficiency.

        Args:
            positive_samples: List of (user_id, item_id) tuples for positive interactions.
            num_items: Total number of items in the catalog.
            k: Rank position for NDCG computation.
            sample_limit: Maximum number of positive samples to evaluate.

        Returns:
            NDCG@K score.
        """
        if not positive_samples:
            return 0.0

        import numpy as np

        sampled = self._sample_positive_interactions(positive_samples, sample_limit)
        users_items = self._group_by_user(sampled)

        ndcg_scores = []
        with torch.no_grad():
            for user_idx, true_items in users_items.items():
                ndcg = self._compute_user_ndcg(user_idx, true_items, num_items, k)
                ndcg_scores.append(ndcg)

        return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

    @staticmethod
    def _sample_positive_interactions(
        positive_samples: list[tuple[int, int]], sample_limit: int
    ) -> list[tuple[int, int]]:
        """Randomly sample a subset of positive interactions."""
        import numpy as np

        indices = np.random.choice(
            len(positive_samples),
            min(sample_limit, len(positive_samples)),
            replace=False,
        )
        return [positive_samples[i] for i in indices]

    @staticmethod
    def _group_by_user(
        interactions: list[tuple[int, int]],
    ) -> dict[int, list[int]]:
        """Group (user, item) pairs by user."""
        users_items: dict[int, list[int]] = {}
        for user, item in interactions:
            users_items.setdefault(user, []).append(item)
        return users_items

    def _compute_user_ndcg(
        self, user_idx: int, true_items: list[int], num_items: int, k: int
    ) -> float:
        """Compute NDCG@K for a single user."""
        import numpy as np

        user_tensor = torch.full((num_items,), user_idx, dtype=torch.long).to(self.device)
        item_tensor = torch.arange(num_items, dtype=torch.long).to(self.device)
        scores = self.model(user_tensor, item_tensor)
        _, top_k_indices = torch.topk(scores, k)
        top_k_list = top_k_indices.cpu().numpy()

        dcg = sum(
            1.0 / np.log2(rank + 2)
            for rank, item_id in enumerate(top_k_list)
            if item_id in true_items
        )
        ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_items), k)))
        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    @staticmethod
    def _is_better(current: float, best: float, mode: str) -> bool:
        return current < best if mode == "min" else current > best

    @staticmethod
    def _resolve_monitor(monitor: str, result: EpochResult) -> float:
        if monitor == "val_loss":
            return result.train_loss
        if monitor in result.eval_metrics:
            return result.eval_metrics[monitor]
        raise ValueError(
            f"Monitored value '{monitor}' not found. Available: "
            f"'val_loss' or {sorted(result.eval_metrics)}"
        )
