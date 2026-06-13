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
from typing import Any, Callable

import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .early_stopping import EarlyStopping


@dataclass
class EpochResult:
    """Container for the metrics of a single epoch.

    Attributes
    ----------
    epoch:
        1-indexed epoch number.
    train_loss:
        Average training loss for the epoch.
    eval_metrics:
        Dictionary of evaluation metrics (e.g. ``auc_roc``,
        ``avg_precision``) returned by :meth:`Trainer.evaluate`.
    learning_rate:
        Learning rate used by the optimizer during the epoch.
    """

    epoch: int
    train_loss: float
    eval_metrics: dict[str, float] = field(default_factory=dict)
    learning_rate: float = 0.0


class Trainer:
    """Encapsulates a single-epoch train/evaluate cycle."""

    def __init__(self, model: nn.Module, config: dict, device: str = "cpu"):
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

        Returns
        -------
        float
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

        Returns
        -------
        float
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
    ) -> dict:
        """Evaluate the model on ``dataloader``.

        Supported metrics:

        * ``"auc_roc"`` - area under the ROC curve
        * ``"avg_precision"`` - average precision (AP)
        * ``"loss"`` - binary cross-entropy over the predictions
        """
        self.model.eval()
        all_preds: list[float] = []
        all_labels: list[float] = []

        with torch.no_grad():
            for users, items, labels in dataloader:
                users = users.to(self.device)
                items = items.to(self.device)

                predictions = self.model(users, items)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())

        result: dict[str, float] = {}
        for metric in metrics:
            if metric == "auc_roc":
                result[metric] = float(roc_auc_score(all_labels, all_preds))
            elif metric == "avg_precision":
                result[metric] = float(
                    average_precision_score(all_labels, all_preds)
                )
            elif metric == "loss":
                preds_tensor = torch.tensor(all_preds, dtype=torch.float32)
                labels_tensor = torch.tensor(all_labels, dtype=torch.float32)
                result[metric] = float(self.criterion(preds_tensor, labels_tensor))
            else:
                raise ValueError(f"Unknown evaluation metric: {metric!r}")
        return result

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

        Parameters
        ----------
        metric_for_best, mode:
            If ``metric_for_best`` is provided, the trainer keeps a
            deep-copy of the model state dict with the best value of
            that metric (according to ``mode`` - ``"min"`` or
            ``"max"``) and reloads it before returning.
        log_callback:
            Optional callable invoked with the :class:`EpochResult`
            of every epoch, useful for MLflow / progress logging.
        """
        results: list[EpochResult] = []
        best_value: float | None = None
        best_state: dict | None = None

        if metric_for_best is not None and mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

        for epoch in range(epochs):
            description = f"Epoch {epoch + 1}/{epochs}"
            train_loss = self.train_epoch(
                train_loader,
                show_progress=show_progress,
                description=description,
            )
            eval_metrics = self.evaluate(val_loader)
            lr = self.optimizer.param_groups[0]["lr"]
            result = EpochResult(
                epoch=epoch + 1,
                train_loss=train_loss,
                eval_metrics=eval_metrics,
                learning_rate=lr,
            )
            results.append(result)

            if metric_for_best is not None and metric_for_best in eval_metrics:
                current = eval_metrics[metric_for_best]
                if best_value is None or self._is_better(current, best_value, mode):
                    best_value = current
                    best_state = deepcopy(self.model.state_dict())

            if log_callback is not None:
                log_callback(result)

        if metric_for_best is not None and best_state is not None:
            self.model.load_state_dict(best_state)
        return results

    def fit_with_early_stopping(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        early_stopping: EarlyStopping,
        monitor: str = "val_loss",
        show_progress: bool = False,
        log_callback: Callable[[EpochResult], None] | None = None,
    ) -> tuple[list[EpochResult], dict]:
        """Train with early stopping.

        Returns
        -------
        tuple
            (history, best)

            history:
                List of EpochResult objects for every executed epoch.

            best:
                Dictionary containing:
                    value       -> best monitored metric
                    epoch       -> epoch where the best metric occurred
                    state_dict  -> best model weights
        """

        history: list[EpochResult] = []

        best: dict[str, Any] = {
            "value": None,
            "epoch": None,
            "state_dict": None,
        }

        for epoch in range(epochs):

            description = f"Epoch {epoch + 1}/{epochs}"

            train_loss = self.train_epoch(
                train_loader,
                show_progress=show_progress,
                description=description,
            )

            eval_metrics = self.evaluate(val_loader)

            result = EpochResult(
                epoch=epoch + 1,
                train_loss=train_loss,
                eval_metrics=eval_metrics,
                learning_rate=self.optimizer.param_groups[0]["lr"],
            )

            print(
                f"Epoch {result.epoch:02d}/{epochs} | "
                f"loss={result.train_loss:.4f} | "
                f"auc={result.eval_metrics['auc_roc']:.4f} | "
                f"ap={result.eval_metrics['avg_precision']:.4f}"
            )

            history.append(result)

            # Optional logging (MLflow, console, etc.)
            if log_callback is not None:
                log_callback(result)

            # Metric used for early stopping
            monitored_value = self._resolve_monitor(monitor, result)

            # Save best model BEFORE checking whether to stop
            if (
                best["value"] is None
                or self._is_better(
                    monitored_value,
                    best["value"],
                    early_stopping.mode,
                )
            ):
                best["value"] = monitored_value
                best["epoch"] = result.epoch
                best["state_dict"] = deepcopy(self.model.state_dict())

            # Ask EarlyStopping whether training should stop
            if early_stopping(
                monitored_value,
                epoch=result.epoch,
            ):
                print(
                    f"Early stopping triggered at epoch {result.epoch}. "
                    f"Best {monitor}: {best['value']:.4f} "
                    f"(epoch {best['epoch']})"
                )
                break

        # Restore best model
        if best["state_dict"] is not None:
            self.model.load_state_dict(best["state_dict"])

        return history, best

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

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
