"""Early stopping helper for the training loop.

The :class:`EarlyStopping` utility tracks a monitored metric across
epochs and signals the trainer when training should stop because the
metric has not improved for ``patience`` consecutive epochs.

This is intentionally a separate component (not baked into
:class:`Trainer`) so the trainer stays focused on a single epoch and
``EarlyStopping`` can be reused by other training loops, unit tests
and notebooks.

Example:

    >>> stopper = EarlyStopping(patience=3, mode="max")
    >>> for epoch in range(epochs):
    ...     metric = train_and_evaluate_one_epoch()
    ...     if stopper(metric):
    ...         break
"""
from __future__ import annotations

from typing import Any, Literal


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        patience: Number of epochs with no improvement after which training
            is stopped. ``0`` or a negative value disables early stopping
            (the stopper will never report ``stop=True``).
        mode: ``"min"`` if a lower metric value is better (e.g. loss) or
            ``"max"`` if a higher value is better (e.g. AUC). Defaults to
            ``"min"``.
        min_delta: Minimum change in the monitored metric that qualifies as an
            improvement. Useful to avoid stopping on noisy fluctuations.
        restore_best: When ``True`` the stopper keeps a deep-copy of the best value
            and the caller can retrieve it via :attr:`best_value`. The
            caller is responsible for restoring the model weights (the
            stopper does not touch the model).
    """

    def __init__(
        self,
        patience: int = 3,
        mode: Literal["min", "max"] = "min",
        min_delta: float = 0.0,
        restore_best: bool = True,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(
                f"mode must be 'min' or 'max', got {mode!r}"
            )
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self.restore_best = bool(restore_best)

        self._best_value: Any | None = None
        self._best_epoch: int | None = None
        self._num_bad_epochs: int = 0

    @property
    def best_value(self) -> Any | None:
        """The best metric value observed so far (or ``None``)."""
        return self._best_value

    @property
    def best_epoch(self) -> int | None:
        """The 1-indexed epoch in which the best metric was observed."""
        return self._best_epoch

    @property
    def num_bad_epochs(self) -> int:
        """Number of consecutive epochs without improvement."""
        return self._num_bad_epochs

    @property
    def is_active(self) -> bool:
        """Whether early stopping can actually trigger."""
        return self.patience > 0

    def _is_improvement(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < best - self.min_delta
        return current > best + self.min_delta

    def __call__(self, metric: float, epoch: int | None = None) -> bool:
        """Update the stopper with a new metric value.

        Args:
            metric: Current metric value.
            epoch: Current epoch number (1-indexed). If None, epoch tracking is disabled.

        Returns:
            ``True`` if training should stop, ``False`` otherwise.
        """
        if not self.is_active:
            return False

        if self._best_value is None or self._is_improvement(metric, self._best_value):
            if self.restore_best:
                self._best_value = float(metric)
                self._best_epoch = epoch
            self._num_bad_epochs = 0
        else:
            self._num_bad_epochs += 1
            if self._num_bad_epochs >= self.patience:
                return True
        return False

    def reset(self) -> None:
        """Reset internal state so the stopper can be reused."""
        self._best_value = None
        self._best_epoch = None
        self._num_bad_epochs = 0
