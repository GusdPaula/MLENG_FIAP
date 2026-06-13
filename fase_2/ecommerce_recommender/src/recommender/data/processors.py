"""Data processing strategies (Strategy pattern).

The training pipeline needs to support multiple, swappable ways of
turning raw event logs into (user, item) interactions with weights
and/or filters. Concretely each strategy implements:

* :meth:`DataProcessor.process` - turn a raw events DataFrame into
  cleaned interactions with ``user_idx``/``item_idx`` columns plus
  mappings ``user2idx``/``item2idx``.

A :class:`DataProcessorContext` selects a concrete strategy at runtime
(the strategy is normally read from config), so the rest of the
pipeline is agnostic to *how* the data was prepared.

Available built-in strategies:

* :class:`WeightedEventProcessor` - weights by event type (default).
* :class:`BinaryInteractionProcessor` - treats any non-view event as
  a positive interaction and drops views (transactional signal only).
* :class:`ImplicitFeedbackProcessor` - keeps all events as positives
  with weight 1 (purely implicit, no type hierarchy).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class DataProcessor(ABC):
    """Abstract strategy interface."""

    name: str = "abstract"

    @abstractmethod
    def process(self, events: pd.DataFrame, **kwargs: Any) -> tuple[pd.DataFrame, dict[int, int], dict[int, int]]:
        """Process events into interactions with user/item mappings.

        Args:
            events: Raw events DataFrame.
            **kwargs: Additional processor-specific parameters.

        Returns:
            Tuple of (interactions DataFrame, user2idx mapping, item2idx mapping).
        """

    # --- shared helpers used by concrete strategies -----------------

    @staticmethod
    def _build_mappings(events: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
        """Build user and item ID to index mappings.

        Args:
            events: DataFrame with visitorid and itemid columns.

        Returns:
            Tuple of (user2idx mapping, item2idx mapping).
        """
        user_ids = events["visitorid"].unique()
        item_ids = events["itemid"].unique()
        user2idx = {int(uid): idx for idx, uid in enumerate(user_ids)}
        item2idx = {int(iid): idx for idx, iid in enumerate(item_ids)}
        return user2idx, item2idx

    @staticmethod
    def _filter_by_min_interactions(
        df: pd.DataFrame, min_interactions: int
    ) -> pd.DataFrame:
        """Filter DataFrame to keep only users/items with minimum interactions.

        Args:
            df: DataFrame with visitorid and itemid columns.
            min_interactions: Minimum number of interactions per user/item.

        Returns:
            Filtered DataFrame.
        """
        if min_interactions > 1:
            user_counts = df["visitorid"].value_counts()
            item_counts = df["itemid"].value_counts()
            df = df[
                df["visitorid"].isin(user_counts[user_counts >= min_interactions].index)
                & df["itemid"].isin(item_counts[item_counts >= min_interactions].index)
            ]
        return df

    @staticmethod
    def _apply_index_columns(
        events: pd.DataFrame,
        user2idx: dict[int, int],
        item2idx: dict[int, int],
    ) -> pd.DataFrame:
        """Apply index columns to events DataFrame.

        Args:
            events: DataFrame with visitorid and itemid columns.
            user2idx: User ID to index mapping.
            item2idx: Item ID to index mapping.

        Returns:
            DataFrame with user_idx and item_idx columns.
        """
        out = events.copy()
        out["user_idx"] = out["visitorid"].map(user2idx)
        out["item_idx"] = out["itemid"].map(item2idx)
        out = out.dropna(subset=["user_idx", "item_idx"])
        out["user_idx"] = out["user_idx"].astype(np.int64)
        out["item_idx"] = out["item_idx"].astype(np.int64)
        return out


class WeightedEventProcessor(DataProcessor):
    """Assigns weights based on event type (view < addtocart < transaction)."""

    name = "weighted"

    DEFAULT_WEIGHTS: dict[str, float] = {
        "view": 1.0,
        "addtocart": 2.0,
        "transaction": 3.0,
    }

    def __init__(self, weights: dict[str, float] | None = None):
        """Initialize the weighted event processor.

        Args:
            weights: Custom weights for event types. If None, uses DEFAULT_WEIGHTS.
        """
        self.weights = weights or self.DEFAULT_WEIGHTS

    def process(
        self, events: pd.DataFrame, min_interactions: int = 1, **_: Any
    ) -> tuple[pd.DataFrame, dict[int, int], dict[int, int]]:
        """Process events with weighted event types.

        Args:
            events: Raw events DataFrame.
            min_interactions: Minimum interactions per user/item. Defaults to 1.
            **_: Additional ignored parameters for interface compatibility.

        Returns:
            Tuple of (interactions DataFrame with weights, user2idx mapping, item2idx mapping).
        """
        df = events.copy()
        df["weight"] = df["event"].map(self.weights).fillna(0.0)
        df = self._filter_by_min_interactions(df, min_interactions)

        user2idx, item2idx = self._build_mappings(df)
        return self._apply_index_columns(df, user2idx, item2idx), user2idx, item2idx


class BinaryInteractionProcessor(DataProcessor):
    """Keeps only add-to-cart and transaction events as positives.

    Useful when the recommender should be optimized on explicit user
    intent (purchases) rather than passive views.
    """

    name = "binary"

    POSITIVE_EVENTS: tuple[str, ...] = ("addtocart", "transaction")

    def __init__(self, positive_events: tuple[str, ...] | None = None):
        """Initialize the binary interaction processor.

        Args:
            positive_events: Tuple of event types to treat as positive. If None, uses POSITIVE_EVENTS.
        """
        self.positive_events = positive_events or self.POSITIVE_EVENTS

    def process(
        self, events: pd.DataFrame, min_interactions: int = 1, **_: Any
    ) -> tuple[pd.DataFrame, dict[int, int], dict[int, int]]:
        """Process events keeping only positive events.

        Args:
            events: Raw events DataFrame.
            min_interactions: Minimum interactions per user/item. Defaults to 1.
            **_: Additional ignored parameters for interface compatibility.

        Returns:
            Tuple of (interactions DataFrame with weight=1.0, user2idx mapping, item2idx mapping).
        """
        df = events[events["event"].isin(self.positive_events)].copy()
        df["weight"] = 1.0
        df = self._filter_by_min_interactions(df, min_interactions)

        user2idx, item2idx = self._build_mappings(df)
        return self._apply_index_columns(df, user2idx, item2idx), user2idx, item2idx


class ImplicitFeedbackProcessor(DataProcessor):
    """Treats every event as a positive (implicit feedback only)."""

    name = "implicit"

    def process(
        self, events: pd.DataFrame, min_interactions: int = 1, **_: Any
    ) -> tuple[pd.DataFrame, dict[int, int], dict[int, int]]:
        """Process events treating all events as positive (implicit feedback).

        Args:
            events: Raw events DataFrame.
            min_interactions: Minimum interactions per user/item. Defaults to 1.
            **_: Additional ignored parameters for interface compatibility.

        Returns:
            Tuple of (interactions DataFrame with weight=1.0, user2idx mapping, item2idx mapping).
        """
        df = events.copy()
        df["weight"] = 1.0
        df = self._filter_by_min_interactions(df, min_interactions)

        user2idx, item2idx = self._build_mappings(df)
        return self._apply_index_columns(df, user2idx, item2idx), user2idx, item2idx


class DataProcessorContext:
    """Holds a :class:`DataProcessor` strategy and exposes ``process``.

    The context lets the rest of the pipeline call a single
    ``context.process(events)`` regardless of the underlying strategy.
    """

    _STRATEGIES: dict[str, type[DataProcessor]] = {
        cls.name: cls
        for cls in (WeightedEventProcessor, BinaryInteractionProcessor, ImplicitFeedbackProcessor)
    }

    def __init__(self, strategy: str | DataProcessor = "weighted", **strategy_kwargs: Any):
        """Initialize the data processor context with a strategy.

        Args:
            strategy: Strategy name or DataProcessor instance. Defaults to "weighted".
            **strategy_kwargs: Additional arguments for strategy initialization.

        Raises:
            ValueError: If strategy name is not recognized.
        """
        if isinstance(strategy, DataProcessor):
            self._strategy = strategy
        elif strategy in self._STRATEGIES:
            self._strategy = self._STRATEGIES[strategy](**strategy_kwargs)
        else:
            available = ", ".join(sorted(self._STRATEGIES)) or "<empty>"
            raise ValueError(
                f"Unknown data processor strategy '{strategy}'. "
                f"Available strategies: {available}."
            )

    @property
    def strategy_name(self) -> str:
        """Return the name of the current strategy."""
        return self._strategy.name

    def process(self, events: pd.DataFrame, **kwargs: Any) -> tuple[pd.DataFrame, dict[int, int], dict[int, int]]:
        """Process events using the configured strategy.

        Args:
            events: Raw events DataFrame.
            **kwargs: Additional arguments for the strategy process method.

        Returns:
            Tuple of (interactions DataFrame, user2idx mapping, item2idx mapping).
        """
        return self._strategy.process(events, **kwargs)

    @classmethod
    def available_strategies(cls) -> list[str]:
        """Return list of available strategy names."""
        return sorted(cls._STRATEGIES)
