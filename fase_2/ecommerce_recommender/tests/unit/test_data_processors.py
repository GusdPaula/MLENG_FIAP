"""Tests for the data-processor Strategy pattern."""
import numpy as np
import pandas as pd
import pytest
from src.recommender.data.processors import (
    BinaryInteractionProcessor,
    DataProcessorContext,
    ImplicitFeedbackProcessor,
    WeightedEventProcessor,
)


def _sample_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "visitorid": [1, 1, 1, 2, 2, 3, 3, 3, 3, 4],
            "itemid": [10, 20, 30, 10, 40, 20, 30, 50, 60, 10],
            "event": [
                "view",
                "view",
                "transaction",
                "addtocart",
                "transaction",
                "view",
                "view",
                "addtocart",
                "transaction",
                "view",
            ],
        }
    )


def test_weighted_processor_assigns_weights():
    df = _sample_events()
    interactions, user2idx, item2idx = WeightedEventProcessor().process(df)

    assert "weight" in interactions.columns
    assert interactions["weight"].min() >= 1.0
    assert user2idx and item2idx
    # All rows should be kept (no filtering by default).
    assert len(interactions) == len(df)


def test_binary_processor_keeps_only_positive_events():
    df = _sample_events()
    interactions, _, _ = BinaryInteractionProcessor().process(df)

    assert len(interactions) < len(df)
    # All retained events must be addtocart or transaction.
    original_event_count = (
        df["event"].isin(["addtocart", "transaction"]).sum()
    )
    assert len(interactions) == original_event_count
    assert (interactions["weight"] == 1.0).all()


def test_implicit_processor_assigns_unit_weight():
    df = _sample_events()
    interactions, _, _ = ImplicitFeedbackProcessor().process(df)

    assert len(interactions) == len(df)
    assert (interactions["weight"] == 1.0).all()


def test_min_interactions_filters_cold_users():
    df = _sample_events()
    interactions, _, _ = WeightedEventProcessor().process(df, min_interactions=3)
    assert len(interactions) == 1
    assert interactions["visitorid"].nunique() == 1
    assert interactions["itemid"].nunique() == 1


def test_context_picks_strategy_by_name():
    df = _sample_events()

    weighted = DataProcessorContext("weighted")
    binary = DataProcessorContext("binary")
    implicit = DataProcessorContext("implicit")

    w_int, _, _ = weighted.process(df)
    b_int, _, _ = binary.process(df)
    i_int, _, _ = implicit.process(df)

    assert len(w_int) == len(df)
    assert len(b_int) < len(df)
    assert len(i_int) == len(df)


def test_context_accepts_instance():
    df = _sample_events()
    ctx = DataProcessorContext(WeightedEventProcessor(weights={"view": 5.0}))
    interactions, _, _ = ctx.process(df)
    assert set(interactions["weight"].unique()) <= {0.0, 5.0}
    assert (interactions.loc[interactions["event"] == "view", "weight"] == 5.0).all()


def test_context_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown data processor strategy"):
        DataProcessorContext("not_a_real_strategy")


def test_context_strategy_name_reflects_choice():
    assert DataProcessorContext("binary").strategy_name == "binary"
    assert DataProcessorContext("weighted").strategy_name == "weighted"
    assert DataProcessorContext("implicit").strategy_name == "implicit"


def test_index_columns_are_dense_int64():
    df = _sample_events()
    interactions, _, _ = WeightedEventProcessor().process(df)
    assert interactions["user_idx"].dtype == np.int64
    assert interactions["item_idx"].dtype == np.int64
