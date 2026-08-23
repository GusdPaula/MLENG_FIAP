"""
Unit tests for data/validate.py — Validador component.

Covers:
  - Happy path: valid DataFrames return True
  - Each failure type in isolation (columns, labels, abstracts, size)
  - Validation order: column checks must precede all subsequent checks
"""

import pytest

from data.validate import validate_data


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_valid_data_returns_true(df_train_valid, df_test_valid):
    """Both valid DataFrames → validate_data() should return exactly True."""
    result = validate_data(df_train_valid, df_test_valid)
    assert result is True


# ---------------------------------------------------------------------------
# Missing columns
# ---------------------------------------------------------------------------

def test_missing_label_column_raises(df_missing_label_column, df_test_valid):
    """Training DataFrame without 'condition_label' → ValueError."""
    with pytest.raises(ValueError, match="condition_label"):
        validate_data(df_missing_label_column, df_test_valid)


def test_missing_abstract_column_raises(df_missing_abstract_column, df_test_valid):
    """Training DataFrame without 'medical_abstract' → ValueError."""
    with pytest.raises(ValueError, match="medical_abstract"):
        validate_data(df_missing_abstract_column, df_test_valid)


# ---------------------------------------------------------------------------
# Invalid labels
# ---------------------------------------------------------------------------

def test_invalid_labels_train_raises(df_invalid_labels_train, df_test_valid):
    """Training DataFrame with labels {0, 6, -1} → ValueError mentioning the values."""
    with pytest.raises(ValueError, match="condition_label"):
        validate_data(df_invalid_labels_train, df_test_valid)


def test_invalid_labels_test_raises(df_train_valid, df_invalid_labels_test):
    """Test DataFrame with label 99 → ValueError mentioning the values."""
    with pytest.raises(ValueError, match="condition_label"):
        validate_data(df_train_valid, df_invalid_labels_test)


# ---------------------------------------------------------------------------
# Invalid abstracts
# ---------------------------------------------------------------------------

def test_null_abstract_raises(df_null_abstract, df_test_valid):
    """Training DataFrame with None in medical_abstract → ValueError."""
    with pytest.raises(ValueError, match="medical_abstract"):
        validate_data(df_null_abstract, df_test_valid)


def test_empty_abstract_raises(df_empty_abstract, df_test_valid):
    """Training DataFrame with empty string in medical_abstract → ValueError."""
    with pytest.raises(ValueError, match="medical_abstract"):
        validate_data(df_empty_abstract, df_test_valid)


def test_whitespace_abstract_raises(df_whitespace_abstract, df_test_valid):
    """Training DataFrame with whitespace-only abstract → ValueError."""
    with pytest.raises(ValueError, match="medical_abstract"):
        validate_data(df_whitespace_abstract, df_test_valid)


# ---------------------------------------------------------------------------
# Insufficient size
# ---------------------------------------------------------------------------

def test_train_too_small_raises(df_train_too_small, df_test_valid):
    """Training DataFrame with 50 rows (< 1000 minimum) → ValueError."""
    with pytest.raises(ValueError, match="treino"):
        validate_data(df_train_too_small, df_test_valid)


def test_test_too_small_raises(df_train_valid, df_test_too_small):
    """Test DataFrame with 10 rows (< 100 minimum) → ValueError."""
    with pytest.raises(ValueError, match="teste"):
        validate_data(df_train_valid, df_test_too_small)


# ---------------------------------------------------------------------------
# Validation order — columns are checked before labels
# ---------------------------------------------------------------------------

def test_validation_order_columns_first(df_missing_label_column, df_test_valid):
    """DataFrame missing 'condition_label' AND would have bad labels elsewhere:
    validate_data() must raise for the missing column before attempting label checks.

    We verify this by asserting the error message is about the missing column,
    not about invalid label values.
    """
    with pytest.raises(ValueError) as exc_info:
        validate_data(df_missing_label_column, df_test_valid)

    # The error should describe the missing column, not label values
    message = str(exc_info.value)
    assert "condition_label" in message
    assert "ausente" in message  # message uses "ausente" for missing column


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st


# ── Helpers ──────────────────────────────────────────────────────────────────

# Strategy for a single valid non-blank abstract text
_valid_abstract_st = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    min_size=1,
).filter(lambda s: s.strip() != "")

# Strategy for a single valid label in {1, 2, 3, 4, 5}
_valid_label_st = st.integers(min_value=1, max_value=5)


def _make_df_from_lists(labels, abstracts) -> pd.DataFrame:
    """Build a minimal DataFrame from label and abstract lists."""
    return pd.DataFrame({
        "condition_label": pd.array(labels, dtype="int64"),
        "medical_abstract": abstracts,
    })


@st.composite
def _valid_train_df(draw) -> pd.DataFrame:
    """Generate a valid training DataFrame (≥ 1 000 rows)."""
    n = draw(st.integers(min_value=1_000, max_value=1_200))
    labels = draw(st.lists(_valid_label_st, min_size=n, max_size=n))
    abstracts = draw(st.lists(_valid_abstract_st, min_size=n, max_size=n))
    return _make_df_from_lists(labels, abstracts)


@st.composite
def _valid_test_df(draw) -> pd.DataFrame:
    """Generate a valid test DataFrame (≥ 100 rows)."""
    n = draw(st.integers(min_value=100, max_value=150))
    labels = draw(st.lists(_valid_label_st, min_size=n, max_size=n))
    abstracts = draw(st.lists(_valid_abstract_st, min_size=n, max_size=n))
    return _make_df_from_lists(labels, abstracts)


# ── Property 2: Validação rejeita DataFrames com colunas ausentes ─────────────
# Validates: Requirements 2.2, 2.3

@given(df=_valid_train_df(), test_df=_valid_test_df(), drop_col=st.sampled_from(["condition_label", "medical_abstract"]))
@settings(max_examples=30)
def test_prop2_missing_column_raises(df, test_df, drop_col):
    """Property 2: validate_data() must raise ValueError for any DataFrame
    that is missing either required column.

    **Validates: Requirements 2.2, 2.3**
    """
    df_no_col = df.drop(columns=[drop_col])
    with pytest.raises(ValueError):
        validate_data(df_no_col, test_df)


# ── Property 3: Validação rejeita rótulos fora do intervalo [1, 5] ────────────
# Validates: Requirements 2.4, 2.5

_invalid_label_st = st.one_of(
    st.integers(max_value=0),
    st.integers(min_value=6),
)


@given(
    df=_valid_train_df(),
    test_df=_valid_test_df(),
    invalid_label=_invalid_label_st,
    inject_into_train=st.booleans(),
)
@settings(max_examples=40)
def test_prop3_invalid_labels_raises(df, test_df, invalid_label, inject_into_train):
    """Property 3: validate_data() must raise ValueError whenever condition_label
    contains at least one value outside {1, 2, 3, 4, 5}.

    **Validates: Requirements 2.4, 2.5**
    """
    if inject_into_train:
        target_df = df.copy()
        target_df.loc[0, "condition_label"] = invalid_label
        with pytest.raises(ValueError, match="condition_label"):
            validate_data(target_df, test_df)
    else:
        target_df = test_df.copy()
        target_df.loc[0, "condition_label"] = invalid_label
        with pytest.raises(ValueError, match="condition_label"):
            validate_data(df, target_df)


# ── Property 4: Validação rejeita abstracts nulos, vazios ou apenas espaços ───
# Validates: Requirements 2.6, 2.7

_bad_abstract_st = st.one_of(
    st.just(None),
    st.just(""),
    st.text(alphabet=" \t\n\r", min_size=1),  # whitespace-only
)


@given(
    df=_valid_train_df(),
    test_df=_valid_test_df(),
    bad_abstract=_bad_abstract_st,
    inject_into_train=st.booleans(),
)
@settings(max_examples=40)
def test_prop4_bad_abstract_raises(df, test_df, bad_abstract, inject_into_train):
    """Property 4: validate_data() must raise ValueError for any DataFrame
    that contains a null, empty, or whitespace-only medical_abstract entry.

    **Validates: Requirements 2.6, 2.7**
    """
    if inject_into_train:
        target_df = df.copy()
        target_df.loc[0, "medical_abstract"] = bad_abstract
        with pytest.raises(ValueError, match="medical_abstract"):
            validate_data(target_df, test_df)
    else:
        target_df = test_df.copy()
        target_df.loc[0, "medical_abstract"] = bad_abstract
        with pytest.raises(ValueError, match="medical_abstract"):
            validate_data(df, target_df)


# ── Property 5: Validação rejeita DataFrames abaixo do tamanho mínimo ─────────
# Validates: Requirements 2.8, 2.9

@st.composite
def _undersized_train_df(draw) -> pd.DataFrame:
    """Generate a valid-content training DataFrame with fewer than 1 000 rows."""
    n = draw(st.integers(min_value=1, max_value=999))
    labels = draw(st.lists(_valid_label_st, min_size=n, max_size=n))
    abstracts = draw(st.lists(_valid_abstract_st, min_size=n, max_size=n))
    return _make_df_from_lists(labels, abstracts)


@st.composite
def _undersized_test_df(draw) -> pd.DataFrame:
    """Generate a valid-content test DataFrame with fewer than 100 rows."""
    n = draw(st.integers(min_value=1, max_value=99))
    labels = draw(st.lists(_valid_label_st, min_size=n, max_size=n))
    abstracts = draw(st.lists(_valid_abstract_st, min_size=n, max_size=n))
    return _make_df_from_lists(labels, abstracts)


@given(small_train=_undersized_train_df(), test_df=_valid_test_df())
@settings(max_examples=20)
def test_prop5a_undersized_train_raises(small_train, test_df):
    """Property 5a: validate_data() must raise ValueError when the training
    DataFrame has fewer than 1 000 rows.

    **Validates: Requirements 2.8, 2.9**
    """
    with pytest.raises(ValueError, match="treino"):
        validate_data(small_train, test_df)


@given(train_df=_valid_train_df(), small_test=_undersized_test_df())
@settings(max_examples=20)
def test_prop5b_undersized_test_raises(train_df, small_test):
    """Property 5b: validate_data() must raise ValueError when the test
    DataFrame has fewer than 100 rows.

    **Validates: Requirements 2.8, 2.9**
    """
    with pytest.raises(ValueError, match="teste"):
        validate_data(train_df, small_test)


# ── Property 6: Dados válidos sempre passam na validação ──────────────────────
# Validates: Requirement 2.10

@given(train_df=_valid_train_df(), test_df=_valid_test_df())
@settings(max_examples=30)
def test_prop6_valid_data_always_passes(train_df, test_df):
    """Property 6: validate_data() must return exactly True for any pair of
    DataFrames that satisfies all constraints simultaneously (columns present,
    labels in [1, 5], abstracts non-empty, size sufficient).

    **Validates: Requirement 2.10**
    """
    result = validate_data(train_df, test_df)
    assert result is True
