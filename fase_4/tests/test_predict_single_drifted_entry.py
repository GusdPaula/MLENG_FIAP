import json
from pathlib import Path

import pandas as pd

from default_pred.data.load import load_entries, load_one_entry
from default_pred.features.engineering import feature_engineering
from default_pred.models.drift import detect_data_drift

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_one_entry_accepts_drifted_mapping():
    with open(PROJECT_ROOT / "data/entries/loan_data_drifted.json", "r") as f:
        row = json.load(f)

    df = load_one_entry(row)

    assert len(df) == 1
    assert df.loc[0, "annual_inc"] > 20000
    assert df.loc[0, "purpose"] is not None
    assert df.loc[0, "grade"] in {"A", "B", "C", "D", "E", "F", "G"}


def test_detect_data_drift_skips_single_row_when_not_enough_variance():
    with open(PROJECT_ROOT / "data/entries/loan_data_row.json", "r") as f:
        row = json.load(f)

    raw_df = load_one_entry(row)
    processed_df = feature_engineering(raw_df.copy())
    reference = pd.read_pickle(PROJECT_ROOT / "data/processed/X_train.pkl")
    current = processed_df.reindex(columns=reference.columns, fill_value=0)

    snapshot = detect_data_drift(X_train=reference, X_test=current)

    assert snapshot is None


def test_load_entries_accepts_multi_row_drifted_json():
    with open(PROJECT_ROOT / "data/entries/loan_data_drifted.json", "r") as f:
        payload = json.load(f)

    df = load_entries(payload)

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 1
    assert set(df.columns) >= {"id", "annual_inc", "purpose", "grade"}
