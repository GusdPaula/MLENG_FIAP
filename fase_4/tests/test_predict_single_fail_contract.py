import json
from pathlib import Path

import pandera.pandas as pa
import pytest

from default_pred.data.load import load_entries, load_one_entry

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_valid_row() -> dict:
    with open(PROJECT_ROOT / "data/entries/loan_data_row.json", "r") as f:
        return json.load(f)


def test_predict_single_entry_contract_fails():
    with open(PROJECT_ROOT / "data/entries/loan_data_row_fail.json", "r") as f:
        row = json.load(f)

    with pytest.raises(pa.errors.SchemaErrors) as exc_info:
        load_one_entry(row)

    failed_columns = set(exc_info.value.failure_cases["column"].dropna())
    assert {"annual_inc", "purpose", "grade"} <= failed_columns


def test_load_one_entry_coerces_integer_fields():
    row = _load_valid_row()
    row["annual_inc"] = 55000
    row["loan_amnt"] = 3600
    df = load_one_entry(row)

    assert df["annual_inc"].dtype == "float64"
    assert df["loan_amnt"].dtype == "float64"
    assert df.loc[0, "annual_inc"] == 55000.0


def test_load_entries_blocks_invalid_batch_and_duplicate_ids():
    row_a = _load_valid_row()
    row_b = _load_valid_row()
    # Force duplicate id, invalid loan_amnt, invalid term, and temporal inversion
    row_b["loan_amnt"] = -500.0
    row_b["term"] = " 120 months"
    row_b["issue_d"] = "Jan-2000"
    row_b["earliest_cr_line"] = "Aug-2003"

    batch_payload = {
        key: {"0": row_a[key], "1": row_b[key]} for key in row_a.keys()
    }

    with pytest.raises(pa.errors.SchemaErrors) as exc_info:
        load_entries(batch_payload)

    failures = exc_info.value.failure_cases
    failed_columns = set(failures["column"].dropna())
    failed_checks = set(failures["check"].dropna())

    assert {"id", "loan_amnt", "term"} <= failed_columns
    assert "field_uniqueness" in failed_checks
    assert "earliest_cr_line must be earlier than or equal to issue_d" in failed_checks
