import json
from pathlib import Path

import pandera.pandas as pa
import pytest

from default_pred.data.load import load_one_entry

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_predict_single_entry_contract_fails():
    with open(PROJECT_ROOT / "data/entries/loan_data_row_fail.json", "r") as f:
        row = json.load(f)

    with pytest.raises(pa.errors.SchemaErrors) as exc_info:
        load_one_entry(row)

    failed_columns = set(exc_info.value.failure_cases["column"].dropna())
    assert {"annual_inc", "purpose", "grade"} <= failed_columns


def test_load_one_entry_coerces_integer_annual_inc():
    with open(PROJECT_ROOT / "data/entries/loan_data_row.json", "r") as f:
        row = json.load(f)

    row["annual_inc"] = 55000  # int from JSON instead of float
    df = load_one_entry(row)

    assert df["annual_inc"].dtype == "float64"
    assert df.loc[0, "annual_inc"] == 55000.0
