import json
from pathlib import Path

import pandera.pandas as pa
import pytest

from default_pred.data.load import load_one_entry

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_predict_single_entry_contract_fails():
    with open(PROJECT_ROOT / "data/entries/loan_data_row_fail.json", "r") as f:
        row = json.load(f)

    with pytest.raises(pa.errors.SchemaErrors):
        load_one_entry(row)
