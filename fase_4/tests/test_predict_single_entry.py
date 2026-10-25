import json
from pathlib import Path

from default_pred.data.load import load_one_entry
from default_pred.models.predict import predict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_predict_single_entry_returns_one_prediction():
    with open(PROJECT_ROOT / "data/entries/loan_data_row.json", "r") as f:
        row = json.load(f)

    df = load_one_entry(row)
    prediction = predict(df)

    assert len(prediction) == 1
