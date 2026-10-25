import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

entry_path = PROJECT_ROOT / "data/entries/loan_data_row.json"
entry_path_drifted = PROJECT_ROOT / "data/entries/loan_data_drifted.json"


def main(path: str | Path = entry_path) -> None:
    from default_pred.data.load import load_entries, load_one_entry
    from default_pred.models.predict import predict

    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and any(
        isinstance(value, dict) for value in data.values()
    ):
        df = load_entries(data)
    else:
        df = load_one_entry(data)

    prediction = predict(df)
    logger.info(f"Prediction: {prediction}")


if __name__ == "__main__":
    main(entry_path_drifted)
