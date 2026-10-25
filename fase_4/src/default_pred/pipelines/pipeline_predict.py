import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main() -> None:
    from default_pred.data.load import load_one_entry
    from default_pred.models.predict import predict

    with open(os.path.join(PROJECT_ROOT, "data/entries/loan_data_row.json"), "r") as f:
        data = json.load(f)

    df = load_one_entry(data)
    prediction = predict(df)
    print(prediction)


if __name__ == "__main__":
    main()
