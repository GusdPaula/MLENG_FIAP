import pandas as pd
from pathlib import Path


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a dataset from a CSV file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)
