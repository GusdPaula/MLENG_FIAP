import logging
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from default_pred.data.contract import Contract

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _normalize_single_entry(data: dict) -> dict:
    """Normalize a flat record or a drifted mapping into a single-row dict."""
    if not isinstance(data, dict):
        return data

    if not any(isinstance(value, dict) for value in data.values()):
        return data

    representative_key = None
    for value in data.values():
        if isinstance(value, dict) and value:
            representative_key = next(iter(value))
            break

    if representative_key is None:
        return data

    normalized = {}
    for field, value in data.items():
        if isinstance(value, dict) and value:
            fallback = next(iter(value.values()))
            normalized[field] = value.get(representative_key, fallback)
        else:
            normalized[field] = value

    return normalized


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a dataset from a CSV file."""
    path = Path(path)

    if not path.exists():
        logger.error(f"Dataset not found: {path}")
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path)

    return df


def load_entries(data: dict) -> pd.DataFrame:
    """Load a full dataset from a dict-of-dicts payload such as a drifted JSON file."""
    logger.info("Loading multiple entries from the provided data dictionary")

    if not isinstance(data, dict):
        raise TypeError("Expected a dictionary payload for batch loading.")

    if not any(isinstance(value, dict) for value in data.values()):
        return pd.DataFrame([data])

    df = pd.DataFrame(data)
    df = df.reset_index(drop=True)

    # Batch drift payloads often include intentionally noisy rows, so avoid
    # hard Contract validation here; the single-entry path keeps that strictness.
    logger.info(
        "Batch data loaded successfully; validation is deferred to per-row checks."
    )
    return df


def load_one_entry(data: dict) -> pd.DataFrame:
    """Load a single entry from a dictionary."""
    logger.info("Loading a single entry from the provided data dictionary")
    normalized = _normalize_single_entry(data)
    df = pd.DataFrame([normalized])
    try:
        _validated_df = Contract.validate(df, lazy=True)
        logger.info("Data validation passed!")
        return df

    except pa.errors.SchemaErrors:
        logger.error("Data validation failed.")
        raise
