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


def _validate_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a DataFrame (single entry or batch) against the Pandera Contract."""
    try:
        validated_df = Contract.validate(df, lazy=True)
        logger.info("Data validation passed (%d row(s))!", len(validated_df))
        return validated_df
    except pa.errors.SchemaErrors:
        logger.error("Data validation failed.")
        raise


def download_dataset_if_missing(target_path: str | Path) -> Path:
    """Download the Lending Club dataset from Kaggle via kagglehub if missing."""
    target_path = Path(target_path)
    if target_path.exists():
        return target_path

    logger.info(
        "Dataset not found locally at %s. Downloading via kagglehub...", target_path
    )
    import kagglehub

    download_dir = Path(
        kagglehub.dataset_download("adarshsng/lending-club-loan-data-csv")
    )
    csv_candidates = list(download_dir.glob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(
            f"No CSV files found in downloaded Kaggle bundle: {download_dir}"
        )

    source_csv = csv_candidates[0]
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Stream a representative subset with sequential IDs to satisfy Stage 1 pipelines
    logger.info("Sampling 5,000 rows from %s into %s", source_csv, target_path)
    header = pd.read_csv(source_csv, nrows=1).columns
    sample_df = pd.read_csv(source_csv, skiprows=range(1, 400000), nrows=5000)
    sample_df.columns = header
    sample_df["id"] = range(1, len(sample_df) + 1)
    sample_df.to_csv(target_path, index=False)
    logger.info("Dataset successfully downloaded and prepared at: %s", target_path)
    return target_path


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a dataset from a CSV file, downloading on demand if missing."""
    path = download_dataset_if_missing(path)
    logger.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path)

    return df


def load_entries(data: dict) -> pd.DataFrame:
    """Load and validate a batch dataset from a dict-of-dicts payload."""
    logger.info("Loading multiple entries from the provided data dictionary")

    if not isinstance(data, dict):
        raise TypeError("Expected a dictionary payload for batch loading.")

    if not any(isinstance(value, dict) for value in data.values()):
        return load_one_entry(data)

    df = pd.DataFrame(data).reset_index(drop=True)
    return _validate_contract(df)


def load_one_entry(data: dict) -> pd.DataFrame:
    """Load and validate a single entry from a dictionary."""
    logger.info("Loading a single entry from the provided data dictionary")
    normalized = _normalize_single_entry(data)
    df = pd.DataFrame([normalized])
    return _validate_contract(df)
