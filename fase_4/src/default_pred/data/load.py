import pandas as pd
from pathlib import Path
import logging
from default_pred.data.contract import Contract
import pandera.pandas as pa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a dataset from a CSV file."""
    path = Path(path)

    if not path.exists():
        logger.error(f"Dataset not found: {path}")
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path)

    return df


def load_one_entry(data: dict) -> pd.DataFrame:
    """Load a single entry from a dictionary."""
    logger.info("Loading a single entry from the provided data dictionary")
    df = pd.DataFrame([data])
    try:
        _validated_df = Contract.validate(df, lazy=True)
        logger.info("Data validation passed!")
        return df

    except pa.errors.SchemaErrors:
        logger.error("Data validation failed.")
        raise
