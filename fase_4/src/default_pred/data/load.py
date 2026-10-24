import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a dataset from a CSV file."""
    path = Path(path)

    if not path.exists():
        logger.error(f"Dataset not found: {path}")
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info(f"Loading dataset from: {path}")
    return pd.read_csv(path)
