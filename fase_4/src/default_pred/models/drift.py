import logging
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from evidently import Report
    from evidently.presets import DataDriftPreset
except ImportError:  # pragma: no cover - optional dependency fallback
    Report = Any
    DataDriftPreset = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_REPORT_PATH = "data_drift_report.html"


def _prepare_drift_data(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align current data to the training schema and sanitize values for drift."""
    common_columns = [
        col for col in reference_data.columns if col in current_data.columns
    ]
    reference_data = reference_data[common_columns].copy()
    current_data = current_data[common_columns].copy()

    if reference_data.empty or current_data.empty:
        logger.warning("No shared columns available for drift detection.")
        return reference_data, current_data

    numeric_columns = list(reference_data.select_dtypes(include=["number"]).columns)
    for column in numeric_columns:
        if column not in current_data.columns:
            continue

        ref_series = reference_data[column]
        cur_series = current_data[column]

        ref_std = ref_series.dropna().std(ddof=0)
        cur_std = cur_series.dropna().std(ddof=0)
        if ref_series.nunique(dropna=True) <= 1 or ref_std == 0 or cur_std == 0:
            reference_data = reference_data.drop(columns=[column])
            current_data = current_data.drop(columns=[column])
            continue

        fill_value = ref_series.median()
        if pd.isna(fill_value):
            fill_value = 0.0

        current_data[column] = cur_series.fillna(fill_value)
        reference_data[column] = ref_series.fillna(fill_value)

    for column in [col for col in current_data.columns if col not in numeric_columns]:
        mode = reference_data[column].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else ""
        current_data[column] = current_data[column].fillna(fill_value)
        reference_data[column] = reference_data[column].fillna(fill_value)

    return reference_data, current_data


def _load_reference_data() -> pd.DataFrame:
    """Load the training reference set or regenerate it from the raw dataset."""
    from default_pred.config import (
        PROCESSED_DATA_DIR,
        RANDOM_STATE,
        RAW_DATA_DIR,
        TARGET_COLUMN,
        TEST_SIZE,
    )
    from default_pred.data.load import load_dataset
    from default_pred.features.engineering import feature_engineering
    from default_pred.features.preprocessing import split_and_scale

    reference_path = PROCESSED_DATA_DIR / "X_train.pkl"
    if reference_path.exists():
        try:
            return pd.read_pickle(reference_path)
        except Exception as exc:  # pragma: no cover - runtime fallback path
            logger.warning(
                "Unable to read cached training data from %s: %s. "
                "Regenerating it from the source dataset.",
                reference_path,
                exc,
            )

    df = feature_engineering(load_dataset(RAW_DATA_DIR / "loan_data.csv"))
    X_train, _, _, _, _, _ = split_and_scale(
        df,
        target=TARGET_COLUMN,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    X_train.to_pickle(reference_path)
    return X_train


def detect_data_drift(
    X_train: pd.DataFrame | None = None,
    X_test: pd.DataFrame | None = None,
    output_path: str | Path = DEFAULT_REPORT_PATH,
) -> Report | None:
    """Detect data drift between the training reference data and current data."""
    if DataDriftPreset is None:
        logger.warning(
            "Evidently is not installed or incompatible; " "skipping drift check."
        )
        return None

    reference_data = X_train if X_train is not None else _load_reference_data()
    current_data = X_test if X_test is not None else reference_data.copy()
    reference_data, current_data = _prepare_drift_data(reference_data, current_data)

    logger.info("Starting data drift detection...")

    if current_data.shape[0] < 2 or reference_data.empty or current_data.empty:
        logger.warning(
            "Drift detection was skipped because the current dataset is too small "
            "or has insufficient variance for correlation-based drift checks."
        )
        return None

    # Comparison versus the production reference distribution learned at training time.
    report = Report([DataDriftPreset()])
    snapshot = report.run(
        current_data=current_data,
        reference_data=reference_data,
    )

    snapshot.save_html(str(output_path))

    logger.info(
        "Data drift detection completed. Report saved as '%s'.", output_path
    )

    return snapshot
