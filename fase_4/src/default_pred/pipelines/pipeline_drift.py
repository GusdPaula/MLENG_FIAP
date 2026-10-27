import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from default_pred.config import (  # noqa: E402
    DRIFT_ALERT_SHARE_THRESHOLD,
    DRIFT_REPORT_PATH,
    EXPERIMENT_DRIFT,
    LOG_DIR,
    MODEL_DIR,
)
from default_pred.observability.drift_summary import (  # noqa: E402
    DriftSummary,
    summarize_drift,
)
from default_pred.observability.run import tracked_run  # noqa: E402
from default_pred.observability.tracker import ExperimentTracker  # noqa: E402

logger = logging.getLogger(__name__)

MIN_PRODUCTION_ROWS = 2
RUN_NAME = "drift_detection"
DRIFTED_ENTRY_PATH = PROJECT_ROOT / "data" / "entries" / "loan_data_drifted.json"

DetectFn = Callable[[pd.DataFrame, pd.DataFrame, Path], object]


def prepare_production_features(
    raw: pd.DataFrame,
    scaler,
    feature_columns: list[str],
    engineer: Callable[[pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
    """Apply the training feature pipeline to raw production rows."""
    df = engineer(raw).reindex(columns=feature_columns, fill_value=0)

    scaler_columns = list(getattr(scaler, "feature_names_in_", []))
    if scaler_columns:
        scaled = pd.DataFrame(
            scaler.transform(df[scaler_columns]),
            columns=scaler_columns,
            index=df.index,
        )
        others = [col for col in feature_columns if col not in scaler_columns]
        df = pd.concat([df[others], scaled], axis=1).reindex(columns=feature_columns)
    return df


def run(
    *,
    tracker: ExperimentTracker,
    reference: pd.DataFrame,
    production: pd.DataFrame,
    detect_fn: DetectFn,
    report_path: Path,
    alert_threshold: float,
) -> DriftSummary | None:
    """Detect drift, summarize it and record everything in the tracker."""
    with tracked_run(tracker, EXPERIMENT_DRIFT, RUN_NAME) as outcome:
        outcome.rows = len(production)
        tracker.log_params(
            {
                "reference_rows": len(reference),
                "production_rows": len(production),
                "alert_threshold": alert_threshold,
            }
        )

        if len(production) < MIN_PRODUCTION_ROWS:
            logger.warning(
                "Drift detection skipped: production sample has %d row(s), "
                "minimum is %d.",
                len(production),
                MIN_PRODUCTION_ROWS,
            )
            outcome.status = "skipped"
            return None

        snapshot = detect_fn(reference, production, report_path)
        if snapshot is None:
            logger.warning("Drift detection returned no report; run skipped.")
            outcome.status = "skipped"
            return None

        summary = summarize_drift(snapshot.dict(), alert_threshold)
        tracker.log_metrics(summary.to_metrics())

        if summary.alert:
            logger.warning(
                "DRIFT ALERT: %d/%d features drifted (share=%.2f >= %.2f)",
                summary.n_drifted,
                summary.n_features,
                summary.share_drifted,
                alert_threshold,
            )

        if report_path.exists():
            tracker.log_artifact(report_path)
        return summary


def main() -> None:  # pragma: no cover - wires real dependencies
    import joblib

    from default_pred.data.load import load_entries
    from default_pred.features.engineering import feature_engineering
    from default_pred.models.drift import _load_reference_data, detect_data_drift
    from default_pred.observability.factory import build_tracker
    from default_pred.utils.logging import setup_logging

    setup_logging(LOG_DIR)

    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")
    with open(DRIFTED_ENTRY_PATH, "r") as file:
        raw = load_entries(json.load(file))

    production = prepare_production_features(
        raw, scaler, feature_columns, feature_engineering
    )
    reference = _load_reference_data()

    def detect(ref: pd.DataFrame, cur: pd.DataFrame, path: Path):
        return detect_data_drift(X_train=ref, X_test=cur, output_path=path)

    run(
        tracker=build_tracker(),
        reference=reference,
        production=production,
        detect_fn=detect,
        report_path=DRIFT_REPORT_PATH,
        alert_threshold=DRIFT_ALERT_SHARE_THRESHOLD,
    )


if __name__ == "__main__":
    main()
