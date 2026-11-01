import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from default_pred.config import (  # noqa: E402
    EXPERIMENT_DRIFT,
    LOG_DIR,
    MODEL_DIR,
    RAW_DATA_DIR,
    TARGET_COLUMN,
)
from default_pred.models.drift_simulation import (  # noqa: E402
    DriftSimulationConfig,
    SimulationReport,
    build_simulation_report,
    sample_production_window,
    simulate_concept_drift,
    simulate_data_drift,
)
from default_pred.observability.run import tracked_run  # noqa: E402
from default_pred.observability.tracker import ExperimentTracker  # noqa: E402
from default_pred.pipelines.pipeline_drift import (  # noqa: E402
    prepare_production_features,
)

logger = logging.getLogger(__name__)

RUN_NAME = "drift_simulation"
DEFAULT_SIMULATED_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "entries" / "loan_data_concept_drifted.json"
)


def run(
    *,
    tracker: ExperimentTracker,
    reference_raw: pd.DataFrame,
    model: Any,
    scaler: Any,
    feature_columns: list[str],
    engineer: Callable[[pd.DataFrame], pd.DataFrame],
    config: DriftSimulationConfig = DriftSimulationConfig(),
    output_json_path: Path | None = None,
) -> SimulationReport:
    """Simulate Data and Concept Drift, compute PSI and track impact in MLflow."""
    with tracked_run(tracker, EXPERIMENT_DRIFT, RUN_NAME) as outcome:
        window_raw = sample_production_window(reference_raw, config)
        data_drifted_raw = simulate_data_drift(window_raw, config)
        both_drifted_raw = simulate_concept_drift(
            data_drifted_raw, config, target_column=TARGET_COLUMN
        )

        outcome.rows = len(both_drifted_raw)
        tracker.log_params(
            {
                "sample_size": len(both_drifted_raw),
                "random_state": config.random_state,
                "income_multiplier": config.income_multiplier,
                "loan_multiplier": config.loan_multiplier,
                "stress_dti_threshold": config.stress_dti_threshold,
                "concept_flip_probability": config.concept_flip_probability,
                "psi_bins": config.psi_bins,
                "psi_alert_threshold": config.psi_alert_threshold,
            }
        )

        engineered_base = engineer(window_raw.copy())
        engineered_concept = engineer(both_drifted_raw.copy())
        common_index = engineered_base.index.intersection(engineered_concept.index)

        y_baseline = engineered_base.loc[common_index, TARGET_COLUMN].astype(int)
        y_concept = engineered_concept.loc[common_index, TARGET_COLUMN].astype(int)

        X_drifted = prepare_production_features(
            both_drifted_raw.loc[common_index],
            scaler,
            feature_columns,
            engineer,
        )

        y_pred = model.predict(X_drifted)
        y_prob = (
            model.predict_proba(X_drifted)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        report = build_simulation_report(
            reference_df=reference_raw,
            drifted_df=both_drifted_raw,
            y_baseline=y_baseline,
            y_concept_drifted=y_concept,
            y_pred=y_pred,
            y_prob=y_prob,
            config=config,
        )

        tracker.log_metrics(report.to_metrics())

        if report.drifted_psi_features:
            logger.warning(
                "PSI DRIFT ALERT: features exceeding threshold %.2f: %s",
                config.psi_alert_threshold,
                ", ".join(
                    f"{feat}={report.psi_scores[feat]:.4f}"
                    for feat in report.drifted_psi_features
                ),
            )

        logger.info(
            "Concept Drift impact: default_rate %.2f%% -> %.2f%% (F1 delta=%.4f)",
            report.concept_impact.baseline_default_rate * 100,
            report.concept_impact.drifted_default_rate * 100,
            report.concept_impact.f1_delta,
        )

        if output_json_path is not None:
            output_json_path.parent.mkdir(parents=True, exist_ok=True)
            both_drifted_raw.to_json(output_json_path, index=False)
            tracker.log_artifact(output_json_path)

        return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Simulate Data Drift, Concept Drift, and PSI metrics."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SIMULATED_OUTPUT_PATH,
        help="Path to save the simulated drifted dataset JSON.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of rows to sample for the production simulation window.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> SimulationReport:  # pragma: no cover
    import joblib

    from default_pred.data.load import load_dataset
    from default_pred.features.engineering import feature_engineering
    from default_pred.observability.factory import build_tracker
    from default_pred.utils.logging import setup_logging

    args = parse_args(argv)
    setup_logging(LOG_DIR)

    reference_raw = load_dataset(RAW_DATA_DIR / "loan_data.csv")
    model = joblib.load(MODEL_DIR / "logistic_regression_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")

    config = DriftSimulationConfig(sample_size=args.sample_size)
    return run(
        tracker=build_tracker(),
        reference_raw=reference_raw,
        model=model,
        scaler=scaler,
        feature_columns=feature_columns,
        engineer=feature_engineering,
        config=config,
        output_json_path=args.output,
    )


if __name__ == "__main__":
    main()
