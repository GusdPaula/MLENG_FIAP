"""Pipeline CLI de Simulação de Data Drift, Prediction Drift e Concept Drift.

Este pipeline materializa em código limpo os dois momentos operacionais do ciclo de
vida de monitoramento de modelos sob *Feedback Delay*:

1. **Monitoramento Pré-Rótulo (t -> t + k)**:
   - Simula *Covariate Shift* (``P(X)``) aplicando compressão de -30% em ``annual_inc``
     e elevação de +30% em ``loan_amnt``.
   - Calcula o **Population Stability Index (PSI)** nas features (``psi.annual_inc``,
     ``psi.loan_amnt``) e nas probabilidades preditas pelo modelo
     (``psi.predicted_prob`` — *Prediction Drift*).
2. **Monitoramento Pós-Rótulo (t + k)**:
   - Simula *Concept Drift Real* (``P(Y | X)``) onde o choque de inflação e juros
     leva tomadores sob estresse financeiro à inadimplência.
   - Executa o detector supervisionado **DDM (Drift Detection Method)** sobre a
     sequência de erros de predição (``ddm.warning_triggered``, ``ddm.drift_triggered``)
     e quantifica a perda de Accuracy, Recall, F1-Score e ROC-AUC no MLflow.
"""

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
    PROJECT_ROOT / "data" / "entries" / "loan_data_drifted.json"
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
    """Executa simulação de Data/Concept Drift, calcula PSI/DDM e loga no MLflow."""
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
                "ddm_min_instances": config.ddm_min_instances,
            }
        )

        engineered_base = engineer(window_raw.copy())
        engineered_concept = engineer(both_drifted_raw.copy())
        common_index = engineered_base.index.intersection(engineered_concept.index)

        y_baseline = engineered_base.loc[common_index, TARGET_COLUMN].astype(int)
        y_concept = engineered_concept.loc[common_index, TARGET_COLUMN].astype(int)

        X_base = prepare_production_features(
            window_raw.loc[common_index],
            scaler,
            feature_columns,
            engineer,
        )
        X_drifted = prepare_production_features(
            both_drifted_raw.loc[common_index],
            scaler,
            feature_columns,
            engineer,
        )

        y_pred = model.predict(X_drifted)
        has_proba = hasattr(model, "predict_proba")
        ref_prob = model.predict_proba(X_base)[:, 1] if has_proba else None
        y_prob = model.predict_proba(X_drifted)[:, 1] if has_proba else None

        report = build_simulation_report(
            reference_df=reference_raw,
            drifted_df=both_drifted_raw,
            y_baseline=y_baseline,
            y_concept_drifted=y_concept,
            y_pred=y_pred,
            y_prob=y_prob,
            reference_prob=ref_prob,
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

        if report.concept_impact.ddm.drift_triggered:
            logger.warning(
                "DDM CONCEPT DRIFT ALERT: p_t=%.4f + s_t=%.4f >= "
                "p_min=%.4f + 3*s_min=%.4f (first_drift_idx=%s)",
                report.concept_impact.ddm.p_current,
                report.concept_impact.ddm.s_current,
                report.concept_impact.ddm.p_min,
                report.concept_impact.ddm.s_min,
                report.concept_impact.ddm.first_drift_index,
            )

        logger.info(
            "Concept Drift impact: default_rate %.2f%% -> %.2f%% | "
            "F1 %.4f -> %.4f (delta=%.4f) | ROC-AUC delta=%.4f",
            report.concept_impact.baseline_default_rate * 100,
            report.concept_impact.drifted_default_rate * 100,
            report.concept_impact.baseline_metrics["f1_score"],
            report.concept_impact.drifted_metrics["f1_score"],
            report.concept_impact.f1_delta,
            report.concept_impact.roc_auc_delta,
        )

        if output_json_path is not None:
            output_json_path.parent.mkdir(parents=True, exist_ok=True)
            both_drifted_raw.to_json(output_json_path, index=False)
            tracker.log_artifact(output_json_path)

        return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Simulate Data Drift, Concept Drift, DDM and PSI metrics."
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
