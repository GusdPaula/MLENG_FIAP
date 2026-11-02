from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from default_pred.models.drift_simulation import (
    DriftSimulationConfig,
    build_simulation_report,
    calculate_feature_psi,
    calculate_psi,
    detect_concept_drift_ddm,
    evaluate_concept_drift_impact,
    sample_production_window,
    simulate_concept_drift,
    simulate_data_drift,
)
from default_pred.observability.tracker import RunStatus
from default_pred.pipelines import pipeline_predict, pipeline_simulate_drift


class RecordingTracker:
    """In-memory ExperimentTracker spy for testing pipeline instrumentation."""

    def __init__(self) -> None:
        self.runs: list[tuple[str, str]] = []
        self.ended: list[RunStatus] = []
        self.params: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}
        self.artifacts: list[Path] = []

    def start_run(self, experiment: str, run_name: str) -> None:
        self.runs.append((experiment, run_name))

    def end_run(self, status: RunStatus) -> None:
        self.ended.append(status)

    def log_params(self, params: Mapping[str, Any]) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        self.metrics.update(metrics)

    def log_artifact(self, path: Path) -> None:
        self.artifacts.append(Path(path))

    def log_model(self, model: Any, name: str) -> None:
        pass


class DummyClassifier:
    """Deterministic binary classifier for testing Concept Drift degradation."""

    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names_in_ = np.array(feature_names)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        scores = np.linspace(0.05, 0.35, len(X))
        return np.column_stack([1.0 - scores, scores])


class DummyScaler:
    """Identity scaler with `feature_names_in_` attribute."""

    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names_in_ = np.array(feature_names)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return X.to_numpy(dtype=float)


def _make_reference_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    return pd.DataFrame(
        {
            "annual_inc": rng.normal(75_000, 10_000, size=n).clip(min=30_000),
            "loan_amnt": rng.normal(18_000, 3_000, size=n).clip(min=5_000),
            "grade": ["C" if i % 2 == 0 else "E" for i in range(n)],
            "loan_status": [
                "Fully Paid" if i % 5 != 0 else "Charged Off" for i in range(n)
            ],
        }
    )


def _simple_engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not pd.api.types.is_numeric_dtype(out["loan_status"]):
        out["loan_status"] = out["loan_status"].map(
            {"Fully Paid": 0, "Charged Off": 1}
        )
    out["grade"] = out["grade"].map({"A": 0, "B": 1, "C": 2, "D": 3, "E": 4})
    return out


def test_simulate_data_drift_is_pure_and_shifts_covariates() -> None:
    df = _make_reference_df(50)
    original_copy = df.copy(deep=True)
    config = DriftSimulationConfig(income_multiplier=0.7, loan_multiplier=1.3)

    drifted = simulate_data_drift(df, config)

    pd.testing.assert_frame_equal(df, original_copy)
    np.testing.assert_allclose(drifted["annual_inc"], df["annual_inc"] * 0.7)
    np.testing.assert_allclose(drifted["loan_amnt"], df["loan_amnt"] * 1.3)


def test_simulate_concept_drift_is_deterministic_and_increases_default_rate() -> None:
    df = _make_reference_df(100)
    original_copy = df.copy(deep=True)
    config = DriftSimulationConfig(
        random_state=42,
        stress_dti_threshold=0.20,
        concept_flip_probability=0.75,
    )

    drifted_a = simulate_concept_drift(df, config)
    drifted_b = simulate_concept_drift(df, config)

    pd.testing.assert_frame_equal(df, original_copy)
    pd.testing.assert_frame_equal(drifted_a, drifted_b)

    base_defaults = (df["loan_status"] == "Charged Off").mean()
    drifted_defaults = (drifted_a["loan_status"] == "Charged Off").mean()
    assert drifted_defaults > base_defaults


def test_simulate_concept_drift_supports_numeric_target_and_empty_frames() -> None:
    df_num = _simple_engineer(_make_reference_df(60))
    config = DriftSimulationConfig(random_state=7, concept_flip_probability=0.8)

    drifted_num = simulate_concept_drift(df_num, config)
    assert drifted_num["loan_status"].mean() > df_num["loan_status"].mean()

    empty_df = pd.DataFrame()
    assert simulate_concept_drift(empty_df, config).empty
    assert sample_production_window(empty_df, config).empty


def test_calculate_psi_distinguishes_stable_and_drifted_distributions() -> None:
    df = _make_reference_df(300)
    config = DriftSimulationConfig(income_multiplier=0.7, loan_multiplier=1.3)
    drifted = simulate_data_drift(df, config)

    same_psi = calculate_psi(df["annual_inc"], df["annual_inc"], bins=10)
    assert same_psi < 0.01

    psi_scores = calculate_feature_psi(df, drifted, columns=("annual_inc", "loan_amnt"))
    assert psi_scores["annual_inc"] > 0.2
    assert psi_scores["loan_amnt"] > 0.2
    assert calculate_psi(pd.Series(dtype=float), df["annual_inc"]) == 0.0


def test_detect_concept_drift_ddm_triggers_warning_and_drift() -> None:
    stable_errors = np.zeros(50, dtype=int)
    stable_errors[::10] = 1  # 10% error rate in baseline window
    drifted_errors = np.ones(60, dtype=int)  # sudden spike to 100% error rate
    stream = np.concatenate([stable_errors, drifted_errors])

    ddm = detect_concept_drift_ddm(stream, min_instances=20)
    assert ddm.warning_triggered is True
    assert ddm.drift_triggered is True
    assert ddm.first_drift_index is not None
    assert ddm.p_current > ddm.p_min

    empty_ddm = detect_concept_drift_ddm([])
    assert empty_ddm.drift_triggered is False
    assert empty_ddm.first_drift_index is None


def test_evaluate_concept_drift_impact_and_report_metrics() -> None:
    y_base = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 1])
    y_concept = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.2, 0.1, 0.9, 0.8, 0.2, 0.1, 0.3, 0.9])

    impact = evaluate_concept_drift_impact(
        y_base, y_concept, y_pred, y_prob, ddm_min_instances=5
    )
    assert impact.baseline_metrics["f1_score"] == 1.0
    assert impact.drifted_metrics["f1_score"] < impact.baseline_metrics["f1_score"]
    assert impact.f1_delta < 0.0
    assert impact.recall_delta < 0.0
    assert impact.roc_auc_delta <= 0.0
    assert impact.default_rate_delta > 0.0

    empty_impact = evaluate_concept_drift_impact([], [], [])
    assert empty_impact.baseline_default_rate == 0.0


def test_pipeline_simulate_drift_tracks_metrics_and_saves_artifact(
    tmp_path: Path,
) -> None:
    ref_df = _make_reference_df(120)
    feature_cols = ["annual_inc", "loan_amnt", "grade"]
    tracker = RecordingTracker()
    model = DummyClassifier(feature_cols)
    scaler = DummyScaler(["annual_inc", "loan_amnt"])
    out_json = tmp_path / "simulated_drift.json"
    config = DriftSimulationConfig(
        sample_size=80, random_state=42, ddm_min_instances=20
    )

    report = pipeline_simulate_drift.run(
        tracker=tracker,
        reference_raw=ref_df,
        model=model,
        scaler=scaler,
        feature_columns=feature_cols,
        engineer=_simple_engineer,
        config=config,
        output_json_path=out_json,
    )

    assert out_json.exists()
    assert out_json in tracker.artifacts
    assert tracker.ended == ["success"]
    assert tracker.metrics["psi.annual_inc"] > 0.2
    assert tracker.metrics["psi.loan_amnt"] > 0.2
    assert tracker.metrics["psi_drifted_count"] == 2.0
    assert "psi.predicted_prob" in tracker.metrics
    assert tracker.metrics["ddm.drift_triggered"] == 1.0
    assert "concept_drift.default_rate_delta" in tracker.metrics
    assert set(report.drifted_psi_features) == {"annual_inc", "loan_amnt"}


def test_build_simulation_report_and_predict_cli_parse_args(tmp_path: Path) -> None:
    df = _make_reference_df(50)
    drifted = simulate_data_drift(df, DriftSimulationConfig())
    report = build_simulation_report(
        reference_df=df,
        drifted_df=drifted,
        y_baseline=[0, 1, 0, 1],
        y_concept_drifted=[1, 1, 1, 1],
        y_pred=[0, 1, 0, 1],
        y_prob=[0.1, 0.9, 0.2, 0.8],
        reference_prob=[0.05, 0.85, 0.1, 0.75],
    )
    metrics = report.to_metrics()
    assert "psi.annual_inc" in metrics
    assert "psi.predicted_prob" in metrics
    assert "ddm.drift_triggered" in metrics

    custom_path = tmp_path / "fail.json"
    parsed = pipeline_predict.parse_args(["--input", str(custom_path)])
    assert parsed.input == custom_path
