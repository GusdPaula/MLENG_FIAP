import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

EPSILON = 1e-4
HIGH_RISK_GRADES = frozenset({"D", "E", "F", "G", 3, 4, 5, 6})
NON_DEFAULT_STRING_STATUSES = frozenset(
    {
        "Fully Paid",
        "Current",
        "Does not meet the credit policy. Status:Fully Paid",
    }
)


@dataclass(frozen=True)
class DriftSimulationConfig:
    """Immutable configuration for Data Drift and Concept Drift simulation."""

    sample_size: int = 1000
    random_state: int = 42
    income_multiplier: float = 0.7
    loan_multiplier: float = 1.3
    stress_dti_threshold: float = 0.25
    concept_flip_probability: float = 0.65
    psi_bins: int = 10
    psi_alert_threshold: float = 0.2


@dataclass(frozen=True)
class ConceptDriftImpact:
    """Comparison of model performance before and after Concept Drift P(Y|X)."""

    baseline_metrics: Mapping[str, float]
    drifted_metrics: Mapping[str, float]
    baseline_default_rate: float
    drifted_default_rate: float

    @property
    def f1_delta(self) -> float:
        return float(
            self.drifted_metrics.get("f1_score", 0.0)
            - self.baseline_metrics.get("f1_score", 0.0)
        )

    @property
    def recall_delta(self) -> float:
        return float(
            self.drifted_metrics.get("recall", 0.0)
            - self.baseline_metrics.get("recall", 0.0)
        )

    @property
    def roc_auc_delta(self) -> float:
        return float(
            self.drifted_metrics.get("roc_auc", 0.0)
            - self.baseline_metrics.get("roc_auc", 0.0)
        )

    @property
    def default_rate_delta(self) -> float:
        return float(self.drifted_default_rate - self.baseline_default_rate)


@dataclass(frozen=True)
class SimulationReport:
    """Aggregated statistical (PSI) and predictive report of a drift simulation."""

    psi_scores: Mapping[str, float]
    psi_alert_threshold: float
    concept_impact: ConceptDriftImpact

    @property
    def drifted_psi_features(self) -> tuple[str, ...]:
        return tuple(
            feature
            for feature, score in self.psi_scores.items()
            if score >= self.psi_alert_threshold
        )

    def to_metrics(self) -> dict[str, float]:
        """Flatten PSI and Concept Drift impact into tracker-ready numeric metrics."""
        metrics: dict[str, float] = {
            "psi_drifted_count": float(len(self.drifted_psi_features)),
            "baseline.default_rate": float(self.concept_impact.baseline_default_rate),
            "concept_drift.default_rate": float(
                self.concept_impact.drifted_default_rate
            ),
            "concept_drift.default_rate_delta": self.concept_impact.default_rate_delta,
            "concept_drift.f1_delta": self.concept_impact.f1_delta,
            "concept_drift.recall_delta": self.concept_impact.recall_delta,
            "concept_drift.roc_auc_delta": self.concept_impact.roc_auc_delta,
        }
        for feature, score in self.psi_scores.items():
            metrics[f"psi.{feature}"] = float(score)
        for name, value in self.concept_impact.baseline_metrics.items():
            metrics[f"baseline.{name}"] = float(value)
        for name, value in self.concept_impact.drifted_metrics.items():
            metrics[f"concept_drift.{name}"] = float(value)
        return metrics


def sample_production_window(
    df: pd.DataFrame, config: DriftSimulationConfig
) -> pd.DataFrame:
    """Deterministically sample a production batch without mutating the source."""
    if df.empty:
        return df.copy()
    n_rows = min(config.sample_size, len(df))
    return df.sample(n=n_rows, random_state=config.random_state).copy()


def simulate_data_drift(
    df: pd.DataFrame, config: DriftSimulationConfig
) -> pd.DataFrame:
    """Simulate covariate shift P(X) caused by income compression and credit demand."""
    drifted = df.copy()
    if "annual_inc" in drifted.columns:
        drifted["annual_inc"] = (
            drifted["annual_inc"].astype(float) * config.income_multiplier
        )
    if "loan_amnt" in drifted.columns:
        drifted["loan_amnt"] = (
            drifted["loan_amnt"].astype(float) * config.loan_multiplier
        )
    return drifted


def _identify_stressed_borrowers(
    df: pd.DataFrame, stress_dti_threshold: float
) -> pd.Series:
    """Identify rows vulnerable to macroeconomic shock (high loan-to-income or risk)."""
    if df.empty:
        return pd.Series(dtype=bool, index=df.index)

    stressed = pd.Series(False, index=df.index)
    if "loan_amnt" in df.columns and "annual_inc" in df.columns:
        safe_income = df["annual_inc"].astype(float).clip(lower=1.0)
        loan_to_income = df["loan_amnt"].astype(float) / safe_income
        stressed = stressed | (loan_to_income >= stress_dti_threshold)

    if "grade" in df.columns:
        stressed = stressed | df["grade"].isin(HIGH_RISK_GRADES)

    return stressed


def simulate_concept_drift(
    df: pd.DataFrame,
    config: DriftSimulationConfig,
    target_column: str = "loan_status",
) -> pd.DataFrame:
    """Simulate posterior shift P(Y|X) where inflation/rates increase defaults."""
    drifted = df.copy()
    if target_column not in drifted.columns or drifted.empty:
        return drifted

    stressed_mask = _identify_stressed_borrowers(drifted, config.stress_dti_threshold)
    rng = np.random.default_rng(config.random_state)
    draws = pd.Series(rng.random(len(drifted)), index=drifted.index)
    flip_Trigger = stressed_mask & (draws < config.concept_flip_probability)

    target_series = drifted[target_column]
    if pd.api.types.is_numeric_dtype(target_series):
        eligible = target_series == 0
        drifted.loc[eligible & flip_Trigger, target_column] = 1
    else:
        eligible = target_series.isin(NON_DEFAULT_STRING_STATUSES)
        drifted.loc[eligible & flip_Trigger, target_column] = "Charged Off"

    return drifted


def calculate_psi(
    reference: pd.Series, current: pd.Series, bins: int = 10
) -> float:
    """Calculate the Population Stability Index (PSI) between two numeric series."""
    ref_clean = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    cur_clean = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)

    if len(ref_clean) == 0 or len(cur_clean) == 0 or bins < 2:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    internal_edges = np.unique(np.quantile(ref_clean, quantiles[1:-1]))
    bin_edges = np.concatenate(([-np.inf], internal_edges, [np.inf]))

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    cur_counts, _ = np.histogram(cur_clean, bins=bin_edges)

    ref_ratios = np.clip(ref_counts / len(ref_clean), EPSILON, None)
    cur_ratios = np.clip(cur_counts / len(cur_clean), EPSILON, None)

    psi_value = float(
        np.sum((cur_ratios - ref_ratios) * np.log(cur_ratios / ref_ratios))
    )
    return max(0.0, psi_value)


def calculate_feature_psi(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: Sequence[str] = ("annual_inc", "loan_amnt"),
    bins: int = 10,
) -> dict[str, float]:
    """Compute PSI scores for the requested numeric columns present in both sets."""
    scores: dict[str, float] = {}
    for col in columns:
        if col in reference.columns and col in current.columns:
            scores[col] = calculate_psi(reference[col], current[col], bins=bins)
    return scores


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = 0.0
    return metrics


def evaluate_concept_drift_impact(
    y_baseline: Sequence[int] | pd.Series,
    y_concept_drifted: Sequence[int] | pd.Series,
    y_pred: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray | None = None,
) -> ConceptDriftImpact:
    """Measure how Concept Drift degrades the baseline classifier's predictions."""
    y_base_arr = np.asarray(y_baseline, dtype=int)
    y_drift_arr = np.asarray(y_concept_drifted, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float) if y_prob is not None else None

    if len(y_base_arr) == 0 or len(y_drift_arr) == 0:
        empty: dict[str, float] = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "roc_auc": 0.0,
        }
        return ConceptDriftImpact(
            baseline_metrics=empty,
            drifted_metrics=empty,
            baseline_default_rate=0.0,
            drifted_default_rate=0.0,
        )

    baseline_metrics = _classification_metrics(y_base_arr, y_pred_arr, y_prob_arr)
    drifted_metrics = _classification_metrics(y_drift_arr, y_pred_arr, y_prob_arr)

    return ConceptDriftImpact(
        baseline_metrics=baseline_metrics,
        drifted_metrics=drifted_metrics,
        baseline_default_rate=float(np.mean(y_base_arr)),
        drifted_default_rate=float(np.mean(y_drift_arr)),
    )


def build_simulation_report(
    *,
    reference_df: pd.DataFrame,
    drifted_df: pd.DataFrame,
    y_baseline: Sequence[int] | pd.Series,
    y_concept_drifted: Sequence[int] | pd.Series,
    y_pred: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray | None = None,
    config: DriftSimulationConfig = DriftSimulationConfig(),
    monitored_columns: Sequence[str] = ("annual_inc", "loan_amnt"),
) -> SimulationReport:
    """Build a complete SimulationReport combining PSI and Concept Drift impact."""
    psi_scores = calculate_feature_psi(
        reference_df, drifted_df, columns=monitored_columns, bins=config.psi_bins
    )
    concept_impact = evaluate_concept_drift_impact(
        y_baseline=y_baseline,
        y_concept_drifted=y_concept_drifted,
        y_pred=y_pred,
        y_prob=y_prob,
    )
    return SimulationReport(
        psi_scores=psi_scores,
        psi_alert_threshold=config.psi_alert_threshold,
        concept_impact=concept_impact,
    )
