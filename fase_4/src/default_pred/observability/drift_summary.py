import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

VALUE_DRIFT_TYPE = "evidently:metric_v2:ValueDrift"


@dataclass(frozen=True)
class FeatureDrift:
    """Drift result of a single feature."""

    name: str
    method: str
    score: float
    threshold: float
    drifted: bool


@dataclass(frozen=True)
class DriftSummary:
    """Backend-agnostic summary of a drift report."""

    features: tuple[FeatureDrift, ...]
    alert_threshold: float

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def n_drifted(self) -> int:
        return sum(1 for feature in self.features if feature.drifted)

    @property
    def share_drifted(self) -> float:
        if not self.features:
            return 0.0
        return self.n_drifted / self.n_features

    @property
    def alert(self) -> bool:
        return self.n_features > 0 and self.share_drifted >= self.alert_threshold

    def to_metrics(self) -> dict[str, float]:
        """Flat numeric metrics suitable for a tracker."""
        metrics = {
            "n_features": float(self.n_features),
            "n_drifted_features": float(self.n_drifted),
            "share_drifted": self.share_drifted,
            "alert_triggered": 1.0 if self.alert else 0.0,
        }
        for feature in self.features:
            metrics[f"drift_score.{feature.name}"] = feature.score
        return metrics


def _is_drifted(method: str, score: float, threshold: float) -> bool:
    """p-value methods drift below the threshold; distance methods at/above it."""
    if "p_value" in method:
        return score < threshold
    return score >= threshold


def _parse_feature(metric: Mapping[str, Any]) -> FeatureDrift:
    try:
        config = metric["config"]
        score = float(metric["value"])
        method = str(config["method"])
        threshold = float(config["threshold"])
        name = str(config["column"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Malformed ValueDrift metric: {metric!r}") from exc
    return FeatureDrift(
        name=name,
        method=method,
        score=score,
        threshold=threshold,
        drifted=_is_drifted(method, score, threshold),
    )


def summarize_drift(
    snapshot_dict: Mapping[str, Any], alert_threshold: float
) -> DriftSummary:
    """Convert an Evidently `snapshot.dict()` into a DriftSummary."""
    metrics = snapshot_dict.get("metrics")
    if not isinstance(metrics, list):
        raise ValueError("Snapshot dict has no 'metrics' list.")

    features = tuple(
        _parse_feature(metric)
        for metric in metrics
        if metric.get("config", {}).get("type") == VALUE_DRIFT_TYPE
    )
    summary = DriftSummary(features=features, alert_threshold=alert_threshold)
    logger.info(
        "Drift summary: %d/%d features drifted (share=%.2f, alert=%s)",
        summary.n_drifted,
        summary.n_features,
        summary.share_drifted,
        summary.alert,
    )
    return summary
