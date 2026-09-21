"""Centraliza o logging de observabilidade no MLflow.

Loga métricas de drift, qualidade e saúde do modelo como um run separado.
"""

import logging
from pathlib import Path

import mlflow

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent / "reports"


def log_drift_run(
    drift_metrics: dict,
    model_metrics_ref: dict,
    model_metrics_prod: dict | None = None,
    experiment_name: str = "credit_scoring_monitoring",
) -> str:
    """Cria um run MLflow com todas as métricas de observabilidade e artefatos."""
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="drift_monitoring") as run:
        # Métricas de drift do dataset
        mlflow.log_metric("n_drifted_features", drift_metrics.get("n_drifted_features", 0))
        mlflow.log_metric(
            "share_drifted_features", drift_metrics.get("share_drifted_features", 0.0)
        )
        mlflow.log_metric("dataset_drift_detected", int(drift_metrics.get("dataset_drift", False)))

        # Score por feature
        for feat, info in drift_metrics.get("feature_scores", {}).items():
            mlflow.log_metric(f"psi_{feat}", info.get("score", 0.0))

        # Métricas do modelo em referência vs produção
        for key, value in model_metrics_ref.items():
            mlflow.log_metric(f"ref_{key}", value)

        if model_metrics_prod:
            for key, value in model_metrics_prod.items():
                mlflow.log_metric(f"prod_{key}", value)

            # Degradação de performance
            if "accuracy" in model_metrics_ref and "accuracy" in model_metrics_prod:
                delta = model_metrics_prod["accuracy"] - model_metrics_ref["accuracy"]
                mlflow.log_metric("accuracy_delta", delta)
                if delta < -0.05:
                    logger.warning("ALERTA: degradação de accuracy de %.3f", abs(delta))

        # Tags de alerta
        alert = drift_metrics.get("dataset_drift", False) or (
            (model_metrics_prod or {}).get("accuracy", 1.0)
            < model_metrics_ref.get("accuracy", 1.0) - 0.05
        )
        mlflow.set_tag("drift_alert", str(alert))
        mlflow.set_tag("drifted_features", ", ".join(drift_metrics.get("drifted_features", [])))
        mlflow.set_tag("n_drifted", str(drift_metrics.get("n_drifted_features", 0)))

        # Artefatos HTML
        for html_file in REPORTS_DIR.glob("*.html"):
            mlflow.log_artifact(str(html_file), artifact_path="reports")
        for json_file in REPORTS_DIR.glob("*.json"):
            mlflow.log_artifact(str(json_file), artifact_path="reports")

        run_id = run.info.run_id
        logger.info(
            "MLflow run '%s' registrado — drift_alert=%s, features com drift: %s",
            run_id,
            alert,
            drift_metrics.get("drifted_features", []),
        )
        return run_id
