"""Detecção de Data Drift e Concept Drift com Evidently AI 0.7.

Gera relatórios HTML comparando dataset de referência vs produção.
Usa Kolmogorov-Smirnov como teste estatístico principal.
"""

import json
import logging
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent / "reports"

NUMERIC_FEATURES = [
    "idade",
    "anos_emprego",
    "renda_mensal",
    "valor_emprestimo",
    "score_credito",
    "possui_conta_poupanca",
]
# inadimplente é tratado como numérico para monitorar concept drift
ALL_NUMERIC = NUMERIC_FEATURES + ["inadimplente"]


def _build_dataset(df: pd.DataFrame) -> Dataset:
    df_num = df.copy()
    df_num["inadimplente"] = df_num["inadimplente"].astype(float)
    data_def = DataDefinition(numerical_columns=ALL_NUMERIC)
    return Dataset.from_pandas(df_num[ALL_NUMERIC].dropna(), data_definition=data_def)


def run_drift_report(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    output_path: Path | None = None,
) -> tuple[Path, dict]:
    """Executa relatório de drift e retorna (caminho_html, métricas_json)."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = output_path or (REPORTS_DIR / "drift_report.html")

    ref_ds = _build_dataset(reference_df)
    prod_ds = _build_dataset(production_df)

    report = Report([DataDriftPreset(), DataSummaryPreset()])
    result = report.run(reference_data=ref_ds, current_data=prod_ds)
    result.save_html(str(output_path))
    logger.info("Relatório de drift salvo em %s", output_path)

    metrics = _extract_drift_metrics(result)
    metrics_path = REPORTS_DIR / "drift_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(
        "Métricas de drift: %d features com drift detectado",
        metrics.get("n_drifted_features", 0),
    )

    return output_path, metrics


def _extract_drift_metrics(result) -> dict:
    """Extrai métricas de drift do resultado do Evidently 0.7."""
    try:
        d = result.dict()
        metrics_raw = d.get("metrics", [])

        drift_metrics: dict = {
            "drifted_features": [],
            "n_drifted_features": 0,
            "share_drifted_features": 0.0,
            "dataset_drift": False,
            "feature_scores": {},
        }

        for m in metrics_raw:
            name = m.get("metric_name", "")
            value = m.get("value")

            # Contagem total de features com drift
            if "DriftedColumnsCount" in name:
                if isinstance(value, dict):
                    drift_metrics["n_drifted_features"] = int(value.get("count", 0))
                    drift_metrics["share_drifted_features"] = float(value.get("share", 0.0))
                    # considera drift no dataset se ≥30% das features têm drift
                    drift_metrics["dataset_drift"] = drift_metrics["share_drifted_features"] >= 0.3

            # Score por feature (p-value do KS test — menor = mais drift)
            elif "ValueDrift" in name:
                try:
                    col = name.split("column=")[1].split(",")[0]
                    p_value = float(value) if value is not None else 1.0
                    # Converte p-value em score de drift (0=sem drift, 1=muito drift)
                    drift_score = max(0.0, 1.0 - p_value)
                    drift_detected = p_value < 0.05
                    drift_metrics["feature_scores"][col] = {
                        "p_value": round(p_value, 6),
                        "drift_score": round(drift_score, 4),
                        "drift_detected": drift_detected,
                    }
                    if drift_detected:
                        drift_metrics["drifted_features"].append(col)
                except (IndexError, ValueError):
                    pass

        return drift_metrics

    except Exception as exc:
        logger.warning("Não foi possível extrair métricas detalhadas: %s", exc)
        return {"error": str(exc), "n_drifted_features": 0, "dataset_drift": False}
