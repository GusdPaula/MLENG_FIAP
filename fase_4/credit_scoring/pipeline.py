"""Pipeline orquestrador do Tech Challenge Fase 4.

Etapas:
  1. Gera dataset de referência → valida contrato (deve passar)
  2. Simula dataset de produção com drift → tenta validar (deve falhar)
  3. Treina modelo baseline no dataset de referência
  4. Avalia modelo no dataset de produção (concept drift)
  5. Roda Evidently para detectar data/concept drift
  6. Loga tudo no MLflow
"""

import logging
import sys

from credit_scoring.data.contracts import validate_safe
from credit_scoring.data.generate import generate_reference_dataset, save_reference
from credit_scoring.data.simulate_drift import save_production, simulate_production_drift
from credit_scoring.models.train import train_baseline
from credit_scoring.monitoring.drift_detector import run_drift_report
from credit_scoring.monitoring.mlflow_logger import log_drift_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEPARATOR = "=" * 60


def _print_section(title: str) -> None:
    logger.info("%s", SEPARATOR)
    logger.info("  %s", title)
    logger.info("%s", SEPARATOR)


def main() -> None:
    # --- Etapa 1: Geração e validação do dataset de referência ---
    _print_section("ETAPA 1 — Geração e Validação de Dados (Contrato Pandera)")

    logger.info("Gerando dataset de referência...")
    reference_df = generate_reference_dataset()
    save_reference(reference_df)

    ref_validated, ref_errors = validate_safe(reference_df)
    if ref_errors:
        logger.error("FALHA: dataset de referência não passou no contrato: %s", ref_errors)
        sys.exit(1)
    logger.info("✓ Dataset de referência validado com sucesso (%d amostras)", len(reference_df))

    # --- Etapa 1b: Demonstrar falha do contrato com dados ruins ---
    logger.info("Simulando dataset com violações de contrato...")
    production_raw = simulate_production_drift(reference_df, introduce_contract_violations=True)
    save_production(production_raw)

    prod_validated, prod_errors = validate_safe(production_raw)
    if prod_validated is None:
        logger.warning(
            "✓ Contrato bloqueou ingestão do dataset de produção (%d violações detectadas)",
            len(prod_errors),
        )
        logger.warning("  Violações: %s", prod_errors[:5])
    else:
        logger.warning("Dataset de produção passou no contrato (sem violações detectadas)")

    # Para análise de drift usamos versão sem violações de contrato
    production_df = simulate_production_drift(reference_df, introduce_contract_violations=False)

    # --- Etapa 2: Treinamento do modelo baseline ---
    _print_section("ETAPA 2 — Treinamento do Modelo Baseline + Simulação de Drift")

    logger.info("Treinando modelo baseline no dataset de referência...")
    train_result = train_baseline(reference_df)
    champion = train_result["champion_model"]
    metrics_ref = train_result["metrics"]

    logger.info(
        "✓ Campeão: %s — AUC=%.3f, Accuracy=%.3f",
        train_result["champion_name"],
        metrics_ref["auc_roc"],
        metrics_ref["accuracy"],
    )

    # Avalia no dataset de produção (concept drift)
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    from credit_scoring.models.train import FEATURE_COLS, _encode_categoricals

    prod_enc = _encode_categoricals(production_df)
    X_prod = prod_enc[[c for c in FEATURE_COLS if c in prod_enc.columns]]
    y_prod = prod_enc["inadimplente"]

    y_pred_prod = champion.predict(X_prod)
    y_proba_prod = champion.predict_proba(X_prod)[:, 1]

    metrics_prod = {
        "accuracy": accuracy_score(y_prod, y_pred_prod),
        "f1": f1_score(y_prod, y_pred_prod, zero_division=0),
        "auc_roc": roc_auc_score(y_prod, y_proba_prod),
    }
    logger.info(
        "Métricas em produção — AUC=%.3f, Accuracy=%.3f (ref: %.3f)",
        metrics_prod["auc_roc"],
        metrics_prod["accuracy"],
        metrics_ref["accuracy"],
    )

    degradation = metrics_ref["accuracy"] - metrics_prod["accuracy"]
    if degradation > 0.03:
        logger.warning(
            "⚠ ALERTA DE CONCEPT DRIFT: degradação de accuracy de %.1f pp",
            degradation * 100,
        )

    # --- Etapa 3: Detecção de Drift com Evidently ---
    _print_section("ETAPA 3 — Observabilidade: Detecção de Drift com Evidently + MLflow")

    logger.info("Executando relatório de drift (Evidently AI)...")
    report_path, drift_metrics = run_drift_report(reference_df, production_df)

    logger.info(
        "✓ Relatório gerado em %s",
        report_path,
    )
    logger.info(
        "  Features com drift: %s (%d de %d)",
        drift_metrics.get("drifted_features", []),
        drift_metrics.get("n_drifted_features", 0),
        len(["renda_mensal", "score_credito", "anos_emprego", "valor_emprestimo", "inadimplente"]),
    )

    if drift_metrics.get("dataset_drift"):
        logger.warning("⚠ ALERTA: DATASET DRIFT DETECTADO — modelo precisa ser retreinado!")

    logger.info("Logando resultados no MLflow...")
    run_id = log_drift_run(
        drift_metrics=drift_metrics,
        model_metrics_ref=metrics_ref,
        model_metrics_prod=metrics_prod,
    )

    # --- Resumo Final ---
    _print_section("RESUMO FINAL")
    logger.info("Pipeline concluída com sucesso!")
    logger.info("  MLflow run ID: %s", run_id)
    logger.info("  Relatório HTML: %s", report_path)
    logger.info(
        "  Data Drift: %s",
        "DETECTADO ⚠" if drift_metrics.get("dataset_drift") else "Não detectado ✓",
    )
    logger.info(
        "  Concept Drift: %s",
        f"DETECTADO ⚠ (Δaccuracy={degradation:.3f})" if degradation > 0.03 else "Não detectado ✓",
    )
    logger.info("")
    logger.info("Para visualizar o MLflow UI: mlflow ui --port 5000")
    logger.info("Relatório HTML: abra credit_scoring/reports/drift_report.html no browser")


if __name__ == "__main__":
    main()
