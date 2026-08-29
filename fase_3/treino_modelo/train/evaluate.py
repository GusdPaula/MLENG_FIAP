import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
REPORT_PATH = ARTIFACTS_DIR / "evaluation_report.txt"

CLASS_NAMES = [
    "neoplasms",
    "digestive system diseases",
    "nervous system diseases",
    "cardiovascular diseases",
    "general pathological conditions",
]

LABELS = [1, 2, 3, 4, 5]
ACCURACY_THRESHOLD = 0.50


def evaluate_model(
    pipeline: Pipeline,
    df_test: pd.DataFrame,
) -> dict[str, float | str]:
    """Avalia o modelo treinado no conjunto de teste.

    Args:
        pipeline: Pipeline scikit-learn já ajustada.
        df_test:  DataFrame de teste com colunas 'medical_abstract'
            e 'condition_label'.

    Returns:
        Dicionário com chaves:
            "accuracy" (float): acurácia global no conjunto de teste.
            "report"   (str):   relatório de classificação completo.

    Raises:
        Exception: Qualquer exceção de pipeline.predict() é registrada
            no log e relançada sem calcular métricas.
    """
    try:
        y_pred = pipeline.predict(df_test["medical_abstract"])
    except Exception as exc:
        logger.error(f"Erro durante a predição: {exc}")
        raise

    y_true = df_test["condition_label"]
    accuracy = float(accuracy_score(y_true, y_pred))
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    logger.info(f"Acurácia global: {accuracy:.4f}")

    if accuracy < ACCURACY_THRESHOLD:
        logger.warning(
            f"Acurácia ({accuracy:.4f}) abaixo do limiar mínimo aceitável "
            f"({ACCURACY_THRESHOLD})."
        )

    # Persiste o relatório; falha de I/O não bloqueia o retorno
    try:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report, encoding="utf-8")
        logger.info(f"Relatório salvo em '{REPORT_PATH}'.")
    except Exception as exc:
        logger.error(f"Falha ao salvar o relatório: {exc}")

    return {"accuracy": accuracy, "report": report}
