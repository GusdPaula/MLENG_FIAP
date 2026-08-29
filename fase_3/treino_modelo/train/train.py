import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"


def build_pipeline() -> Pipeline:
    """Constrói e retorna a pipeline scikit-learn (não treinada).

    Returns:
        Pipeline com TfidfVectorizer ('tfidf') e LinearSVC ('clf').
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 1),
            min_df=5,
            max_df=0.95,
            lowercase=True,
        )),
        ("clf", LinearSVC(
            C=0.06,
            random_state=42,
            dual=False,
        )),
    ])


def train_model(df_train: pd.DataFrame) -> Pipeline:
    """Treina a pipeline e persiste o modelo em disco.

    Args:
        df_train: DataFrame de treino com colunas 'medical_abstract'
            e 'condition_label'.

    Returns:
        Pipeline ajustada (fitted).

    Raises:
        KeyError: Se as colunas 'medical_abstract' ou 'condition_label'
            não existirem em df_train.
        Exception: Qualquer exceção de sklearn ou joblib é registrada
            no log e relançada.
    """
    logger.info("Iniciando treinamento do modelo...")
    start = time.time()

    # Validação de presença das colunas antes do fit
    for col in ("medical_abstract", "condition_label"):
        if col not in df_train.columns:
            msg = f"Coluna obrigatória ausente no DataFrame de treino: '{col}'"
            logger.error(msg)
            raise KeyError(msg)

    try:
        pipeline = build_pipeline()
        X_train = df_train["medical_abstract"]
        y_train = df_train["condition_label"]
        pipeline.fit(X_train, y_train)
    except Exception as exc:
        logger.error(f"Erro durante o ajuste da pipeline: {exc}")
        raise

    try:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
    except Exception as exc:
        logger.error(f"Erro durante a serialização do modelo: {exc}")
        raise

    elapsed = time.time() - start
    logger.info(f"Treinamento concluído em {elapsed:.2f}s. Modelo salvo em '{MODEL_PATH}'.")
    return pipeline
