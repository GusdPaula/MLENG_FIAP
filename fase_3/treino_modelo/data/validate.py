import logging
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["condition_label", "medical_abstract"]
VALID_LABELS = set(range(1, 6))       # {1, 2, 3, 4, 5}
MIN_TRAIN_ROWS = 1_000
MIN_TEST_ROWS  = 100


def validate_data(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> bool:
    """Valida a integridade dos DataFrames de treino e teste.

    As validações são executadas nesta ordem obrigatória:
      1. Presença de colunas obrigatórias
      2. Tipo e intervalo de condition_label
      3. Ausência de valores nulos/vazios em medical_abstract
      4. Tamanho mínimo dos DataFrames

    Returns:
        True se todas as validações passarem.

    Raises:
        ValueError: Em qualquer falha de validação, com mensagem
            descritiva indicando a causa e o DataFrame afetado.
    """
    # 1. Presença de colunas
    _check_columns(df_train, "treino")
    _check_columns(df_test, "teste")

    # 2. Tipo e intervalo de condition_label
    _check_labels(df_train, "treino")
    _check_labels(df_test, "teste")

    # 3. Valores nulos/vazios em medical_abstract
    _check_abstracts(df_train, "treino")
    _check_abstracts(df_test, "teste")

    # 4. Tamanho mínimo
    _check_size(df_train, "treino", MIN_TRAIN_ROWS)
    _check_size(df_test, "teste", MIN_TEST_ROWS)

    logger.info("Validação concluída com sucesso.")
    return True


# ── Funções auxiliares ────────────────────────────────────────────

def _check_columns(df: pd.DataFrame, name: str) -> None:
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(
                f"Coluna obrigatória '{col}' ausente no DataFrame de {name}."
            )


def _check_labels(df: pd.DataFrame, name: str) -> None:
    invalid = df[~df["condition_label"].isin(VALID_LABELS)]["condition_label"].unique()
    if len(invalid) > 0:
        raise ValueError(
            f"Valores inválidos em 'condition_label' no DataFrame de {name}: "
            f"{sorted(invalid.tolist())}. Esperado: valores inteiros entre 1 e 5."
        )


def _check_abstracts(df: pd.DataFrame, name: str) -> None:
    null_count   = df["medical_abstract"].isna().sum()
    empty_count  = (df["medical_abstract"].astype(str).str.strip() == "").sum()
    invalid_count = null_count + empty_count
    if invalid_count > 0:
        raise ValueError(
            f"{invalid_count} entradas inválidas (nulas, vazias ou apenas espaços) "
            f"em 'medical_abstract' no DataFrame de {name}."
        )


def _check_size(df: pd.DataFrame, name: str, minimum: int) -> None:
    if len(df) < minimum:
        raise ValueError(
            f"DataFrame de {name} possui {len(df)} registros; "
            f"mínimo esperado: {minimum}."
        )
