"""Contrato de dados com Pandera para o pipeline de Credit Scoring.

Regras obrigatórias (≥3 conforme requisito do Tech Challenge):
  1. idade > 18
  2. renda_mensal não pode ser nula e deve ser positiva
  3. valor_emprestimo deve ser positivo
  4. score_credito no intervalo [300, 900]
  5. sem linhas duplicadas
"""

import logging

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

logger = logging.getLogger(__name__)

HISTORICO_VALIDOS = {"excelente", "bom", "regular", "ruim"}
FINALIDADES_VALIDAS = {"automovel", "reforma", "educacao", "pessoal", "negocio"}

credit_schema = DataFrameSchema(
    columns={
        "idade": Column(
            int,
            Check.greater_than(18),
            nullable=False,
            description="Idade do solicitante em anos (deve ser maior que 18)",
        ),
        "anos_emprego": Column(
            int,
            Check.greater_than_or_equal_to(0),
            nullable=False,
        ),
        "renda_mensal": Column(
            float,
            [
                Check.greater_than(0, error="Renda mensal deve ser positiva"),
            ],
            nullable=False,
            description="Renda mensal bruta em R$ (não pode ser nula)",
        ),
        "valor_emprestimo": Column(
            float,
            Check.greater_than(0, error="Valor do empréstimo deve ser positivo"),
            nullable=False,
        ),
        "score_credito": Column(
            float,
            Check.in_range(300, 900, error="Score de crédito deve estar entre 300 e 900"),
            nullable=False,
        ),
        "historico_pagamentos": Column(
            str,
            Check.isin(HISTORICO_VALIDOS, error=f"Histórico deve ser um de: {HISTORICO_VALIDOS}"),
            nullable=False,
        ),
        "finalidade": Column(
            str,
            Check.isin(
                FINALIDADES_VALIDAS, error=f"Finalidade deve ser uma de: {FINALIDADES_VALIDAS}"
            ),
            nullable=False,
        ),
        "possui_conta_poupanca": Column(
            int,
            Check.isin([0, 1], error="possui_conta_poupanca deve ser 0 ou 1"),
            nullable=False,
        ),
        "inadimplente": Column(
            int,
            Check.isin([0, 1], error="inadimplente deve ser 0 ou 1"),
            nullable=False,
        ),
    },
    checks=[
        Check(
            lambda df: df.duplicated().sum() == 0,
            error="Dataset contém linhas duplicadas",
        )
    ],
    name="CreditScoringSchema",
)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Valida o DataFrame contra o contrato. Levanta SchemaError se falhar."""
    logger.info("Iniciando validação do contrato de dados (%d linhas)...", len(df))
    validated = credit_schema.validate(df, lazy=True)
    logger.info("Contrato validado com sucesso.")
    return validated


def validate_safe(df: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str]]:
    """Valida sem lançar exceção. Retorna (df_validado, lista_de_erros)."""
    try:
        return validate(df), []
    except pa.errors.SchemaErrors as exc:
        try:
            errors = exc.failure_cases["failure_case"].astype(str).tolist()
        except Exception:
            errors = [str(exc)]
        logger.warning("Contrato falhou com %d violações: %s", len(errors), errors[:5])
        return None, errors
    except pa.errors.SchemaError as exc:
        logger.warning("Contrato falhou: %s", exc)
        return None, [str(exc)]
