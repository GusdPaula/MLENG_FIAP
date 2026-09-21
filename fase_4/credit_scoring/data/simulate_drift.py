"""Simulação de Data Drift e Concept Drift no dataset de produção.

Alterações intencionais para representar mudança econômica:
  - renda_mensal: aumento de 80% (inflação / novos perfis de clientes)
  - score_credito: redução de 30% da distribuição (crise econômica)
  - anos_emprego: redução média (desemprego estrutural)
  - introdução de nulos em finalidade (5%) para disparar contrato
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


def simulate_production_drift(
    reference_df: pd.DataFrame,
    seed: int = 123,
    introduce_contract_violations: bool = True,
) -> pd.DataFrame:
    """Cria dataset de produção com drift aplicado ao dataset de referência."""
    rng = np.random.default_rng(seed)
    df = reference_df.copy()
    n = len(df)

    # Data Drift 1: renda aumenta ~80% (inflação / novos perfis premium)
    inflation_factor = rng.uniform(1.5, 2.2, size=n)
    df["renda_mensal"] = (df["renda_mensal"] * inflation_factor).round(2)

    # Data Drift 2: score_credito sofre compressão para baixo (crise)
    df["score_credito"] = (df["score_credito"] * rng.uniform(0.55, 0.80, size=n)).clip(300, 900).round(1)

    # Data Drift 3: anos_emprego reduz (instabilidade no mercado de trabalho)
    df["anos_emprego"] = np.clip(
        df["anos_emprego"] - rng.integers(0, 8, size=n),
        0,
        None,
    ).astype(int)

    # Concept Drift: valor do empréstimo aumenta proporcionalmente à renda
    df["valor_emprestimo"] = (df["valor_emprestimo"] * rng.uniform(1.2, 1.8, size=n)).round(2)

    # Concept Drift: mais inadimplências por conta da piora do score
    score_normalized = (df["score_credito"] - 300) / 600
    prob_inadimplente = np.clip(0.7 - 0.5 * score_normalized, 0.10, 0.85)
    df["inadimplente"] = (rng.uniform(size=n) < prob_inadimplente).astype(int)

    logger.info(
        "Drift aplicado: renda +%.0f%%, score -%.0f%%, inadimplência %.1f%% → %.1f%%",
        (df["renda_mensal"] / reference_df["renda_mensal"] - 1).mean() * 100,
        (1 - df["score_credito"] / reference_df["score_credito"]).mean() * 100,
        reference_df["inadimplente"].mean() * 100,
        df["inadimplente"].mean() * 100,
    )

    if introduce_contract_violations:
        # Viola contrato: insere nulos em finalidade (5% das linhas)
        null_idx = rng.choice(n, size=int(n * 0.05), replace=False)
        df.loc[null_idx, "finalidade"] = None

        # Viola contrato: duas linhas duplicadas
        dup_idx = rng.choice(n, size=2, replace=False)
        df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

        logger.info(
            "Violações de contrato inseridas: %d nulos em 'finalidade', 2 duplicatas",
            len(null_idx),
        )

    return df


def save_production(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or (DATA_DIR / "production.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Dataset de produção salvo em %s", path)
    return path
