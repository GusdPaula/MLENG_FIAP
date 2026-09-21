"""Geração do dataset sintético German Credit para o dataset de referência (treino)."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SEED = 42
N_SAMPLES = 1000

DATA_DIR = Path(__file__).parent.parent / "data"


def generate_reference_dataset(n_samples: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    """Gera dataset sintético inspirado no German Credit com distribuição estável."""
    rng = np.random.default_rng(seed)

    idade = rng.integers(19, 75, size=n_samples).astype(int)
    anos_emprego = rng.integers(0, 30, size=n_samples).astype(int)

    renda_mensal = rng.lognormal(mean=8.5, sigma=0.5, size=n_samples).round(2)
    valor_emprestimo = (renda_mensal * rng.uniform(1.0, 8.0, size=n_samples)).round(2)
    score_credito = rng.uniform(300, 900, size=n_samples).round(1)
    historico_pagamentos = rng.choice(
        ["excelente", "bom", "regular", "ruim"],
        size=n_samples,
        p=[0.30, 0.40, 0.20, 0.10],
    )
    finalidade = rng.choice(
        ["automovel", "reforma", "educacao", "pessoal", "negocio"],
        size=n_samples,
        p=[0.25, 0.20, 0.15, 0.25, 0.15],
    )
    possui_conta_poupanca = rng.choice([0, 1], size=n_samples, p=[0.35, 0.65]).astype(int)

    # inadimplente: 1 = bom pagador, 0 = mau pagador (convenção German Credit)
    prob_bom = (
        0.3 * (score_credito - 300) / 600
        + 0.2 * (renda_mensal / renda_mensal.max())
        + 0.2 * (anos_emprego / 30)
        + 0.15 * np.where(historico_pagamentos == "excelente", 1.0, 0.0)
        + 0.15 * np.where(historico_pagamentos == "bom", 0.6, 0.0)
        + 0.1 * possui_conta_poupanca
    )
    prob_bom = np.clip(prob_bom, 0.05, 0.95)
    inadimplente = (rng.uniform(size=n_samples) > prob_bom).astype(int)

    df = pd.DataFrame(
        {
            "idade": idade,
            "anos_emprego": anos_emprego,
            "renda_mensal": renda_mensal,
            "valor_emprestimo": valor_emprestimo,
            "score_credito": score_credito,
            "historico_pagamentos": historico_pagamentos,
            "finalidade": finalidade,
            "possui_conta_poupanca": possui_conta_poupanca,
            "inadimplente": inadimplente,
        }
    )

    logger.info(
        "Dataset de referência gerado: %d amostras, %.1f%% inadimplentes",
        n_samples,
        df["inadimplente"].mean() * 100,
    )
    return df


def save_reference(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or (DATA_DIR / "reference.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Referência salva em %s", path)
    return path
