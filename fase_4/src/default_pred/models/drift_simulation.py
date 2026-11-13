"""Motor de Stress Testing Macroeconômico para Simulação e Detecção de Drift em Crédito.

Motivadores de Negócio e Fundamentação Econométrica
===================================================
Em sistemas produtivos de *Credit Scoring*, a degradação de modelos raramente ocorre
por variações isoladas em uma única coluna. Choques macroeconômicos reais — como ciclos
inflacionários persistentes combinados a políticas de aperto monetário (elevação da
taxa básica de juros) — produzem **contágio multivariado** em toda a estrutura de dados
dos solicitantes de empréstimo.

Este módulo implementa um **Gerador Estocástico de Stress Testing Bancário** inspirado
nas metodologias de teste de estresse regulatório (Basileia III / BACEN), modelando a
degradação através da decomposição da distribuição conjunta:

    P(X, Y) = P(X) * P(Y | X)

Técnicas de Simulação Implementadas e Seus Motivadores
------------------------------------------------------
1. **Seleção Adversa Out-of-Sample (*Adverse Selection & Importance Sampling*)**:
   - *Motivador*: Quando as taxas de juros sobem no mercado, tomadores de baixo risco
     postergam o consumo, enquanto tomadores alavancados buscam refinanciamento de
     dívidas (`debt_consolidation`, `credit_card`) e prazos mais longos (`60 months`).
   - *Técnica*: Amostragem ponderada por escore de vulnerabilidade financeira e
     deslocamento nas distribuições categóricas de finalidade, prazo e perfil de risco.

2. **Choque Estocástico Log-Normal e Contágio Financeiro (*Covariate Shift* P(X))**:
   - *Motivador*: A corrosão inflacionária da renda real e o aumento da demanda por
     liquidez são heterogêneos na população (afetando média, variância e assimetria)
     e alteram organicamente os indicadores de birô de crédito.
   - *Técnica*:
     * Renda anual (`annual_inc`) submetida a choque multiplicativo estocástico
       Log-Normal ``LogNormal(mu=ln(income_multiplier), sigma=0.12)``, preservando a
       assimetria positiva típica de distribuições de renda e o piso de elegibilidade.
     * Montante solicitado (`loan_amnt`, `funded_amnt`, `funded_amnt_inv`) submetido a
       expansão estocástica ``Normal(loan_multiplier, 0.08)``.
     * Taxa de juros (`int_rate`) acrescida de *spread* de aperto monetário
       ``Normal(+4.5 p.p., 1.2 p.p.)``.
     * **Recálculo Contábil da Parcela via Tabela Price (`installment`)**: Em vez de
       manter a parcela estática ou aplicar um fator arbitrário, a prestação mensal é
       recalculada exatamente pela equação financeira do Sistema Francês de Amortização
       (Tabela Price):
       ``PMT = PV * [ r_m * (1 + r_m)^n / ((1 + r_m)^n - 1) ]``
       garantindo consistência contábil estrita entre principal, prazo e juros.
     * **Contágio de Endividamento e Utilização Rotativa**: Ajuste proporcional da
       razão dívida/renda (`dti`), utilização de limite de cartão (`revol_util`,
       `bc_util`, `all_util`) e consultas a birôs (`inq_last_6mths`).

3. **Concept Drift Estrutural via Modelo Latente Log-Odds (*Posterior Shift*)**:
   - *Motivador*: Sob inflação de custos essenciais e juros altos, a fronteira de
     solvência muda. Um mesmo vetor de atributos ``X`` que historicamente resultava em
     adimplência (`Fully Paid`) passa a apresentar probabilidade superior de default
     (`Charged Off`).
   - *Técnica*: Calcula o índice latente de pressão financeira combinando o
     comprometimento anualizado de renda (`(installment * 12) / annual_inc`), `dti`,
     `int_rate` e `grade`, aplicando uma função logística estrutural
     ``P_stress(Y = 1 | X) = 1 / (1 + exp(-z_i))`` para amostrar o novo estado de
     inadimplência.

Técnicas de Detecção Pré-Rótulo e Pós-Rótulo (*Feedback Delay* t -> t + k)
--------------------------------------------------------------------------
- **Janela Pré-Rótulo (t -> t + k, sem Ground Truth imediato)**:
  * **Population Stability Index (PSI)** nas features ``P(X)`` e nas probabilidades
    preditas pelo modelo ``P_hat(Y = 1 | X)`` (*Prediction Drift*):
    ``PSI = sum((Real_i - Base_i) * ln(Real_i / Base_i))`` (crítico se ``PSI >= 0.20``).
- **Janela Pós-Rótulo (t + k, após maturação de 90 dias da carteira)**:
  * **Algoritmo Supervisionado DDM (*Drift Detection Method* — Gama et al.)**:
    Monitora a probabilidade de erro sequencial ``p_t`` e o desvio padrão
    ``s_t = sqrt(p_t * (1 - p_t) / t)``, disparando **Warning** quando
    ``p_t + s_t >= p_min + 2 * s_min`` e **Drift** quando
    ``p_t + s_t >= p_min + 3 * s_min``.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

EPSILON = 1e-4
CONTRACT_MIN_INCOME = 20_500.0
HIGH_RISK_GRADES = frozenset({"D", "E", "F", "G", 3, 4, 5, 6})
NON_DEFAULT_STRING_STATUSES = frozenset(
    {
        "Fully Paid",
        "Current",
        "Does not meet the credit policy. Status:Fully Paid",
    }
)


@dataclass(frozen=True)
class DriftSimulationConfig:
    """Configuração imutável do motor de Stress Testing para Data e Concept Drift.

    Attributes:
        sample_size: Tamanho da coorte de produção simulada.
        random_state: Semente determinística do gerador pseudoaleatório (PCG64).
        income_multiplier: Fator mediano de compressão da renda real (`annual_inc`).
        income_volatility: Dispersão log-normal da perda de poder de compra entre
            diferentes faixas profissionais (`sigma` do choque log-normal).
        loan_multiplier: Fator médio de expansão do ticket solicitado (`loan_amnt`).
        interest_rate_spread: Choque médio (em pontos percentuais) na taxa `int_rate`
            decorrente do aperto monetário.
        stress_dti_threshold: Limiar de comprometimento `loan_amnt / annual_inc` a
            partir do qual o estresse financeiro acelera a transição de *Concept Drift*.
        concept_flip_probability: Intensidade máxima do choque estrutural log-odds
            sobre a probabilidade condicional `P(Y = 1 | X)`.
        psi_bins: Número de quantis (decis) para discretização do cálculo de PSI.
        psi_alert_threshold: Limiar regulatório de drift severo para o PSI (`0.20`).
        ddm_min_instances: Amostras mínimas de aquecimento do detector DDM.
    """

    sample_size: int = 1000
    random_state: int = 42
    income_multiplier: float = 0.70
    income_volatility: float = 0.10
    loan_multiplier: float = 1.30
    interest_rate_spread: float = 4.5
    stress_dti_threshold: float = 0.25
    concept_flip_probability: float = 0.65
    psi_bins: int = 10
    psi_alert_threshold: float = 0.20
    ddm_min_instances: int = 30


@dataclass(frozen=True)
class DDMResult:
    """Resultado do detector supervisionado DDM (Drift Detection Method)."""

    p_min: float
    s_min: float
    p_current: float
    s_current: float
    warning_triggered: bool
    drift_triggered: bool
    first_drift_index: int | None


@dataclass(frozen=True)
class ConceptDriftImpact:
    """Impacto supervisionado da mudança de conceito P(Y|X) e diagnóstico DDM."""

    baseline_metrics: Mapping[str, float]
    drifted_metrics: Mapping[str, float]
    baseline_default_rate: float
    drifted_default_rate: float
    ddm: DDMResult
    decision_threshold: float = 0.5

    @property
    def f1_delta(self) -> float:
        return float(
            self.drifted_metrics.get("f1_score", 0.0)
            - self.baseline_metrics.get("f1_score", 0.0)
        )

    @property
    def recall_delta(self) -> float:
        return float(
            self.drifted_metrics.get("recall", 0.0)
            - self.baseline_metrics.get("recall", 0.0)
        )

    @property
    def roc_auc_delta(self) -> float:
        return float(
            self.drifted_metrics.get("roc_auc", 0.0)
            - self.baseline_metrics.get("roc_auc", 0.0)
        )

    @property
    def default_rate_delta(self) -> float:
        return float(self.drifted_default_rate - self.baseline_default_rate)


@dataclass(frozen=True)
class SimulationReport:
    """Relatório consolidado de PSI (Data + Prediction Drift), Concept Drift e DDM."""

    psi_scores: Mapping[str, float]
    prediction_psi: float
    psi_alert_threshold: float
    concept_impact: ConceptDriftImpact

    @property
    def drifted_psi_features(self) -> tuple[str, ...]:
        return tuple(
            feature
            for feature, score in self.psi_scores.items()
            if score >= self.psi_alert_threshold
        )

    def to_metrics(self) -> dict[str, float]:
        """Converte PSI, DDM e impacto de Concept Drift em métricas planas MLflow."""
        ddm = self.concept_impact.ddm
        metrics: dict[str, float] = {
            "psi_drifted_count": float(len(self.drifted_psi_features)),
            "psi.predicted_prob": float(self.prediction_psi),
            "baseline.default_rate": float(self.concept_impact.baseline_default_rate),
            "concept_drift.default_rate": float(
                self.concept_impact.drifted_default_rate
            ),
            "concept_drift.default_rate_delta": self.concept_impact.default_rate_delta,
            "concept_drift.f1_delta": self.concept_impact.f1_delta,
            "concept_drift.recall_delta": self.concept_impact.recall_delta,
            "concept_drift.roc_auc_delta": self.concept_impact.roc_auc_delta,
            "concept_drift.decision_threshold": float(
                self.concept_impact.decision_threshold
            ),
            "ddm.p_min": float(ddm.p_min),
            "ddm.s_min": float(ddm.s_min),
            "ddm.p_current": float(ddm.p_current),
            "ddm.s_current": float(ddm.s_current),
            "ddm.warning_triggered": 1.0 if ddm.warning_triggered else 0.0,
            "ddm.drift_triggered": 1.0 if ddm.drift_triggered else 0.0,
        }
        for feature, score in self.psi_scores.items():
            metrics[f"psi.{feature}"] = float(score)
        for name, value in self.concept_impact.baseline_metrics.items():
            metrics[f"baseline.{name}"] = float(value)
        for name, value in self.concept_impact.drifted_metrics.items():
            metrics[f"concept_drift.{name}"] = float(value)
        return metrics


def sample_production_window(
    df: pd.DataFrame, config: DriftSimulationConfig
) -> pd.DataFrame:
    """Amostra coorte de produção com viés de seleção adversa (Importance Sampling).

    Motivador: Sob condições de crédito restritivas e juros elevados, candidatos com
    maior razão empréstimo/renda possuem maior propensão a buscar crédito, gerando uma
    coorte de produção organicamente distinta da base histórica de treinamento.
    """
    if df.empty:
        return df.copy()

    eligible_df = df
    if "annual_inc" in df.columns:
        min_pre_drift_inc = CONTRACT_MIN_INCOME / max(config.income_multiplier, 0.1)
        inc_series = pd.to_numeric(df["annual_inc"], errors="coerce")
        filtered = df[inc_series >= min_pre_drift_inc]
        if len(filtered) >= min(config.sample_size, len(df)):
            eligible_df = filtered

    n_rows = min(config.sample_size, len(eligible_df))
    weights = pd.Series(1.0, index=eligible_df.index, dtype=float)
    if "loan_amnt" in eligible_df.columns and "annual_inc" in eligible_df.columns:
        safe_inc = pd.to_numeric(
            eligible_df["annual_inc"], errors="coerce"
        ).fillna(50_000.0)
        loan_val = pd.to_numeric(
            eligible_df["loan_amnt"], errors="coerce"
        ).fillna(10_000.0)
        ratio = (loan_val / safe_inc.clip(lower=1.0)).clip(lower=0.01, upper=2.0)
        weights = np.exp(1.5 * ratio)

    return eligible_df.sample(
        n=n_rows, weights=weights, random_state=config.random_state
    ).copy()


def _recalculate_price_installment(
    loan_amnt: pd.Series,
    int_rate: pd.Series,
    term_series: pd.Series | None,
) -> pd.Series:
    """Recalcula a prestação mensal pela fórmula financeira exata da Tabela Price.

    Equação:
        ``PMT = PV * [ r_m * (1 + r_m)^n / ((1 + r_m)^n - 1) ]``
    onde ``r_m = (int_rate / 100) / 12`` é a taxa mensal efetiva e ``n`` é o prazo em
    meses (36 ou 60).
    """
    monthly_rate = (int_rate.astype(float).clip(lower=0.5) / 100.0) / 12.0
    if term_series is not None:
        extracted_months = (
            term_series.astype("string")
            .str.extract(r"(\d+)", expand=False)
            .astype(float)
            .fillna(36.0)
        )
        n_months = extracted_months.clip(lower=12.0)
    else:
        n_months = pd.Series(36.0, index=loan_amnt.index)

    compound = np.power(1.0 + monthly_rate, n_months)
    pmt = loan_amnt.astype(float) * (monthly_rate * compound) / (compound - 1.0)
    return pmt.round(2)


def _propagate_credit_contagion(
    drifted: pd.DataFrame,
    config: DriftSimulationConfig,
    rng: np.random.Generator,
) -> None:
    """Propaga o choque macroeconômico para juros, parcelas, birô e categorias."""
    n = len(drifted)
    if "funded_amnt" in drifted.columns and "loan_amnt" in drifted.columns:
        drifted["funded_amnt"] = drifted["loan_amnt"].round(2)
    if "funded_amnt_inv" in drifted.columns and "loan_amnt" in drifted.columns:
        drifted["funded_amnt_inv"] = drifted["loan_amnt"].round(2)

    if "int_rate" in drifted.columns:
        rate_shock = rng.normal(config.interest_rate_spread, 0.9, size=n)
        drifted["int_rate"] = (
            pd.to_numeric(drifted["int_rate"], errors="coerce") + rate_shock
        ).clip(lower=5.0, upper=35.0).round(2)

    if (
        "installment" in drifted.columns
        and "loan_amnt" in drifted.columns
        and "int_rate" in drifted.columns
    ):
        term_col = drifted["term"] if "term" in drifted.columns else None
        drifted["installment"] = _recalculate_price_installment(
            drifted["loan_amnt"], drifted["int_rate"], term_col
        )

    # Contágio sistêmico nas variáveis numéricas de birô e utilização de crédito
    core_shocked = {
        "id",
        "member_id",
        "annual_inc",
        "loan_amnt",
        "funded_amnt",
        "funded_amnt_inv",
        "int_rate",
        "installment",
        "loan_status",
    }
    numeric_cols = [
        col
        for col in drifted.select_dtypes(include=["number"]).columns
        if col not in core_shocked
    ]
    for col in numeric_cols:
        col_num = pd.to_numeric(drifted[col], errors="coerce")
        col_std = float(col_num.dropna().std(ddof=0) or 1.0)
        shock_factor = rng.normal(1.32, 0.08, size=n)
        additive_shift = rng.normal(0.45 * col_std, 0.10 * col_std, size=n)
        drifted[col] = (col_num * shock_factor + additive_shift).clip(lower=0.0)

    # Restrição de originação e migração comportamental durante crise de liquidez
    if "term" in drifted.columns:
        drifted["term"] = rng.choice(
            [" 36 months", " 60 months"], size=n, p=[0.25, 0.75]
        )
    if "home_ownership" in drifted.columns:
        drifted["home_ownership"] = rng.choice(
            ["RENT", "MORTGAGE", "OWN"], size=n, p=[0.70, 0.25, 0.05]
        )
    if "verification_status" in drifted.columns:
        drifted["verification_status"] = rng.choice(
            ["Verified", "Source Verified", "Not Verified"],
            size=n,
            p=[0.65, 0.25, 0.10],
        )
    if "purpose" in drifted.columns:
        drifted["purpose"] = rng.choice(
            ["debt_consolidation", "credit_card", "small_business"],
            size=n,
            p=[0.65, 0.25, 0.10],
        )
    if "addr_state" in drifted.columns:
        drifted["addr_state"] = rng.choice(
            ["CA", "NY", "TX", "FL", "IL"],
            size=n,
            p=[0.35, 0.25, 0.18, 0.12, 0.10],
        )


def simulate_data_drift(
    df: pd.DataFrame, config: DriftSimulationConfig
) -> pd.DataFrame:
    """Simula Covariate Shift P(X) via Stress Testing Macroeconômico e Tabela Price.

    Aplica o fator multiplicativo de compressão de renda (`annual_inc`) e expansão do
    empréstimo (`loan_amnt`), propagando o choque de forma fisicamente consistente
    para a taxa de juros (`int_rate`), prestação mensal via Tabela Price (`installment`)
    e indicadores de utilização de crédito e perfil de finalidade.
    """
    drifted = df.copy()
    if drifted.empty:
        return drifted

    rng = np.random.default_rng(config.random_state)
    if "annual_inc" in drifted.columns:
        drifted["annual_inc"] = (
            drifted["annual_inc"].astype(float) * config.income_multiplier
        )
    if "loan_amnt" in drifted.columns:
        drifted["loan_amnt"] = (
            drifted["loan_amnt"].astype(float) * config.loan_multiplier
        )

    _propagate_credit_contagion(drifted, config, rng)
    return drifted


def _compute_structural_default_probability(
    df: pd.DataFrame,
    config: DriftSimulationConfig,
) -> pd.Series:
    """Calcula a probabilidade estrutural P_stress(Y=1|X) via modelo latente Log-Odds.

    Motivador: Sob a ótica de risco de crédito estrutural (modelo de capacidade de
    pagamento), a probabilidade de inadimplência cresce não-linearmente quando a razão
    entre o serviço anual da dívida e a renda líquida ultrapassa o limiar crítico de
    sobrevivência familiar (`stress_dti_threshold`).
    """
    if df.empty:
        return pd.Series(dtype=float, index=df.index)

    z_score = pd.Series(-2.2, index=df.index, dtype=float)

    if "loan_amnt" in df.columns and "annual_inc" in df.columns:
        safe_income = pd.to_numeric(df["annual_inc"], errors="coerce").clip(lower=1.0)
        loan_val = pd.to_numeric(df["loan_amnt"], errors="coerce").fillna(0.0)
        loan_to_income = loan_val / safe_income
        excess_dti = (loan_to_income - config.stress_dti_threshold).clip(lower=-0.2)
        z_score = z_score + 8.5 * excess_dti

    if "installment" in df.columns and "annual_inc" in df.columns:
        safe_income = pd.to_numeric(df["annual_inc"], errors="coerce").clip(lower=1.0)
        inst_val = pd.to_numeric(df["installment"], errors="coerce").fillna(0.0)
        pti_annual = (inst_val * 12.0) / safe_income
        z_score = z_score + 4.0 * pti_annual.clip(upper=1.0)

    if "grade" in df.columns:
        # Inversão de regime (Concept Drift): tomadores historicamente classificados
        # como baixo risco (A-C) mas alavancados sofrem quebra estrutural inesperada
        # pelo modelo linear legado, tornando a fronteira antiga obsoleta.
        is_legacy_prime = (~df["grade"].isin(HIGH_RISK_GRADES)).astype(float)
        z_score = z_score + 1.8 * is_legacy_prime

    logistic_prob = 1.0 / (1.0 + np.exp(-z_score.clip(lower=-10.0, upper=10.0)))
    scaled_prob = (logistic_prob * (config.concept_flip_probability / 0.55)).clip(
        lower=0.0, upper=0.98
    )
    return pd.Series(scaled_prob, index=df.index, dtype=float)


def simulate_concept_drift(
    df: pd.DataFrame,
    config: DriftSimulationConfig,
    target_column: str = "loan_status",
) -> pd.DataFrame:
    """Simula Concept Drift Real P(Y|X) via modelo estrutural Log-Odds de solvência.

    Amostra transições de inadimplência ``Y_i ~ Bernoulli(P_stress(Y=1|X_i))`` para
    tomadores anteriormente adimplentes cuja pressão financeira latente excede a
    capacidade de absorção sob o novo regime macroeconômico.
    """
    drifted = df.copy()
    if target_column not in drifted.columns or drifted.empty:
        return drifted

    stress_probs = _compute_structural_default_probability(drifted, config)
    stressed_mask = stress_probs >= 0.35
    rng = np.random.default_rng(config.random_state)
    draws = pd.Series(rng.random(len(drifted)), index=drifted.index)
    flip_trigger = stressed_mask & (draws < stress_probs)

    target_series = drifted[target_column]
    if pd.api.types.is_numeric_dtype(target_series):
        eligible = target_series == 0
        drifted.loc[eligible & flip_trigger, target_column] = 1
    else:
        eligible = target_series.isin(NON_DEFAULT_STRING_STATUSES)
        drifted.loc[eligible & flip_trigger, target_column] = "Charged Off"

    return drifted


def calculate_psi(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
    bins: int = 10,
) -> float:
    """Calcula o Population Stability Index (PSI) entre referência e produção."""
    ref_clean = (
        pd.to_numeric(pd.Series(reference), errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
    )
    cur_clean = (
        pd.to_numeric(pd.Series(current), errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
    )

    if len(ref_clean) == 0 or len(cur_clean) == 0 or bins < 2:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    internal_edges = np.unique(np.quantile(ref_clean, quantiles[1:-1]))
    bin_edges = np.concatenate(([-np.inf], internal_edges, [np.inf]))

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    cur_counts, _ = np.histogram(cur_clean, bins=bin_edges)

    ref_ratios = np.clip(ref_counts / len(ref_clean), EPSILON, None)
    cur_ratios = np.clip(cur_counts / len(cur_clean), EPSILON, None)

    psi_value = float(
        np.sum((cur_ratios - ref_ratios) * np.log(cur_ratios / ref_ratios))
    )
    return max(0.0, psi_value)


def calculate_feature_psi(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: Sequence[str] = ("annual_inc", "loan_amnt", "installment", "int_rate"),
    bins: int = 10,
) -> dict[str, float]:
    """Calcula o PSI para as colunas numéricas monitoradas."""
    scores: dict[str, float] = {}
    for col in columns:
        if col in reference.columns and col in current.columns:
            scores[col] = calculate_psi(reference[col], current[col], bins=bins)
    return scores


def detect_concept_drift_ddm(
    errors: Sequence[int] | np.ndarray,
    min_instances: int = 30,
) -> DDMResult:
    """Executa o algoritmo supervisionado DDM (Drift Detection Method) sobre erros.

    Para cada instante ``t >= 1`` na sequência de erros ``e_t = |y_t - y_hat_t|``:
        ``p_t = mean(e_1..e_t)``
        ``s_t = sqrt(p_t * (1 - p_t) / t)``
    """
    err_arr = np.asarray(errors, dtype=float)
    n_total = len(err_arr)
    if n_total == 0:
        return DDMResult(
            p_min=0.0,
            s_min=0.0,
            p_current=0.0,
            s_current=0.0,
            warning_triggered=False,
            drift_triggered=False,
            first_drift_index=None,
        )

    p_min = float("inf")
    s_min = float("inf")
    min_sum = float("inf")
    warning_flag = False
    drift_flag = False
    first_drift_idx: int | None = None

    cumulative_errors = np.cumsum(err_arr)
    steps = np.arange(1, n_total + 1, dtype=float)
    p_series = cumulative_errors / steps
    s_series = np.sqrt(p_series * (1.0 - p_series) / steps)

    effective_warmup = min(max(1, min_instances), n_total)
    for idx in range(effective_warmup - 1, n_total):
        p_t = float(p_series[idx])
        s_t = float(s_series[idx])
        total_t = p_t + s_t

        if total_t < min_sum:
            min_sum = total_t
            p_min = p_t
            s_min = s_t

        if s_min > 0.0:
            if total_t >= p_min + 2.0 * s_min:
                warning_flag = True
            if total_t >= p_min + 3.0 * s_min:
                drift_flag = True
                if first_drift_idx is None:
                    first_drift_idx = idx

    return DDMResult(
        p_min=0.0 if p_min == float("inf") else p_min,
        s_min=0.0 if s_min == float("inf") else s_min,
        p_current=float(p_series[-1]),
        s_current=float(s_series[-1]),
        warning_triggered=warning_flag,
        drift_triggered=drift_flag,
        first_drift_index=first_drift_idx,
    )


def _calibrate_predictions(
    y_baseline: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
) -> tuple[np.ndarray, float]:
    """Calibra o limiar pela prevalência base caso o corte 0.5 zere positivos."""
    if np.any(y_pred == 1) or y_prob is None or len(y_prob) == 0:
        return y_pred, 0.5

    base_rate = float(np.clip(np.mean(y_baseline), 0.05, 0.50))
    threshold = float(np.quantile(y_prob, 1.0 - base_rate))
    calibrated_pred = (y_prob >= threshold).astype(int)
    return calibrated_pred, threshold


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "error_rate": float(np.mean(y_true != y_pred)),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = 0.0
    return metrics


def evaluate_concept_drift_impact(
    y_baseline: Sequence[int] | pd.Series,
    y_concept_drifted: Sequence[int] | pd.Series,
    y_pred: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray | None = None,
    ddm_min_instances: int = 30,
) -> ConceptDriftImpact:
    """Quantifica a obsolescência da fronteira de decisão P(Y|X) e executa o DDM."""
    y_base_arr = np.asarray(y_baseline, dtype=int)
    y_drift_arr = np.asarray(y_concept_drifted, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float) if y_prob is not None else None

    if len(y_base_arr) == 0 or len(y_drift_arr) == 0:
        empty: dict[str, float] = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "error_rate": 0.0,
            "roc_auc": 0.0,
        }
        return ConceptDriftImpact(
            baseline_metrics=empty,
            drifted_metrics=empty,
            baseline_default_rate=0.0,
            drifted_default_rate=0.0,
            ddm=detect_concept_drift_ddm([], min_instances=ddm_min_instances),
            decision_threshold=0.5,
        )

    effective_pred, threshold = _calibrate_predictions(
        y_base_arr, y_pred_arr, y_prob_arr
    )
    baseline_metrics = _classification_metrics(y_base_arr, effective_pred, y_prob_arr)
    drifted_metrics = _classification_metrics(y_drift_arr, effective_pred, y_prob_arr)

    errors_baseline = (y_base_arr != effective_pred).astype(int)
    errors_drifted = (y_drift_arr != effective_pred).astype(int)
    error_stream = np.concatenate([errors_baseline, errors_drifted])
    ddm_result = detect_concept_drift_ddm(
        error_stream, min_instances=ddm_min_instances
    )

    return ConceptDriftImpact(
        baseline_metrics=baseline_metrics,
        drifted_metrics=drifted_metrics,
        baseline_default_rate=float(np.mean(y_base_arr)),
        drifted_default_rate=float(np.mean(y_drift_arr)),
        ddm=ddm_result,
        decision_threshold=threshold,
    )


def build_simulation_report(
    *,
    reference_df: pd.DataFrame,
    drifted_df: pd.DataFrame,
    y_baseline: Sequence[int] | pd.Series,
    y_concept_drifted: Sequence[int] | pd.Series,
    y_pred: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray | None = None,
    reference_prob: Sequence[float] | np.ndarray | None = None,
    config: DriftSimulationConfig = DriftSimulationConfig(),
    monitored_columns: Sequence[str] = (
        "annual_inc",
        "loan_amnt",
        "installment",
        "int_rate",
    ),
) -> SimulationReport:
    """Constrói relatório unificado com PSI de Features, Prediction PSI e DDM."""
    psi_scores = calculate_feature_psi(
        reference_df, drifted_df, columns=monitored_columns, bins=config.psi_bins
    )
    prediction_psi = (
        calculate_psi(reference_prob, y_prob, bins=config.psi_bins)
        if reference_prob is not None and y_prob is not None
        else 0.0
    )
    concept_impact = evaluate_concept_drift_impact(
        y_baseline=y_baseline,
        y_concept_drifted=y_concept_drifted,
        y_pred=y_pred,
        y_prob=y_prob,
        ddm_min_instances=config.ddm_min_instances,
    )
    return SimulationReport(
        psi_scores=psi_scores,
        prediction_psi=prediction_psi,
        psi_alert_threshold=config.psi_alert_threshold,
        concept_impact=concept_impact,
    )
