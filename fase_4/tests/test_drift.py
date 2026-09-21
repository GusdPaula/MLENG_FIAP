"""Testa detecção de drift entre referência e produção."""

import numpy as np
from scipy import stats


class TestDataDrift:
    def test_renda_tem_drift_significativo(self, reference_df, production_df):
        """Renda deve ter distribuição estatisticamente diferente após drift."""
        stat, p_value = stats.ks_2samp(
            reference_df["renda_mensal"].values,
            production_df["renda_mensal"].values,
        )
        # p < 0.05 confirma que as distribuições são diferentes
        assert p_value < 0.05, f"KS test falhou: p={p_value:.4f} — drift não detectado em renda"

    def test_score_credito_tem_drift_significativo(self, reference_df, production_df):
        """Score de crédito deve ter distribuição estatisticamente diferente."""
        stat, p_value = stats.ks_2samp(
            reference_df["score_credito"].values,
            production_df["score_credito"].values,
        )
        assert p_value < 0.05, f"KS test falhou: p={p_value:.4f} — drift não detectado em score_credito"

    def test_renda_media_aumentou(self, reference_df, production_df):
        """A renda média de produção deve ser maior que a de referência (inflação simulada)."""
        assert production_df["renda_mensal"].mean() > reference_df["renda_mensal"].mean() * 1.3

    def test_score_medio_diminuiu(self, reference_df, production_df):
        """O score médio de produção deve ser menor que o de referência (crise simulada)."""
        assert production_df["score_credito"].mean() < reference_df["score_credito"].mean() * 0.9

    def test_inadimplencia_aumentou(self, reference_df, production_df):
        """Taxa de inadimplência deve ser maior na produção (concept drift)."""
        assert production_df["inadimplente"].mean() > reference_df["inadimplente"].mean()

    def test_referencia_nao_tem_drift_contra_si_mesma(self, reference_df):
        """Dataset de referência comparado consigo mesmo não deve ter drift."""
        stat, p_value = stats.ks_2samp(
            reference_df["renda_mensal"].values[:100],
            reference_df["renda_mensal"].values[100:200],
        )
        # p > 0.05 significa que as amostras vêm da mesma distribuição
        assert p_value > 0.05, "Falso positivo: referência detectada com drift contra si mesma"


class TestPSI:
    """Population Stability Index (PSI > 0.2 = drift significativo)."""

    @staticmethod
    def _psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        def _scale_range(arr, mn, mx):
            arr = np.clip(arr, mn, mx)
            return arr

        mn = min(expected.min(), actual.min())
        mx = max(expected.max(), actual.max())
        bins = np.linspace(mn, mx, buckets + 1)

        expected_perc, _ = np.histogram(expected, bins=bins)
        actual_perc, _ = np.histogram(actual, bins=bins)

        expected_perc = np.where(expected_perc == 0, 0.0001, expected_perc) / len(expected)
        actual_perc = np.where(actual_perc == 0, 0.0001, actual_perc) / len(actual)

        return float(np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc)))

    def test_psi_renda_acima_limiar(self, reference_df, production_df):
        psi = self._psi(reference_df["renda_mensal"].values, production_df["renda_mensal"].values)
        assert psi > 0.2, f"PSI renda={psi:.3f} — esperado > 0.2 para drift significativo"

    def test_psi_score_acima_limiar(self, reference_df, production_df):
        psi = self._psi(
            reference_df["score_credito"].values, production_df["score_credito"].values
        )
        assert psi > 0.2, f"PSI score={psi:.3f} — esperado > 0.2 para drift significativo"
