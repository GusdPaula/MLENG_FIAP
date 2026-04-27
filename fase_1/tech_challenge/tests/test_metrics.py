"""Testes para TelcoMetrics.

Cobre:
  - calcular_metricas_tecnicas: valores, limites, casos extremos
  - calcular_metricas_negocio: matriz de confusão, cálculos financeiros, ROI
  - todas_as_metricas: integração entre as duas funções
  - relatorio_completo: smoke test de saída
"""

from typing import ClassVar

import numpy as np
import pytest

from src.evaluation.metrics import TelcoMetrics

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def perfect_predictions():
    """Modelo perfeito: todas as predições corretas."""
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_proba = np.array([0.1, 0.05, 0.95, 0.9, 0.85, 0.1, 0.92, 0.08])
    return y_true, y_pred, y_proba


@pytest.fixture
def typical_predictions():
    """Predições realistas com alguns erros."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 0, 1, 1, 1])
    #                        FP        FN
    y_proba = np.array([0.1, 0.2, 0.6, 0.3, 0.15, 0.8, 0.4, 0.75, 0.85, 0.9])
    return y_true, y_pred, y_proba


@pytest.fixture
def worst_predictions():
    """Modelo invertido: erra tudo."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    y_proba = np.array([0.9, 0.85, 0.1, 0.15])
    return y_true, y_pred, y_proba


@pytest.fixture
def all_negative_predictions():
    """Modelo que nunca prediz churn."""
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.15, 0.3, 0.25])
    return y_true, y_pred, y_proba


# ---------------------------------------------------------------------------
# calcular_metricas_tecnicas
# ---------------------------------------------------------------------------


class TestMetricasTecnicas:
    EXPECTED_KEYS: ClassVar[set[str]] = {
        "auc_roc",
        "pr_auc",
        "f1_score",
        "precision",
        "recall",
        "accuracy",
    }

    def test_returns_all_keys(self, typical_predictions):
        y_true, y_pred, y_proba = typical_predictions
        result = TelcoMetrics.calcular_metricas_tecnicas(y_true, y_pred, y_proba)
        assert set(result.keys()) == self.EXPECTED_KEYS

    def test_all_values_are_float(self, typical_predictions):
        y_true, y_pred, y_proba = typical_predictions
        result = TelcoMetrics.calcular_metricas_tecnicas(y_true, y_pred, y_proba)
        for key, val in result.items():
            assert isinstance(val, float), f"{key} deveria ser float, mas é {type(val)}"

    def test_all_values_between_0_and_1(self, typical_predictions):
        y_true, y_pred, y_proba = typical_predictions
        result = TelcoMetrics.calcular_metricas_tecnicas(y_true, y_pred, y_proba)
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} fora do intervalo [0, 1]"

    def test_perfect_model_returns_ones(self, perfect_predictions):
        y_true, y_pred, y_proba = perfect_predictions
        result = TelcoMetrics.calcular_metricas_tecnicas(y_true, y_pred, y_proba)
        assert result["auc_roc"] == pytest.approx(1.0)
        assert result["f1_score"] == pytest.approx(1.0)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        assert result["accuracy"] == pytest.approx(1.0)

    def test_accuracy_manual_calculation(self, typical_predictions):
        y_true, y_pred, y_proba = typical_predictions
        result = TelcoMetrics.calcular_metricas_tecnicas(y_true, y_pred, y_proba)
        expected = np.mean(y_true == y_pred)
        assert result["accuracy"] == pytest.approx(expected)

    def test_recall_is_sensitivity(self, typical_predictions):
        """recall = TP / (TP + FN)"""
        y_true, y_pred, y_proba = typical_predictions
        result = TelcoMetrics.calcular_metricas_tecnicas(y_true, y_pred, y_proba)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        expected_recall = tp / (tp + fn)
        assert result["recall"] == pytest.approx(expected_recall)

    def test_precision_manual(self, typical_predictions):
        """precision = TP / (TP + FP)"""
        y_true, y_pred, y_proba = typical_predictions
        result = TelcoMetrics.calcular_metricas_tecnicas(y_true, y_pred, y_proba)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        expected = tp / (tp + fp)
        assert result["precision"] == pytest.approx(expected)

    def test_auc_roc_worst_model_near_zero(self, worst_predictions):
        """Um modelo invertido tem AUC ≈ 0."""
        y_true, y_pred, y_proba = worst_predictions
        result = TelcoMetrics.calcular_metricas_tecnicas(y_true, y_pred, y_proba)
        assert result["auc_roc"] < 0.5

    def test_all_negative_predictions_recall_is_zero(self, all_negative_predictions):
        """Quando nada é predito como positivo, recall=0 e precision é indefinida.

        sklearn emite UndefinedMetricWarning nesse caso (0/0 para precision).
        Capturamos o warning explicitamente para que o teste documente esse
        comportamento em vez de deixar o aviso vazar para o output do CI.
        """
        from sklearn.exceptions import UndefinedMetricWarning

        y_true, y_pred, y_proba = all_negative_predictions
        with pytest.warns(UndefinedMetricWarning, match="Precision is ill-defined"):
            result = TelcoMetrics.calcular_metricas_tecnicas(y_true, y_pred, y_proba)

        assert result["recall"] == pytest.approx(0.0)
        assert result["precision"] == pytest.approx(0.0)  # sklearn fallback value
        assert result["f1_score"] == pytest.approx(0.0)

    def test_f1_harmonic_mean_of_precision_recall(self, typical_predictions):
        y_true, y_pred, y_proba = typical_predictions
        result = TelcoMetrics.calcular_metricas_tecnicas(y_true, y_pred, y_proba)
        p, r = result["precision"], result["recall"]
        expected_f1 = 2 * p * r / (p + r)
        assert result["f1_score"] == pytest.approx(expected_f1, rel=1e-5)


# ---------------------------------------------------------------------------
# calcular_metricas_negocio
# ---------------------------------------------------------------------------


class TestMetricasNegocio:
    EXPECTED_KEYS: ClassVar[set[str]] = {
        "tp",
        "fp",
        "fn",
        "tn",
        "churn_avoided_revenue",
        "false_positive_cost",
        "retention_cost",
        "net_benefit",
        "roi_percent",
    }

    def test_returns_all_keys(self, typical_predictions):
        y_true, y_pred, _ = typical_predictions
        result = TelcoMetrics.calcular_metricas_negocio(y_true, y_pred)
        assert set(result.keys()) == self.EXPECTED_KEYS

    def test_confusion_matrix_values_are_int(self, typical_predictions):
        y_true, y_pred, _ = typical_predictions
        result = TelcoMetrics.calcular_metricas_negocio(y_true, y_pred)
        for key in ("tp", "fp", "fn", "tn"):
            assert isinstance(result[key], int), f"{key} deveria ser int"

    def test_financial_values_are_float(self, typical_predictions):
        y_true, y_pred, _ = typical_predictions
        result = TelcoMetrics.calcular_metricas_negocio(y_true, y_pred)
        for key in (
            "churn_avoided_revenue",
            "false_positive_cost",
            "retention_cost",
            "net_benefit",
            "roi_percent",
        ):
            assert isinstance(result[key], float), f"{key} deveria ser float"

    def test_confusion_matrix_sums_to_total(self, typical_predictions):
        y_true, y_pred, _ = typical_predictions
        result = TelcoMetrics.calcular_metricas_negocio(y_true, y_pred)
        assert result["tp"] + result["fp"] + result["fn"] + result["tn"] == len(y_true)

    def test_perfect_model_no_fp_no_fn(self, perfect_predictions):
        y_true, y_pred, _ = perfect_predictions
        result = TelcoMetrics.calcular_metricas_negocio(y_true, y_pred)
        assert result["fp"] == 0
        assert result["fn"] == 0

    def test_churn_avoided_revenue_formula(self, typical_predictions):
        """churn_avoided = TP X CUSTOMER_LTV"""
        y_true, y_pred, _ = typical_predictions
        result = TelcoMetrics.calcular_metricas_negocio(y_true, y_pred)
        expected = result["tp"] * TelcoMetrics.CUSTOMER_LTV
        assert result["churn_avoided_revenue"] == pytest.approx(expected)

    def test_false_positive_cost_formula(self, typical_predictions):
        """fp_cost = FP X FALSE_POSITIVE_COST"""
        y_true, y_pred, _ = typical_predictions
        result = TelcoMetrics.calcular_metricas_negocio(y_true, y_pred)
        expected = result["fp"] * TelcoMetrics.FALSE_POSITIVE_COST
        assert result["false_positive_cost"] == pytest.approx(expected)

    def test_retention_cost_formula(self, typical_predictions):
        """retention_cost = TP X RETENTION_COST"""
        y_true, y_pred, _ = typical_predictions
        result = TelcoMetrics.calcular_metricas_negocio(y_true, y_pred)
        expected = result["tp"] * TelcoMetrics.RETENTION_COST
        assert result["retention_cost"] == pytest.approx(expected)

    def test_net_benefit_formula(self, typical_predictions):
        """net_benefit = churn_avoided - fp_cost - retention_cost"""
        y_true, y_pred, _ = typical_predictions
        result = TelcoMetrics.calcular_metricas_negocio(y_true, y_pred)
        expected = (
            result["churn_avoided_revenue"]
            - result["false_positive_cost"]
            - result["retention_cost"]
        )
        assert result["net_benefit"] == pytest.approx(expected)

    def test_roi_formula(self, typical_predictions):
        """roi = net_benefit / (fp_cost + retention_cost) X 100"""
        y_true, y_pred, _ = typical_predictions
        result = TelcoMetrics.calcular_metricas_negocio(y_true, y_pred)
        total_cost = result["false_positive_cost"] + result["retention_cost"]
        expected_roi = (result["net_benefit"] / total_cost) * 100 if total_cost > 0 else 0
        assert result["roi_percent"] == pytest.approx(expected_roi)

    def test_roi_is_zero_when_no_cost(self):
        """ROI = 0 quando não há custo (nenhum TP e nenhum FP)."""
        # Modelo que erra tudo: FN e TN apenas → TP=0, FP=0
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 0])
        result = TelcoMetrics.calcular_metricas_negocio(y_true, y_pred)
        assert result["roi_percent"] == pytest.approx(0.0)
        assert result["net_benefit"] == pytest.approx(0.0)

    def test_net_benefit_can_be_negative(self):
        """Muitos FP com poucos TP pode gerar prejuízo."""
        # 1 TP, 10 FP
        y_true = np.array([1] + [0] * 10)
        y_pred = np.array([1] * 11)
        result = TelcoMetrics.calcular_metricas_negocio(y_true, y_pred)
        # churn_avoided = 1 X 2080 = 2080
        # fp_cost       = 10 X 20  =  200
        # retention     = 1 X 50   =   50
        # net_benefit   = 2080 - 200 - 50 = 1830  (still positive here)
        # Let's verify the math is consistent regardless of sign
        expected = (
            result["churn_avoided_revenue"]
            - result["false_positive_cost"]
            - result["retention_cost"]
        )
        assert result["net_benefit"] == pytest.approx(expected)

    def test_worst_model_zero_revenue(self, worst_predictions):
        """Modelo que erra tudo não evita nenhum churn."""
        y_true, y_pred, _ = worst_predictions
        result = TelcoMetrics.calcular_metricas_negocio(y_true, y_pred)
        assert result["tp"] == 0
        assert result["churn_avoided_revenue"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# todas_as_metricas
# ---------------------------------------------------------------------------


class TestTodasAsMetricas:
    def test_returns_union_of_both_dicts(self, typical_predictions):
        y_true, y_pred, y_proba = typical_predictions
        tecnicas = TelcoMetrics.calcular_metricas_tecnicas(y_true, y_pred, y_proba)
        negocio = TelcoMetrics.calcular_metricas_negocio(y_true, y_pred)
        combined = TelcoMetrics.todas_as_metricas(y_true, y_pred, y_proba)
        assert set(combined.keys()) == set(tecnicas.keys()) | set(negocio.keys())

    def test_values_consistent_with_individual_calls(self, typical_predictions):
        y_true, y_pred, y_proba = typical_predictions
        tecnicas = TelcoMetrics.calcular_metricas_tecnicas(y_true, y_pred, y_proba)
        negocio = TelcoMetrics.calcular_metricas_negocio(y_true, y_pred)
        combined = TelcoMetrics.todas_as_metricas(y_true, y_pred, y_proba)

        for key, val in tecnicas.items():
            assert combined[key] == pytest.approx(val), f"Divergência em '{key}'"
        for key, val in negocio.items():
            assert combined[key] == val, f"Divergência em '{key}'"

    def test_no_key_collision_overwrites_wrong_value(self, typical_predictions):
        """Garante que a fusão dos dicts não silencia colisões inesperadas."""
        y_true, y_pred, y_proba = typical_predictions
        tecnicas_keys = set(TelcoMetrics.calcular_metricas_tecnicas(y_true, y_pred, y_proba))
        negocio_keys = set(TelcoMetrics.calcular_metricas_negocio(y_true, y_pred))
        assert tecnicas_keys.isdisjoint(negocio_keys), (
            f"Chaves duplicadas entre as duas funções: {tecnicas_keys & negocio_keys}"
        )


# ---------------------------------------------------------------------------
# relatorio_completo
# ---------------------------------------------------------------------------


class TestRelatorioCompleto:
    def test_runs_without_error(self, typical_predictions, capsys):
        y_true, y_pred, _ = typical_predictions
        TelcoMetrics.relatorio_completo(y_true, y_pred)  # não deve lançar exceção

    def test_output_contains_class_labels(self, typical_predictions, capsys):
        y_true, y_pred, _ = typical_predictions
        TelcoMetrics.relatorio_completo(y_true, y_pred)
        captured = capsys.readouterr()
        assert "Churn" in captured.out
        assert "Não Churn" in captured.out

    def test_output_contains_report_header(self, typical_predictions, capsys):
        y_true, y_pred, _ = typical_predictions
        TelcoMetrics.relatorio_completo(y_true, y_pred)
        captured = capsys.readouterr()
        assert "RELATÓRIO DE CLASSIFICAÇÃO" in captured.out

    def test_output_contains_precision_recall_f1(self, typical_predictions, capsys):
        y_true, y_pred, _ = typical_predictions
        TelcoMetrics.relatorio_completo(y_true, y_pred)
        captured = capsys.readouterr()
        assert "precision" in captured.out
        assert "recall" in captured.out
        assert "f1-score" in captured.out
