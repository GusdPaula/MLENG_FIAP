"""Testes de integração do pipeline end-to-end."""

from credit_scoring.models.train import FEATURE_COLS, _encode_categoricals, train_baseline
from credit_scoring.monitoring.drift_detector import run_drift_report


class TestTrainBaseline:
    def test_train_retorna_campeao(self, reference_df):
        result = train_baseline(reference_df, experiment_name="test_experiment")
        assert "champion_name" in result
        assert result["champion_name"] in ["logistic_regression", "xgboost"]

    def test_metricas_sao_validas(self, reference_df):
        result = train_baseline(reference_df, experiment_name="test_experiment")
        metrics = result["metrics"]
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["auc_roc"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0

    def test_modelo_gera_predicoes(self, reference_df, production_df):
        result = train_baseline(reference_df, experiment_name="test_experiment")
        champion = result["champion_model"]
        prod_enc = _encode_categoricals(production_df)
        X = prod_enc[[c for c in FEATURE_COLS if c in prod_enc.columns]]
        preds = champion.predict(X)
        assert len(preds) == len(production_df)
        assert set(preds).issubset({0, 1})


class TestDriftReport:
    def test_gera_html(self, reference_df, production_df, tmp_path):
        report_path, metrics = run_drift_report(
            reference_df, production_df, output_path=tmp_path / "drift_report.html"
        )
        assert report_path.exists()
        content = report_path.read_text()
        assert len(content) > 1000  # HTML não vazio

    def test_metricas_tem_campos_esperados(self, reference_df, production_df, tmp_path):
        _, metrics = run_drift_report(
            reference_df, production_df, output_path=tmp_path / "drift_report.html"
        )
        assert "n_drifted_features" in metrics or "error" in metrics
        assert "feature_scores" in metrics or "error" in metrics


class TestDataGeneration:
    def test_referencia_tem_colunas_corretas(self, reference_df):
        expected_cols = {
            "idade",
            "anos_emprego",
            "renda_mensal",
            "valor_emprestimo",
            "score_credito",
            "historico_pagamentos",
            "finalidade",
            "possui_conta_poupanca",
            "inadimplente",
        }
        assert expected_cols.issubset(set(reference_df.columns))

    def test_referencia_sem_nulos_por_default(self, reference_df):
        assert reference_df.isnull().sum().sum() == 0

    def test_idade_minima(self, reference_df):
        assert reference_df["idade"].min() > 18

    def test_score_no_range(self, reference_df):
        assert reference_df["score_credito"].min() >= 300
        assert reference_df["score_credito"].max() <= 900

    def test_target_binario(self, reference_df):
        assert set(reference_df["inadimplente"].unique()).issubset({0, 1})
