"""Testes para pipeline e modelos."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from src.models.pipeline import TelcoPipeline


class TestTelcoPipeline:
    """Testes para TelcoPipeline."""

    def test_create_logistic_regression(self):
        """Testa criação de pipeline LogisticRegression."""
        pipeline = TelcoPipeline()
        pipe = pipeline.create_logistic_regression()

        assert pipeline.pipeline is not None
        assert pipeline.model_name == 'LogisticRegression'
        assert 'scaler' in pipe.named_steps
        assert 'classifier' in pipe.named_steps

    def test_create_random_forest(self):
        """Testa criação de pipeline RandomForest."""
        pipeline = TelcoPipeline()
        pipe = pipeline.create_random_forest()

        assert pipeline.model_name == 'RandomForest'
        assert 'classifier' in pipe.named_steps

    def test_create_xgboost(self):
        """Testa criação de pipeline XGBoost."""
        pipeline = TelcoPipeline()
        pipe = pipeline.create_xgboost()

        assert pipeline.model_name == 'XGBoost'

    def test_train_predict_logistic(self, X_train_sample, y_train_sample, X_test_sample):
        """Testa treinamento e predição com LogisticRegression."""
        pipeline = TelcoPipeline()
        pipeline.create_logistic_regression()
        pipeline.train(X_train_sample, y_train_sample)

        predictions = pipeline.predict(X_test_sample)
        assert predictions.shape == (X_test_sample.shape[0],)
        assert np.all((predictions == 0) | (predictions == 1))

    def test_predict_proba(self, X_train_sample, y_train_sample, X_test_sample):
        """Testa probabilidades."""
        pipeline = TelcoPipeline()
        pipeline.create_logistic_regression()
        pipeline.train(X_train_sample, y_train_sample)

        proba = pipeline.predict_proba(X_test_sample)
        assert proba.shape == (X_test_sample.shape[0], 2)
        assert np.all(proba >= 0) and np.all(proba <= 1)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_save_load_pipeline(self, X_train_sample, y_train_sample, X_test_sample):
        """Testa salvamento e carregamento de pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_pipeline.pkl"

            # Treinar e salvar
            pipeline1 = TelcoPipeline(random_state=42)
            pipeline1.create_logistic_regression()
            pipeline1.train(X_train_sample, y_train_sample)
            pred1 = pipeline1.predict(X_test_sample)
            pipeline1.save(str(filepath))

            # Carregar e predizer
            pipeline2 = TelcoPipeline()
            pipeline2.load(str(filepath))
            pred2 = pipeline2.predict(X_test_sample)

            # Predições devem ser idênticas
            assert np.array_equal(pred1, pred2)

    def test_from_dict_config(self, X_train_sample, y_train_sample):
        """Testa criação de pipeline a partir de config dict."""
        config = {
            'model_type': 'logistic_regression',
            'class_weight': 'balanced',
            'random_state': 42
        }

        pipeline = TelcoPipeline.from_dict(config)
        pipeline.train(X_train_sample, y_train_sample)

        assert pipeline.model_name == 'LogisticRegression'

    def test_invalid_model_type(self):
        """Testa erro com tipo de modelo inválido."""
        config = {'model_type': 'invalid_model'}

        with pytest.raises(ValueError):
            TelcoPipeline.from_dict(config)

    def test_pipeline_not_trained_error(self, X_test_sample):
        """Testa erro ao predizer sem treinar."""
        pipeline = TelcoPipeline()
        pipeline.create_logistic_regression()

        with pytest.raises(ValueError):
            pipeline.predict(X_test_sample)


@pytest.mark.smoke
class TestSmokeTests:
    """Smoke tests - verificações básicas de integração."""

    def test_module_imports(self):
        """Testa que todos os módulos podem ser importados."""
        from src.config import get_config
        from src.data import TelcoDataLoader
        from src.models import TelcoPipeline, PredictionService
        from src.evaluation import TelcoMetrics

        assert get_config is not None
        assert TelcoDataLoader is not None
        assert TelcoPipeline is not None
        assert PredictionService is not None
        assert TelcoMetrics is not None

    def test_config_loads_without_error(self):
        """Testa que config carrega sem erro."""
        from src.config import get_config
        config = get_config('development')
        assert config is not None

    def test_pipeline_workflow(self, X_train_sample, y_train_sample, X_test_sample):
        """Testa workflow completo do pipeline."""
        from src.models.pipeline import TelcoPipeline

        # Criar, treinar, predizer
        pipeline = TelcoPipeline(random_state=42)
        pipeline.create_logistic_regression()
        pipeline.train(X_train_sample, y_train_sample)
        predictions = pipeline.predict(X_test_sample)

        assert predictions is not None
        assert len(predictions) == len(X_test_sample)

    def test_all_model_types_trainable(self, X_train_sample, y_train_sample):
        """Testa que todos os tipos de modelo podem ser treinados."""
        from src.models.pipeline import TelcoPipeline

        model_types = [
            ('logistic_regression', 'create_logistic_regression'),
            ('random_forest', 'create_random_forest'),
            ('xgboost', 'create_xgboost'),
        ]

        for model_name, create_method in model_types:
            pipeline = TelcoPipeline(random_state=42)
            getattr(pipeline, create_method)()

            # Deve treinar sem erro
            try:
                pipeline.train(X_train_sample, y_train_sample)
                assert pipeline.pipeline is not None
            except Exception as e:
                pytest.fail(f"{model_name} falhou ao treinar: {e}")
