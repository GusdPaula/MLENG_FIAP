"""Testes unitários para modelos baseline e experimentos."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.baseline import BaselineExperiment, MLPWrapper

# -----------------------------
# Global warning filters
# -----------------------------
pytestmark = pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.UndefinedMetricWarning",
    "ignore::FutureWarning",
    "ignore::UserWarning:mlflow.*",
)


# -----------------------------
# Fixtures
# -----------------------------


@pytest.fixture
def sample_data():
    X = np.random.rand(50, 5)
    y = np.random.randint(0, 2, 50)
    return X, y


@pytest.fixture
def train_test_data():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)

    return (
        X[:80],
        X[80:],
        y[:80],
        y[80:],
    )


# -----------------------------
# MLPWrapper
# -----------------------------


class TestMLPWrapper:
    """Testes para MLPWrapper."""

    def test_build_network(self):
        model = MLPWrapper(input_size=5, hidden_sizes=[10, 5], dropout_rates=[0.1, 0.0])
        net = model._build_network()

        assert net is not None
        assert len(list(net.children())) > 0

    def test_fit_runs(self, sample_data):
        X, y = sample_data
        model = MLPWrapper(input_size=5, epochs=2)

        model.fit(X, y)

        assert model.model is not None

    def test_predict_output_shape(self, sample_data):
        X, y = sample_data
        model = MLPWrapper(input_size=5, epochs=2)

        model.fit(X, y)
        preds = model.predict(X)

        assert preds.shape[0] == X.shape[0]
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, sample_data):
        X, y = sample_data
        model = MLPWrapper(input_size=5, epochs=2)

        model.fit(X, y)
        proba = model.predict_proba(X)

        assert proba.shape == (X.shape[0], 2)

    def test_get_set_params(self):
        model = MLPWrapper(input_size=5)
        params = model.get_params()

        assert "input_size" in params

        model.set_params(input_size=10)
        assert model.input_size == 10


# -----------------------------
# BaselineExperiment
# -----------------------------


class TestBaselineExperiment:
    """Testes para BaselineExperiment."""

    @patch("mlflow.end_run")
    @patch("mlflow.active_run")
    @patch("mlflow.set_experiment")
    @patch("mlflow.create_experiment")
    @patch("mlflow.get_experiment_by_name")
    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.start_run")
    @patch("mlflow.log_param")
    @patch("mlflow.log_metrics")
    @patch("mlflow.log_dict")
    @patch("mlflow.log_artifact")
    @patch("mlflow.sklearn.log_model")
    @patch("mlflow.data.from_pandas")
    @patch("mlflow.log_input")
    def test_treinar_modelo_runs(
        self,
        mock_log_input,
        mock_from_pandas,
        mock_log_model,
        mock_log_artifact,
        mock_log_dict,
        mock_log_metrics,
        mock_log_param,
        mock_start_run,
        mock_set_tracking_uri,
        mock_get_exp,
        mock_create_exp,
        mock_set_exp,
        mock_active_run,
        mock_end_run,
        train_test_data,
        tmp_path,
    ):

        X_train, X_test, y_train, y_test = train_test_data

        mock_get_exp.return_value = None
        mock_from_pandas.return_value = MagicMock()
        mock_active_run.return_value = None

        exp = BaselineExperiment(mlflow_uri=str(tmp_path))

        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=100)

        trained_model, metrics = exp.treinar_modelo(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            nome_modelo="test_model",
        )

        assert trained_model is not None
        assert "test_auc_roc" in metrics
        assert "test_accuracy" in metrics

    @patch("mlflow.end_run")
    @patch("mlflow.active_run")
    @patch("mlflow.set_experiment")
    @patch("mlflow.create_experiment")
    @patch("mlflow.get_experiment_by_name")
    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.log_metrics")
    @patch("mlflow.log_dict")
    @patch("mlflow.log_artifact")
    @patch("mlflow.log_input")
    @patch("mlflow.sklearn.log_model")
    @patch("mlflow.start_run")
    def test_treinar_modelo_stores_results(
        self,
        mock_start_run,
        mock_log_model,
        mock_log_input,
        mock_log_artifact,
        mock_log_dict,
        mock_log_metrics,
        mock_set_tracking_uri,
        mock_get_exp,
        mock_create_exp,
        mock_set_exp,
        mock_active_run,
        mock_end_run,
        train_test_data,
        tmp_path,
    ):
        X_train, X_test, y_train, y_test = train_test_data

        # ✅ Fix experiment setup
        mock_get_exp.return_value = None
        mock_active_run.return_value = None

        # ✅ FIX: mock context manager
        mock_run = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = False

        exp = BaselineExperiment(mlflow_uri=str(tmp_path))

        from sklearn.dummy import DummyClassifier

        model = DummyClassifier()

        exp.treinar_modelo(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            nome_modelo="dummy",
        )

        # ✅ Assertions
        assert "dummy" in exp.resultados
        assert "dummy" in exp._modelos

        mock_log_metrics.assert_called()
        mock_log_model.assert_called()

    def test_comparar_baselines(self):
        exp = BaselineExperiment()

        resultados = {
            "model_a": {"test_auc_roc": 0.7, "test_accuracy": 0.6},
            "model_b": {"test_auc_roc": 0.8, "test_accuracy": 0.65},
        }

        df = exp.comparar_baselines(resultados)

        assert isinstance(df, pd.DataFrame)
        assert df.index[0] == "model_b"


# -----------------------------
# Controlled Pipeline
# -----------------------------


class TestControlledPipeline:
    """Testes para esteira controlada."""

    @patch("mlflow.end_run")
    @patch("mlflow.active_run")
    @patch("mlflow.set_experiment")
    @patch("mlflow.create_experiment")
    @patch("mlflow.get_experiment_by_name")
    @patch("mlflow.set_tracking_uri")
    @patch.object(BaselineExperiment, "treinar_modelo")
    def test_controlled_pipeline_runs(
        self,
        mock_train,
        mock_set_tracking_uri,
        mock_get_exp,
        mock_create_exp,
        mock_set_exp,
        mock_active_run,
        mock_end_run,
        train_test_data,
        tmp_path,
    ):
        X_train, X_test, y_train, y_test = train_test_data

        mock_get_exp.return_value = None
        mock_active_run.return_value = None

        exp = BaselineExperiment(mlflow_uri=str(tmp_path))

        # ✅ Side-effect mock (CRITICAL FIX)
        def mock_treinar(*args, **kwargs):
            nome_modelo = kwargs.get("nome_modelo", "mock_model")
            exp.resultados[nome_modelo] = {"test_auc_roc": 0.5}
            return None, {"test_auc_roc": 0.5}

        mock_train.side_effect = mock_treinar

        df, scaler = exp.treinar_esteira_controlada(
            X_train,
            X_test,
            y_train,
            y_test,
            aplicar_scaling=True,
            include_mlp=False,
            include_xgb_tuned=False,
        )

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert scaler is not None
