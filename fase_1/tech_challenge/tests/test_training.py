from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Ajuste o import conforme o nome do seu arquivo de treino
from src.models.model_training_pipeline import PipelineBuilder, main


@pytest.fixture
def sample_df():
    """Cria um DataFrame sintético maior para suportar o stratified split."""
    data = {
        "customerid": [str(i) for i in range(10)],
        "gender": ["Male", "Female"] * 5,
        "senior_citizen": ["No", "Yes"] * 5,
        "tenure_months": [1, 24, 12, 6, 12, 24, 1, 5, 10, 15],
        "monthly_charges": [20.0, 70.0, 50.0, 100.0, 30.0, 80.0, 25.0, 60.0, 45.0, 110.0],
        "Churn Value": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # 5 de cada classe
        "count": [1] * 10,
        "country": ["Brazil"] * 10,
        "state": ["SP"] * 10,
        "city": ["São Paulo"] * 10,
        "zip_code": [123] * 10,
        "lat_long": ["0,0"] * 10,
        "latitude": [0.0] * 10,
        "longitude": [0.0] * 10,
        "churn_label": ["Yes", "No"] * 5,
        "churn_reason": ["Price", None] * 5,
        "cltv": [100] * 10,
        "churn_score": [80] * 10,
    }
    return pd.DataFrame(data)


@pytest.fixture
def builder():
    """Instancia o PipelineBuilder."""
    return PipelineBuilder(data_path="fake_path.csv")


def test_carregar_dados(builder, sample_df):
    """Testa se o carregamento de CSV é chamado corretamente."""
    with patch("pandas.read_csv", return_value=sample_df) as mock_read:
        df = builder.carregar()
        mock_read.assert_called_once_with("fake_path.csv")
        # Atualizado de (4, 18) para (10, 18) para bater com a nova fixture
        assert df.shape == (10, 18)


def test_preparar_features_target(builder, sample_df):
    """Testa a remoção de colunas e separação de X e y."""
    builder.df = sample_df
    X, y = builder.preparar_features_target()

    # Verifica se colunas irrelevantes foram removidas
    assert "customerid" not in X.columns
    assert "churn_value" not in X.columns
    assert "cltv" not in X.columns
    # Verifica se o rename funcionou
    assert y.name == "churn_value"
    # Verifica tamanho (deve sobrar apenas gender, senior_citizen, tenure_months, monthly_charges)
    # pois as outras listadas no drop_cols saíram
    assert X.shape[1] == 4


def test_identify_feature_types(builder, sample_df):
    """Verifica a separação correta entre numéricas e categóricas."""
    builder.df = sample_df
    X, _ = builder.preparar_features_target()
    cat, num = builder.identify_feature_types(X)

    assert "gender" in cat
    assert "tenure_months" in num
    assert len(cat) == 2  # gender, senior_citizen
    assert len(num) == 2  # tenure_months, monthly_charges


def test_create_full_pipeline(builder, sample_df):
    """Verifica se a estrutura do pipeline sklearn está correta."""
    builder.df = sample_df
    X, _ = builder.preparar_features_target()
    builder.identify_feature_types(X)

    pipeline = builder.create_full_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
    assert isinstance(pipeline.named_steps["preprocessor"], ColumnTransformer)


def test_train_flow(builder, sample_df):
    """Testa o fluxo completo de treinamento."""
    builder.df = sample_df
    X, y = builder.preparar_features_target()
    builder.split_treino_teste(X, y)
    builder.identify_feature_types(builder.X_train)
    builder.create_full_pipeline()

    # Treina
    builder.train(builder.X_train, builder.y_train)

    # Verifica se o modelo está 'fitado' (não deve disparar NotFittedError)
    pred = builder.get_pipeline().predict(builder.X_test)
    assert len(pred) == len(builder.y_test)


def test_error_without_setup(builder):
    """Verifica se erros são lançados ao pular etapas."""
    with pytest.raises(ValueError, match="Chame identify_feature_types"):
        builder.create_preprocessor()

    with pytest.raises(ValueError, match="Chame create_full_pipeline"):
        builder.train(pd.DataFrame(), pd.Series())


@patch("src.models.model_training_pipeline.mlflow")
@patch("src.models.model_training_pipeline.MlflowClient")
@patch("pandas.read_csv")
def test_main_full_execution(mock_read_csv, mock_client_class, mock_mlflow, sample_df):
    """
    Testa a função main completa, garantindo que o fluxo do MLflow,
    treinamento e registro do modelo 'champion' sejam executados.
    """
    # 1. Configurar Mocks de Dados
    mock_read_csv.return_value = sample_df

    # 2. Configurar Mocks do MLflow
    # Simula o retorno do log_model para acessar registered_model_version
    mock_model_info = MagicMock()
    mock_model_info.registered_model_version = "1"
    mock_mlflow.sklearn.log_model.return_value = mock_model_info

    # Simula a busca de experimento (retorna None para forçar a criação)
    mock_mlflow.get_experiment_by_name.return_value = None

    # 3. Configurar Mock do MlflowClient
    mock_client_instance = mock_client_class.return_value

    # 4. Executar o Main
    # Usamos patch.dict para garantir que a URI de tracking seja simulada
    with patch.dict("os.environ", {"MLFLOW_TRACKING_URI": "http://localhost:5000"}):
        pipeline = main()

    # 5. Asserts de Cobertura e Lógica
    assert isinstance(pipeline, Pipeline)

    # Verifica se o experimento foi criado e setado
    mock_mlflow.create_experiment.assert_called_once_with("TelcoChurnPipeline")
    mock_mlflow.set_experiment.assert_called_with("TelcoChurnPipeline")

    # Verifica se as métricas foram logadas
    assert mock_mlflow.log_metric.called

    # Verifica se o modelo foi registrado
    mock_mlflow.sklearn.log_model.assert_called()

    # Verifica se o alias 'champion' foi atribuído
    mock_client_instance.set_registered_model_alias.assert_called_once_with(
        name="TelcoChurnPipeline", alias="champion", version="1"
    )
