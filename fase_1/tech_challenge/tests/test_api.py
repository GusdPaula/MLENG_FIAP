from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Ajuste o import para o local correto do seu create_app
from src.api.main import create_app


@pytest.fixture
def valid_customer_data():
    """Retorna um dicionário com todos os campos obrigatórios de CustomerFeatures."""
    return {
        "gender": "Female",
        "senior_citizen": "No",
        "partner": "Yes",
        "dependents": "No",
        "tenure_months": 12,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "DSL",
        "online_security": "Yes",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "Yes",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 70.35,
        "total_charges": 844.2,
    }


@pytest.fixture
def mock_pipeline():
    """Mock do pipeline do Scikit-learn."""
    pipeline = MagicMock()
    pipeline.predict.return_value = np.array([0])
    pipeline.predict_proba.return_value = np.array([[0.7, 0.3]])

    # Mock para model-info
    model_mock = MagicMock()
    prep_mock = MagicMock()
    prep_mock.get_feature_names_out.return_value = np.array(["tenure_months", "monthly_charges"])
    pipeline.named_steps = {"model": model_mock, "preprocessor": prep_mock}
    return pipeline


@pytest.fixture
def client(mock_pipeline):
    """Configura o cliente com o ModelManager mockado."""
    with patch("src.api.main.model_manager") as mocked_manager:
        mocked_manager.load_from_mlflow.return_value = True
        mocked_manager.pipeline = mock_pipeline
        # O método predict do manager deve retornar um array compatível
        mocked_manager.predict.return_value = np.array([0])

        app = create_app()
        with TestClient(app) as c:
            yield c


# --- Testes de Endpoint ---


def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_single_success(client, valid_customer_data):
    payload = {"features": valid_customer_data, "return_probability": True}
    response = client.post("/api/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["probability"] == 0.3  # Valor definido no mock_pipeline


def test_predict_batch_success(client, valid_customer_data, mock_pipeline):
    # 1. Preparamos o mock para retornar EXATAMENTE 2 resultados
    # Isso sobrescreve o valor padrão definido na fixture do client
    with patch("src.api.main.model_manager.predict") as mocked_predict:
        mocked_predict.return_value = np.array([0, 1])  # Dois resultados

        # 2. Mockamos as probabilidades para 2 amostras (se seu código usa predict_proba)
        mock_pipeline.predict_proba.return_value = np.array(
            [
                [0.9, 0.1],  # Amostra 1
                [0.4, 0.6],  # Amostra 2
            ]
        )

        payload = {
            "samples": [valid_customer_data, valid_customer_data],  # 2 samples
            "return_probabilities": True,
        }

        response = client.post("/api/predict-batch", json=payload)

        # 3. Verificações
        assert response.status_code == 200
        data = response.json()

        assert data["batch_size"] == 2
        assert len(data["predictions"]) == 2
        assert data["predictions"] == [0, 1]
        assert data["probabilities"] == [0.1, 0.6]


def test_model_info(client):
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["n_features"] == 2
    assert "tenure_months" in data["features_used"]


def test_predict_service_unavailable(client, valid_customer_data):
    """Testa erro 503 quando o pipeline some do state."""
    # Resetamos o state e impedimos recarregamento
    with patch("src.api.main.model_manager") as mocked_manager:
        mocked_manager.load_from_mlflow.return_value = False
        client.app.state.pipeline = None

        payload = {"features": valid_customer_data, "return_probability": False}
        response = client.post("/api/predict", json=payload)

        assert response.status_code == 503
        assert "Modelo não disponível" in response.json()["detail"]
