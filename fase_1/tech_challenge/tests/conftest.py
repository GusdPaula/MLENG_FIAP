"""Fixtures e configurações para testes."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture()
def client():
    # Don't mock the state.
    # Just ensure the real model is loaded before yielding the client.
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def tmp_data_dir():
    """Cria diretório temporário para testes de dados."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture()
def sample_csv(tmp_data_dir):
    """Cria arquivo CSV de amostra para testes."""
    df = pd.DataFrame(
        {
            "age": [25, 35, 45, 55, 65],
            "tenure": [1, 12, 24, 36, 48],
            "monthly_charges": [50.0, 75.0, 100.0, 125.0, 150.0],
            "internet_service": ["Yes", "No", "Yes", "Yes", "No"],
            "churn_value": [0, 1, 0, 1, 0],
        }
    )

    path = Path(tmp_data_dir) / "sample.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture()
def X_train_sample():
    """Features de treino para testes."""
    return np.array(
        [
            [25, 1, 50.0],
            [35, 12, 75.0],
            [45, 24, 100.0],
            [55, 36, 125.0],
        ]
    )


@pytest.fixture()
def y_train_sample():
    """Labels de treino para testes."""
    return np.array([0, 1, 0, 1])


@pytest.fixture()
def X_test_sample():
    """Features de teste para testes."""
    return np.array(
        [
            [30, 6, 60.0],
            [50, 30, 110.0],
        ]
    )


@pytest.fixture()
def y_test_sample():
    """Labels de teste para testes."""
    return np.array([0, 1])


@pytest.fixture()
def sample_dataframe():
    """DataFrame de amostra para testes de preprocessing."""
    return pd.DataFrame(
        {
            "age": [25, 35, 45, 55],
            "tenure": [1, 12, 24, 36],
            "monthly_charges": [50.0, 75.0, 100.0, 125.0],
            "internet_service": ["Yes", "No", "Yes", "Yes"],
            "contract_type": ["Month-to-Month", "One Year", "Two Year", "Month-to-Month"],
            "churn_value": [0, 1, 0, 1],
        }
    )


@pytest.fixture(autouse=True)
def _env_vars():
    """Define variáveis de ambiente para testes."""
    os.environ["ENV"] = "testing"
    os.environ["DEBUG"] = "False"
    yield
    # Cleanup
    if "ENV" in os.environ:
        del os.environ["ENV"]
    if "DEBUG" in os.environ:
        del os.environ["DEBUG"]


@pytest.fixture()
def real_data_path():
    """Retorna caminho do dataset real ou pula teste."""
    data_path = "data/processed/telco_churn_processed.csv"
    if not Path(data_path).exists():
        pytest.skip(f"Dataset não encontrado em {data_path}")
    return data_path


@pytest.fixture()
def sample_features_dict():
    """Dicionário de exemplo com 19 features nomeadas."""
    return {
        "gender": "Male",
        "senior_citizen": "No",
        "partner": "No",
        "dependents": "No",
        "tenure_months": 24,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "DSL",
        "online_security": "Yes",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Two year",
        "paperless_billing": "No",
        "payment_method": "Bank transfer (automatic)",
        "monthly_charges": 89.0,
        "total_charges": 2136.0,
    }


def pytest_configure(config):
    """Configuração inicial do pytest."""
    config.addinivalue_line("markers", "slow: marca testes que são lentos")
    config.addinivalue_line("markers", "unit: marca testes unitários")
    config.addinivalue_line("markers", "integration: marca testes de integração")
    config.addinivalue_line("markers", "smoke: marca smoke tests")
