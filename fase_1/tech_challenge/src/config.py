"""Configurações centralizadas para o projeto Telco Churn."""

import os
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Configuração de caminhos, divisão treino/teste e colunas para remover."""

    raw_data_path: str = "data/raw/telco_churn_processed.csv"
    processed_data_path: str = "data/processed/telco_churn_processed.csv"
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42

    # Colunas para drop (leakage, IDs, etc)
    drop_columns: list = None

    def __post_init__(self):
        if self.drop_columns is None:
            self.drop_columns = [
                "CustomerID",
                "Count",
                "Country",
                "State",
                "City",
                "Zip Code",
                "Lat Long",
                "Latitude",
                "Longitude",
                "Churn Label",
                "Churn Reason",
                "CLTV",
                "Churn Score",
            ]


@dataclass
class ModelConfig:
    """Configuração de arquitetura MLP e hiperparâmetros de treinamento."""

    # Arquitetura MLP
    input_size: int = None  # Definido dinamicamente
    hidden_sizes: list = None
    dropout_rates: list = None
    activation: str = "relu"
    output_activation: str = "sigmoid"

    # Treinamento
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    early_stopping_patience: int = 10
    device: str = "cpu"

    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [128, 64, 32]
        if self.dropout_rates is None:
            self.dropout_rates = [0.3, 0.2, 0.0]


@dataclass
class MetricsConfig:
    """Configuração de métricas de negócio (custo/benefício de churn)."""

    customer_ltv: float = 2080.0  # Lifetime Value em USD
    retention_cost: float = 50.0  # Custo de retenção por cliente
    false_positive_cost: float = 20.0  # Custo de campanha ineficaz

    # Thresholds de performance
    auc_roc_target: float = 0.85
    f1_score_target: float = 0.65


@dataclass
class APIConfig:
    """Configurações da API."""

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    workers: int = 1

    # Timeouts
    request_timeout: int = 60
    prediction_timeout: int = 30


@dataclass
class LoggingConfig:
    """Configurações de logging."""

    level: str = "INFO"
    log_file: str = "logs/app.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5


# Configurações globais padrão
DEFAULT_DATA_CONFIG = DataConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_METRICS_CONFIG = MetricsConfig()
DEFAULT_API_CONFIG = APIConfig()
DEFAULT_LOGGING_CONFIG = LoggingConfig()


def get_config(env: str | None = None) -> dict:
    """Retorna configurações baseado no ambiente."""
    if env is None:
        env = os.getenv("ENV", "development")

    configs = {
        "development": {
            "data": DEFAULT_DATA_CONFIG,
            "model": DEFAULT_MODEL_CONFIG,
            "metrics": DEFAULT_METRICS_CONFIG,
            "api": APIConfig(debug=True, reload=True, workers=1),
            "logging": LoggingConfig(level="DEBUG"),
        },
        "production": {
            "data": DEFAULT_DATA_CONFIG,
            "model": DEFAULT_MODEL_CONFIG,
            "metrics": DEFAULT_METRICS_CONFIG,
            "api": APIConfig(debug=False, reload=False, workers=4),
            "logging": DEFAULT_LOGGING_CONFIG,
        },
        "testing": {
            "data": DataConfig(test_size=0.3, random_state=42),
            "model": ModelConfig(batch_size=16, epochs=10),
            "metrics": DEFAULT_METRICS_CONFIG,
            "api": DEFAULT_API_CONFIG,
            "logging": LoggingConfig(level="DEBUG"),
        },
    }

    return configs.get(env, configs["development"])
