"""Testes unitários para config module."""

from src.config import (
    APIConfig,
    DataConfig,
    LoggingConfig,
    MetricsConfig,
    ModelConfig,
    get_config,
)


class TestDataConfig:
    """Testes para DataConfig."""

    def test_default_values(self):
        """Testa valores padrão."""
        config = DataConfig()
        assert config.test_size == 0.2
        assert config.val_size == 0.2
        assert config.random_state == 42
        assert "CustomerID" in config.drop_columns

    def test_custom_values(self):
        """Testa valores customizados."""
        config = DataConfig(test_size=0.3, random_state=123)
        assert config.test_size == 0.3
        assert config.random_state == 123

    def test_drop_columns_populated(self):
        """Testa que drop_columns é populado."""
        config = DataConfig()
        assert isinstance(config.drop_columns, list)
        assert len(config.drop_columns) > 0


class TestModelConfig:
    """Testes para ModelConfig."""

    def test_default_values(self):
        """Testa valores padrão."""
        config = ModelConfig()
        assert config.batch_size == 32
        assert config.learning_rate == 0.001
        assert config.epochs == 100
        assert config.hidden_sizes == [128, 64, 32]

    def test_custom_architecture(self):
        """Testa arquitetura customizada."""
        config = ModelConfig(hidden_sizes=[256, 128], dropout_rates=[0.4, 0.2])
        assert config.hidden_sizes == [256, 128]
        assert config.dropout_rates == [0.4, 0.2]


class TestMetricsConfig:
    """Testes para MetricsConfig."""

    def test_business_metrics(self):
        """Testa métricas de negócio."""
        config = MetricsConfig()
        assert config.customer_ltv == 2080.0
        assert config.retention_cost == 50.0
        assert config.false_positive_cost == 20.0


class TestAPIConfig:
    """Testes para APIConfig."""

    def test_development_config(self):
        """Testa configuração de desenvolvimento."""
        config = APIConfig(debug=True, reload=True)
        assert config.debug is True
        assert config.reload is True

    def test_production_config(self):
        """Testa configuração de produção."""
        config = APIConfig(debug=False, reload=False, workers=4)
        assert config.debug is False
        assert config.workers == 4


class TestLoggingConfig:
    """Testes para LoggingConfig."""

    def test_logging_levels(self):
        """Testa diferentes níveis de logging."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            config = LoggingConfig(level=level)
            assert config.level == level


class TestGetConfig:
    """Testes para function get_config."""

    def test_development_environment(self):
        """Testa configuração de desenvolvimento."""
        config = get_config("development")
        assert config["api"].debug is True
        assert config["logging"].level == "DEBUG"

    def test_production_environment(self):
        """Testa configuração de produção."""
        config = get_config("production")
        assert config["api"].debug is False
        assert config["api"].workers == 4

    def test_testing_environment(self):
        """Testa configuração de testes."""
        config = get_config("testing")
        assert config["model"].epochs == 10

    def test_config_has_required_keys(self):
        """Testa que config tem todas as chaves necessárias."""
        config = get_config()
        required_keys = ["data", "model", "metrics", "api", "logging"]
        for key in required_keys:
            assert key in config, f"Chave '{key}' não encontrada em config"

    def test_all_config_objects_are_dataclasses(self):
        """Testa que todos os configs são dataclasses."""
        config = get_config()
        from dataclasses import is_dataclass

        for key, value in config.items():
            assert is_dataclass(value), f"{key} não é um dataclass"
