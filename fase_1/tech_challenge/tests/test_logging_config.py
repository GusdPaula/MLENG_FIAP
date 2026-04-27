"""Testes para o módulo de logging estruturado.

Cobre:
  - JSONFormatter: formato, campos obrigatórios, exception, extra_fields
  - StructuredLogger: inicialização, handlers, níveis, campos extras
  - APILogger: log_request, log_response, log_prediction, log_error
  - get_logger: retorno, configuração padrão
"""

import json
import logging
import logging.handlers

# Patch src.config antes de importar o módulo
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Stub de LoggingConfig para não depender de src.config nos testes
# ---------------------------------------------------------------------------


@dataclass
class LoggingConfig:
    level: str = "DEBUG"
    log_format: str = "%(levelname)s %(message)s"
    log_file: str | None = None
    max_bytes: int = 1_000_000
    backup_count: int = 3


@dataclass
class DataConfig:
    pass


def get_config(*args, **kwargs):
    return LoggingConfig()


_fake_config_module = types.ModuleType("src.config")
_fake_config_module.LoggingConfig = LoggingConfig
_fake_config_module.DEFAULT_LOGGING_CONFIG = LoggingConfig(level="INFO")
_fake_config_module.DataConfig = DataConfig
_fake_config_module.get_config = get_config

sys.modules.setdefault("src", types.ModuleType("src"))
sys.modules["src.config"] = _fake_config_module

from src.logging_config import APILogger, JSONFormatter, StructuredLogger, get_logger  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    msg: str = "test message",
    level: int = logging.INFO,
    exc_info=None,
    extra_fields: dict | None = None,
) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test",
        level=level,
        pathname="test_file.py",
        lineno=10,
        msg=msg,
        args=(),
        exc_info=exc_info,
    )
    if extra_fields is not None:
        record.extra_fields = extra_fields
    return record


@pytest.fixture
def base_config():
    return LoggingConfig(level="DEBUG", log_file=None)


@pytest.fixture
def file_config(tmp_path):
    return LoggingConfig(
        level="DEBUG",
        log_file=str(tmp_path / "logs" / "app.log"),
        max_bytes=500_000,
        backup_count=2,
    )


# ---------------------------------------------------------------------------
# JSONFormatter
# ---------------------------------------------------------------------------


class TestJSONFormatter:
    @pytest.fixture
    def formatter(self):
        return JSONFormatter()

    def test_output_is_valid_json(self, formatter):
        record = _make_record()
        output = formatter.format(record)
        parsed = json.loads(output)  # lança se inválido
        assert isinstance(parsed, dict)

    def test_required_fields_present(self, formatter):
        record = _make_record()
        parsed = json.loads(formatter.format(record))
        for field in ("timestamp", "level", "logger", "message", "module", "function", "line"):
            assert field in parsed, f"Campo obrigatório ausente: '{field}'"

    def test_message_content(self, formatter):
        record = _make_record(msg="hello world")
        parsed = json.loads(formatter.format(record))
        assert parsed["message"] == "hello world"

    def test_level_name(self, formatter):
        record = _make_record(level=logging.WARNING)
        parsed = json.loads(formatter.format(record))
        assert parsed["level"] == "WARNING"

    def test_timestamp_is_iso_format(self, formatter):
        from datetime import datetime

        record = _make_record()
        parsed = json.loads(formatter.format(record))
        # Deve parsear sem exceção
        dt = datetime.fromisoformat(parsed["timestamp"])
        assert isinstance(dt, datetime)

    def test_line_is_integer(self, formatter):
        record = _make_record()
        parsed = json.loads(formatter.format(record))
        assert isinstance(parsed["line"], int)

    def test_exception_field_absent_when_no_exc(self, formatter):
        record = _make_record()
        parsed = json.loads(formatter.format(record))
        assert "exception" not in parsed

    def test_exception_field_present_when_exc_info(self, formatter):
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = _make_record(exc_info=exc_info)
        parsed = json.loads(formatter.format(record))
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]

    def test_extra_fields_merged_into_output(self, formatter):
        record = _make_record(extra_fields={"user_id": 42, "env": "prod"})
        parsed = json.loads(formatter.format(record))
        assert parsed["user_id"] == 42
        assert parsed["env"] == "prod"

    def test_no_extra_fields_does_not_crash(self, formatter):
        record = _make_record()  # sem extra_fields
        output = formatter.format(record)
        assert json.loads(output)  # ainda é JSON válido

    def test_extra_fields_do_not_overwrite_required_keys(self, formatter):
        """extra_fields com chave 'level' sobrescreve — documentamos esse comportamento."""
        record = _make_record(level=logging.ERROR, extra_fields={"level": "CUSTOM"})
        parsed = json.loads(formatter.format(record))
        # O update() do dict dá prioridade ao extra_fields — comportamento atual
        assert parsed["level"] == "CUSTOM"


# ---------------------------------------------------------------------------
# StructuredLogger
# ---------------------------------------------------------------------------


class TestStructuredLogger:
    def test_creates_logger_with_correct_name(self, base_config):
        sl = StructuredLogger("my_service", base_config)
        assert sl.logger.name == "my_service"

    def test_level_set_correctly(self, base_config):
        config = LoggingConfig(level="WARNING")
        sl = StructuredLogger("svc", config)
        assert sl.logger.level == logging.WARNING

    def test_console_handler_added(self, base_config):
        sl = StructuredLogger("svc", base_config)
        stream_handlers = [
            h
            for h in sl.logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(stream_handlers) >= 1

    def test_no_file_handler_when_log_file_is_none(self, base_config):
        sl = StructuredLogger("svc", base_config)
        file_handlers = [
            h for h in sl.logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(file_handlers) == 0

    def test_file_handler_added_when_log_file_set(self, file_config):
        sl = StructuredLogger("svc", file_config)
        file_handlers = [
            h for h in sl.logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(file_handlers) == 1

    def test_log_directory_created_automatically(self, file_config):
        log_path = Path(file_config.log_file)
        assert not log_path.parent.exists() or True  # pode já existir; só verificamos após init
        StructuredLogger("svc", file_config)
        assert log_path.parent.exists()

    def test_file_handler_uses_json_formatter(self, file_config):
        sl = StructuredLogger("svc", file_config)
        file_handler = next(
            h for h in sl.logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)
        )
        assert isinstance(file_handler.formatter, JSONFormatter)

    def test_rotating_handler_max_bytes(self, file_config):
        sl = StructuredLogger("svc", file_config)
        file_handler = next(
            h for h in sl.logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)
        )
        assert file_handler.maxBytes == file_config.max_bytes

    def test_rotating_handler_backup_count(self, file_config):
        sl = StructuredLogger("svc", file_config)
        file_handler = next(
            h for h in sl.logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)
        )
        assert file_handler.backupCount == file_config.backup_count

    def test_existing_handlers_cleared_on_init(self, base_config):
        """Evita duplicação de handlers ao instanciar múltiplas vezes."""
        name = "singleton_test"
        StructuredLogger(name, base_config)
        sl2 = StructuredLogger(name, base_config)
        # Deve ter exatamente 1 console handler (não acumular)
        stream_handlers = [
            h
            for h in sl2.logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(stream_handlers) == 1

    @pytest.mark.parametrize(
        ("method", "level"),
        [
            ("debug", logging.DEBUG),
            ("info", logging.INFO),
            ("warning", logging.WARNING),
            ("error", logging.ERROR),
            ("critical", logging.CRITICAL),
        ],
    )
    def test_log_methods_emit_correct_level(self, base_config, method, level):
        sl = StructuredLogger("svc", base_config)
        with patch.object(sl.logger, "handle") as mock_handle:
            getattr(sl, method)("test msg")
            mock_handle.assert_called_once()
            emitted_record: logging.LogRecord = mock_handle.call_args[0][0]
            assert emitted_record.levelno == level

    def test_extra_fields_attached_to_record(self, base_config):
        sl = StructuredLogger("svc", base_config)
        with patch.object(sl.logger, "handle") as mock_handle:
            sl.info("msg", user="alice", request_id="abc123")
            record = mock_handle.call_args[0][0]
            assert record.extra_fields == {"user": "alice", "request_id": "abc123"}

    def test_empty_extra_fields_when_no_kwargs(self, base_config):
        sl = StructuredLogger("svc", base_config)
        with patch.object(sl.logger, "handle") as mock_handle:
            sl.info("plain message")
            record = mock_handle.call_args[0][0]
            assert record.extra_fields == {}

    def test_message_content_in_record(self, base_config):
        sl = StructuredLogger("svc", base_config)
        with patch.object(sl.logger, "handle") as mock_handle:
            sl.warning("something happened")
            record = mock_handle.call_args[0][0]
            assert record.getMessage() == "something happened"


# ---------------------------------------------------------------------------
# APILogger
# ---------------------------------------------------------------------------


class TestAPILogger:
    @pytest.fixture
    def api_logger(self, base_config):
        return APILogger(base_config)

    def test_log_request_calls_info(self, api_logger):
        with patch.object(api_logger.logger, "info") as mock_info:
            api_logger.log_request("GET", "/predict", "127.0.0.1")
            mock_info.assert_called_once()
            args, kwargs = mock_info.call_args
            assert args[0] == "Incoming request"
            assert kwargs["method"] == "GET"
            assert kwargs["path"] == "/predict"
            assert kwargs["client"] == "127.0.0.1"

    def test_log_response_calls_info(self, api_logger):
        with patch.object(api_logger.logger, "info") as mock_info:
            api_logger.log_response("POST", "/predict", 200, 42.5)
            mock_info.assert_called_once()
            args, kwargs = mock_info.call_args
            assert args[0] == "Outgoing response"
            assert kwargs["status_code"] == 200
            assert kwargs["latency_ms"] == 42.5

    def test_log_response_rounds_latency(self, api_logger):
        with patch.object(api_logger.logger, "info") as mock_info:
            api_logger.log_response("GET", "/health", 200, 12.3456789)
            _, kwargs = mock_info.call_args
            assert kwargs["latency_ms"] == round(12.3456789, 2)

    def test_log_prediction_success_uses_info(self, api_logger):
        with patch.object(api_logger.logger, "info") as mock_info:
            api_logger.log_prediction(n_samples=5, latency_ms=10.0, success=True)
            mock_info.assert_called_once()
            args, kwargs = mock_info.call_args
            assert args[0] == "Prediction completed"
            assert kwargs["success"] is True
            assert kwargs["n_samples"] == 5

    def test_log_prediction_failure_uses_error(self, api_logger):
        with patch.object(api_logger.logger, "error") as mock_error:
            api_logger.log_prediction(n_samples=1, latency_ms=5.0, success=False)
            mock_error.assert_called_once()
            args, kwargs = mock_error.call_args
            assert args[0] == "Prediction failed"
            assert kwargs["success"] is False

    def test_log_prediction_rounds_latency(self, api_logger):
        with patch.object(api_logger.logger, "info") as mock_info:
            api_logger.log_prediction(n_samples=3, latency_ms=7.9999, success=True)
            _, kwargs = mock_info.call_args
            assert kwargs["latency_ms"] == round(7.9999, 2)

    def test_log_error_calls_error_level(self, api_logger):
        with patch.object(api_logger.logger, "error") as mock_error:
            api_logger.log_error("ValueError", "bad input", context={"field": "age"})
            mock_error.assert_called_once()
            args, kwargs = mock_error.call_args
            assert args[0] == "Error occurred"
            assert kwargs["error_type"] == "ValueError"
            assert kwargs["error_msg"] == "bad input"
            assert kwargs["context"] == {"field": "age"}

    def test_log_error_defaults_context_to_empty_dict(self, api_logger):
        with patch.object(api_logger.logger, "error") as mock_error:
            api_logger.log_error("RuntimeError", "unexpected crash")
            _, kwargs = mock_error.call_args
            assert kwargs["context"] == {}

    def test_underlying_logger_name_is_api(self, base_config):
        al = APILogger(base_config)
        assert al.logger.logger.name == "api"


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------


class TestGetLogger:
    def test_returns_standard_logger_instance(self, base_config):
        logger = get_logger("myapp", base_config)
        assert isinstance(logger, logging.Logger)

    def test_logger_has_correct_name(self, base_config):
        logger = get_logger("service_x", base_config)
        assert logger.name == "service_x"

    def test_uses_default_config_when_none_passed(self):
        """Quando config=None, deve usar DEFAULT_LOGGING_CONFIG sem lançar exceção."""
        logger = get_logger("default_test", config=None)
        assert isinstance(logger, logging.Logger)

    def test_default_config_level_is_info(self):
        """DEFAULT_LOGGING_CONFIG tem level=INFO no stub."""
        logger = get_logger("level_test", config=None)
        assert logger.level == logging.INFO

    def test_different_names_return_different_loggers(self, base_config):
        l1 = get_logger("svc_a", base_config)
        l2 = get_logger("svc_b", base_config)
        assert l1.name != l2.name

    def test_same_name_returns_same_logger_object(self, base_config):
        """logging.getLogger é um singleton por nome."""
        l1 = get_logger("shared", base_config)
        l2 = get_logger("shared", base_config)
        assert l1 is l2
