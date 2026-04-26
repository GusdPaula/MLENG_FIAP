"""Logging estruturado e configurável."""

import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path

from src.config import LoggingConfig


class JSONFormatter(logging.Formatter):
    """Formatter que exporta logs em JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Formata record como JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Adicionar exception se disponível
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Adicionar campos extras
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


class StructuredLogger:
    """Logger estruturado com suporte a diferentes formatos."""

    def __init__(self, name: str, config: LoggingConfig):
        """
        Inicializa logger estruturado.

        Args:
            name: Nome do logger
            config: Configuração de logging
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config.level))

        # Remover handlers existentes
        self.logger.handlers.clear()

        # Handler para console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, config.level))
        console_formatter = logging.Formatter(config.log_format)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # Handler para arquivo (se configurado)
        if config.log_file:
            # Criar diretório se não existir
            log_path = Path(config.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                config.log_file,
                maxBytes=config.max_bytes,
                backupCount=config.backup_count
            )
            file_handler.setLevel(getattr(logging, config.level))
            file_formatter = JSONFormatter()
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def debug(self, message: str, **kwargs):
        """Log debug."""
        self._log(logging.DEBUG, message, kwargs)

    def info(self, message: str, **kwargs):
        """Log info."""
        self._log(logging.INFO, message, kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning."""
        self._log(logging.WARNING, message, kwargs)

    def error(self, message: str, **kwargs):
        """Log error."""
        self._log(logging.ERROR, message, kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical."""
        self._log(logging.CRITICAL, message, kwargs)

    def _log(self, level: int, message: str, extra_fields: dict):
        """Log com campos extras."""
        record = self.logger.makeRecord(
            self.logger.name,
            level,
            "(unknown file)",
            0,
            message,
            (),
            None
        )
        record.extra_fields = extra_fields
        self.logger.handle(record)


class APILogger:
    """Logger especializado para API."""

    def __init__(self, config: LoggingConfig):
        """Inicializa API logger."""
        self.logger = StructuredLogger("api", config)

    def log_request(self, method: str, path: str, client: str):
        """Log de request HTTP."""
        self.logger.info(
            "Incoming request",
            method=method,
            path=path,
            client=client
        )

    def log_response(self, method: str, path: str, status_code: int, latency_ms: float):
        """Log de response HTTP."""
        self.logger.info(
            "Outgoing response",
            method=method,
            path=path,
            status_code=status_code,
            latency_ms=round(latency_ms, 2)
        )

    def log_prediction(self, n_samples: int, latency_ms: float, success: bool):
        """Log de predição."""
        level = "info" if success else "error"
        msg = "Prediction completed" if success else "Prediction failed"
        getattr(self.logger, level)(
            msg,
            n_samples=n_samples,
            latency_ms=round(latency_ms, 2),
            success=success
        )

    def log_error(self, error_type: str, error_msg: str, context: dict | None = None):
        """Log de erro."""
        self.logger.error(
            "Error occurred",
            error_type=error_type,
            error_msg=error_msg,
            context=context or {}
        )


def get_logger(name: str, config: LoggingConfig | None = None) -> logging.Logger:
    """Retorna logger configurado."""
    if config is None:
        from src.config import DEFAULT_LOGGING_CONFIG
        config = DEFAULT_LOGGING_CONFIG

    structured_logger = StructuredLogger(name, config)
    return structured_logger.logger
