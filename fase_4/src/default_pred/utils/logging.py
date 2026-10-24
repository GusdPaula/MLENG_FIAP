# src/loan_prediction/logging_config.py

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(log_dir: Path | None = None) -> None:
    """Configure application-wide logging."""

    if log_dir is None:
        log_dir = Path("logs")

    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "loan_prediction.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5_000_000,  # 5 MB per file
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler],
        force=True,
    )
