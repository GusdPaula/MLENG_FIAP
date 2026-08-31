"""Logging configuration.

Called once from main.py's lifespan at startup. Centralizing this here
means log format/level is consistent across app code, services, and
uvicorn's own request logs - not configured ad hoc per module.
"""

import logging
import sys

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configures the root logger.

    Plain text formatting for now (readable in `docker compose logs`);
    swap the Formatter for a JSON one later if logs need to be shipped
    to a log aggregator - the call site in main.py won't need to change.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))

    root.handlers = [handler]

    # Uvicorn's access log is noisy at DEBUG and duplicates what our own
    # request-metrics middleware already reports — keep it at INFO.
    logging.getLogger("uvicorn.access").setLevel("INFO")
