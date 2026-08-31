"""Domain exceptions and their mapping to HTTP responses.

Route handlers and services raise these instead of HTTPException directly,
so error-to-status-code mapping lives in exactly one place instead of
being repeated (and drifting) across every endpoint.
"""

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base class for all application errors. Carries the HTTP status code
    it should map to, so handlers don't need a lookup table.
    """

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class ModelNotReadyError(AppError):
    """Raised when a request arrives before the model finished loading,
    or if the model was unloaded during shutdown. Maps to 503 — this is
    a "try again shortly" condition, not a client error.
    """

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


class InvalidInputError(AppError):
    """Raised for business-rule validation that goes beyond what Pydantic
    schemas can express — e.g. a batch exceeding max_batch_size. Kept
    distinct from Pydantic's own 422s so the two failure modes are still
    easy to tell apart in logs and metrics.
    """

    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY


class InferenceError(AppError):
    """Raised when the ONNX Runtime session itself fails during
    session.run() — a genuine server-side failure, not something the
    caller can fix by changing their request.
    """

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


def _error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": message})


def register_exception_handlers(app: FastAPI) -> None:
    """Registers handlers so every error - expected or not - returns
    structured JSON instead of crashing the worker or leaking a raw
    traceback to the caller.
    """

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.warning("app_error", extra={"path": request.url.path, "error": exc.message})
        return _error_response(exc.status_code, exc.message)

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_error", extra={"path": request.url.path})
        return _error_response(status.HTTP_500_INTERNAL_SERVER_ERROR, "Internal server error")
