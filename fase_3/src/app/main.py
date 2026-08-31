"""Application entrypoint.

Uses the app-factory pattern (create_app()) rather than a bare module-level
`app = FastAPI()` so tests can spin up isolated instances, and lifespan
(not the deprecated @app.on_event) to load the ONNX model exactly once at
startup rather than on every request.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.monitoring.metrics import setup_metrics
from app.services.inference import InferenceService

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hook.

    Loads the ONNX Runtime session once and stores it on app.state, so
    every request reuses the same in-memory session instead of paying
    model-load cost per call — this is the single biggest latency win,
    before any model-level optimization.
    """
    configure_logging(settings.log_level)

    inference_service = InferenceService(
        model_path=settings.model_path,
        class_labels=settings.class_labels,
        model_version=settings.model_version,
    )
    inference_service.warm_up()  # fires a dummy inference so the first
    # real request isn't penalized by lazy ONNX Runtime initialization

    app.state.inference_service = inference_service
    app.state.settings = settings

    yield

    app.state.inference_service = None


def create_app() -> FastAPI:
    """Builds and wires the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
    )

    register_exception_handlers(app)
    setup_metrics(app)  # mounts /metrics + request-timing middleware
    app.include_router(api_router)

    return app


app = create_app()
