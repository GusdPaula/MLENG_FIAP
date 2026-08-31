"""GET /health.

Used by Docker Compose / an orchestrator to gate traffic until the model
is actually loaded, so a request never hits an uninitialized model.
"""

from fastapi import APIRouter, Request, Response, status

from app.schemas.health import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request, response: Response) -> HealthResponse:
    inference_service = getattr(request.app.state, "inference_service", None)
    model_loaded = inference_service is not None and inference_service.is_ready()

    if not model_loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="ok" if model_loaded else "unavailable",
        model_loaded=model_loaded,
    )
