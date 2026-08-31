from fastapi import HTTPException, Request

from app.services.inference import InferenceService


def get_inference_service(request: Request) -> InferenceService:
    """Dependency that returns the app's singleton InferenceService.

    Raises 503 if the service is not yet available.
    """
    service: InferenceService | None = getattr(request.app.state, "inference_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="inference service not available")
    return service
