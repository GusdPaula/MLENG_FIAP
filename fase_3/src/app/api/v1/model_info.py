"""GET /v1/model/info.

Exposes what the model actually is — version, class labels, optimization
applied — so this can be checked in the STAR video without digging
through logs, and so a caller can validate they're talking to the
expected model version.
"""

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.schemas.model_info import ModelInfoResponse

router = APIRouter()


@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info(settings: Settings = Depends(get_settings)) -> ModelInfoResponse:
    return ModelInfoResponse(
        model_version=settings.model_version,
        class_labels=settings.class_labels,
        num_classes=len(settings.class_labels),
        optimization="onnx",
    )
