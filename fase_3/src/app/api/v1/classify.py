"""POST /v1/classify/batch and POST /v1/classify.

Batch is the primary, documented flow (see README section 2.1) — it's
also the endpoint single-item classification is implemented in terms of,
so there's exactly one code path through InferenceService to test and
optimize.
"""

from fastapi import APIRouter, Depends

from app.api.dependencies import get_inference_service
from app.schemas.classify import (
    BatchClassifyRequest,
    BatchClassifyResponse,
    ClassifyRequest,
    ClassifyResponse,
)
from app.services.inference import InferenceService

router = APIRouter()


@router.post("/classify/batch", response_model=BatchClassifyResponse)
async def classify_batch(
    payload: BatchClassifyRequest,
    inference_service: InferenceService = Depends(get_inference_service),
) -> BatchClassifyResponse:
    """Primary endpoint. Runs the whole list of laudos through a single
    ONNX Runtime session.run() call — see services/inference.py for why
    this is one call instead of a loop.
    """
    return inference_service.predict_batch(payload.texts)


@router.post("/classify", response_model=ClassifyResponse)
async def classify_single(
    payload: ClassifyRequest,
    inference_service: InferenceService = Depends(get_inference_service),
) -> ClassifyResponse:
    """Secondary endpoint, for ad-hoc/single lookups and /docs demos.
    Implemented as a batch of size 1 so there's no separate inference
    code path to maintain or drift out of sync with the batch endpoint.
    """
    batch_result = inference_service.predict_batch([payload.text])
    prediction = batch_result.results[0]
    return ClassifyResponse(
        label=prediction.label,
        confidence=prediction.confidence,
        scores=prediction.scores,
        model_version=batch_result.model_version,
        inference_ms=batch_result.inference_ms,
    )
