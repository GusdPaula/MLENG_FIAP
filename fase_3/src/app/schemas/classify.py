"""Schemas for POST /v1/classify and POST /v1/classify/batch.

max_length constraints are pulled from Settings rather than hardcoded, so
changing app.config.Settings.max_batch_size / max_text_length is the only
place that needs to change — these schemas stay in sync automatically.
"""

from pydantic import BaseModel, Field, field_validator

from app.config import get_settings

_settings = get_settings()


class ClassifyRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=_settings.max_text_length,
        description="Raw text of the medical abstract/laudo to classify.",
    )

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty or whitespace-only")
        return value


class BatchClassifyRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=_settings.max_batch_size,
        description=f"Laudos to classify in one call (max {_settings.max_batch_size}).",
    )

    @field_validator("texts")
    @classmethod
    def no_blank_entries(cls, values: list[str]) -> list[str]:
        if any(not v.strip() for v in values):
            raise ValueError("texts must not contain empty or whitespace-only entries")
        return values


class PredictionResult(BaseModel):
    """A single prediction — the part that varies per input text."""

    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    scores: dict[str, float]


class ClassifyResponse(PredictionResult):
    """Response for the single-item endpoint — a prediction plus the
    run metadata that, for a batch, is reported once at the top level
    instead of being repeated per item (see BatchClassifyResponse).
    """

    model_version: str
    inference_ms: float


class BatchClassifyResponse(BaseModel):
    results: list[PredictionResult]
    model_version: str
    batch_size: int
    inference_ms: float = Field(..., description="Total time for the whole batch, not per item.")
