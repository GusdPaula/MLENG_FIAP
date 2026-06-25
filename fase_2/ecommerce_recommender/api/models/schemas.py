"""Data transfer objects (DTOs) for prediction API requests and responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Request model for single prediction.

    Attributes:
        user_id: The user ID to generate predictions for.
        item_ids: List of item IDs to predict scores for.
        k: Number of top recommendations to return (optional).
    """

    user_id: int = Field(..., description="The user ID to generate predictions for")
    item_ids: list[int] | None = Field(default=None, description="List of item IDs to predict scores for")
    k: int | None = Field(default=None, ge=1, description="Number of top recommendations to return")

    @field_validator("k")
    @classmethod
    def validate_k(cls, v: int | None) -> int | None:
        """Validate k is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError("k must be a positive integer")
        return v


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions.

    Attributes:
        user_item_pairs: List of (user_id, item_ids) tuples where item_ids is a list of item IDs.
        k: Number of top recommendations per user (optional).
    """

    user_item_pairs: list[tuple[int, list[int]]] = Field(..., description="List of (user_id, item_ids) tuples")
    k: int | None = Field(default=None, ge=1, description="Number of top recommendations per user")

    @field_validator("k")
    @classmethod
    def validate_k(cls, v: int | None) -> int | None:
        """Validate k is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError("k must be a positive integer")
        return v


class PredictionResponse(BaseModel):
    """Response model for single prediction.

    Attributes:
        user_id: The user ID.
        item_scores: Dictionary mapping item IDs to predicted scores.
        metadata: Additional metadata about the prediction.
    """

    user_id: int = Field(..., description="The user ID")
    item_scores: dict[int, float] = Field(default_factory=dict, description="Dictionary mapping item IDs to predicted scores")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the prediction")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions.

    Attributes:
        predictions: List of PredictionResponse objects.
        metadata: Additional metadata about the batch prediction.
    """

    predictions: list[PredictionResponse] = Field(default_factory=list, description="List of prediction responses")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the batch prediction")


class RecommendationResponse(BaseModel):
    """Response model for top-k recommendations.

    Attributes:
        user_id: The user ID.
        recommendations: List of (item_id, score) tuples, sorted by score.
        metadata: Additional metadata about the recommendation.
    """

    user_id: int = Field(..., description="The user ID")
    recommendations: list[tuple[int, float]] = Field(default_factory=list, description="List of (item_id, score) tuples, sorted by score")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the recommendation")
