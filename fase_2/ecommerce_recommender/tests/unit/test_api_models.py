"""Unit tests for API models.

Tests Pydantic models for request/response validation.
"""

import pytest
from api.models.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
    RecommendationResponse,
)
from pydantic import ValidationError


class TestPredictionRequest:
    """Tests for PredictionRequest model."""

    def test_valid_prediction_request(self):
        """Test creating a valid prediction request."""
        request = PredictionRequest(user_id=123, item_ids=[1, 2, 3, 4, 5])
        assert request.user_id == 123
        assert request.item_ids == [1, 2, 3, 4, 5]
        assert request.k is None

    def test_prediction_request_with_k(self):
        """Test creating a prediction request with k parameter."""
        request = PredictionRequest(user_id=123, item_ids=[1, 2, 3], k=10)
        assert request.user_id == 123
        assert request.item_ids == [1, 2, 3]
        assert request.k == 10

    def test_prediction_request_with_none_item_ids(self):
        """Test creating a prediction request with None item_ids."""
        request = PredictionRequest(user_id=123, item_ids=None, k=10)
        assert request.user_id == 123
        assert request.item_ids is None
        assert request.k == 10

    def test_invalid_k_negative(self):
        """Test that negative k raises validation error."""
        with pytest.raises(ValidationError):
            PredictionRequest(user_id=123, item_ids=[1, 2, 3], k=-1)

    def test_invalid_k_zero(self):
        """Test that zero k raises validation error."""
        with pytest.raises(ValidationError):
            PredictionRequest(user_id=123, item_ids=[1, 2, 3], k=0)


class TestPredictionResponse:
    """Tests for PredictionResponse model."""

    def test_valid_prediction_response(self):
        """Test creating a valid prediction response."""
        response = PredictionResponse(
            user_id=123,
            item_scores={"1": 0.95, "2": 0.87, "3": 0.72},
            metadata={"predictor": "single_user"},
        )
        assert response.user_id == 123
        assert response.item_scores == {1: 0.95, 2: 0.87, 3: 0.72}
        assert response.metadata == {"predictor": "single_user"}

    def test_prediction_response_with_empty_metadata(self):
        """Test creating a prediction response with empty metadata."""
        response = PredictionResponse(user_id=123, item_scores={"1": 0.95, "2": 0.87})
        assert response.user_id == 123
        assert response.item_scores == {1: 0.95, 2: 0.87}
        assert response.metadata == {}


class TestBatchPredictionRequest:
    """Tests for BatchPredictionRequest model."""

    def test_valid_batch_request(self):
        """Test creating a valid batch prediction request."""
        request = BatchPredictionRequest(
            user_item_pairs=[
                [123, [1, 2, 3]],
                [456, [4, 5, 6]],
            ],
            k=None,
        )
        assert len(request.user_item_pairs) == 2
        assert request.user_item_pairs[0] == (123, [1, 2, 3])
        assert request.k is None

    def test_batch_request_with_k(self):
        """Test creating a batch request with k parameter."""
        request = BatchPredictionRequest(
            user_item_pairs=[
                [123, [1, 2, 3]],
                [456, [4, 5, 6]],
            ],
            k=10,
        )
        assert len(request.user_item_pairs) == 2
        assert request.k == 10

    def test_invalid_k_negative(self):
        """Test that negative k raises validation error."""
        with pytest.raises(ValidationError):
            BatchPredictionRequest(user_item_pairs=[[123, [1, 2, 3]]], k=-1)

    def test_invalid_k_zero(self):
        """Test that zero k raises validation error."""
        with pytest.raises(ValidationError):
            BatchPredictionRequest(user_item_pairs=[[123, [1, 2, 3]]], k=0)


class TestBatchPredictionResponse:
    """Tests for BatchPredictionResponse model."""

    def test_valid_batch_response(self):
        """Test creating a valid batch prediction response."""
        response = BatchPredictionResponse(
            predictions=[
                PredictionResponse(
                    user_id=123,
                    item_scores={"1": 0.95, "2": 0.87},
                ),
                PredictionResponse(
                    user_id=456,
                    item_scores={"3": 0.72, "4": 0.65},
                ),
            ],
            metadata={"num_requests": 2},
        )
        assert len(response.predictions) == 2
        assert response.predictions[0].user_id == 123
        assert response.metadata == {"num_requests": 2}

    def test_batch_response_with_empty_predictions(self):
        """Test creating a batch response with empty predictions."""
        response = BatchPredictionResponse(predictions=[], metadata={})
        assert response.predictions == []
        assert response.metadata == {}


class TestRecommendationResponse:
    """Tests for RecommendationResponse model."""

    def test_valid_recommendation_response(self):
        """Test creating a valid recommendation response."""
        response = RecommendationResponse(
            user_id=123,
            recommendations=[
                [1, 0.95],
                [2, 0.87],
                [3, 0.72],
            ],
            metadata={"predictor": "top_k", "k": 10},
        )
        assert response.user_id == 123
        assert len(response.recommendations) == 3
        assert response.recommendations[0] == (1, 0.95)
        assert response.metadata == {"predictor": "top_k", "k": 10}

    def test_recommendation_response_with_empty_recommendations(self):
        """Test creating a recommendation response with empty recommendations."""
        response = RecommendationResponse(user_id=123, recommendations=[], metadata={})
        assert response.user_id == 123
        assert response.recommendations == []
        assert response.metadata == {}
