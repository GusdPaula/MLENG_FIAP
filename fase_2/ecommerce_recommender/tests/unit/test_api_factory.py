"""Unit tests for API factory.

Tests the PredictorFactory for creating predictor instances.
"""

import pytest
import torch
from api.domain.base_predictor import BasePredictor
from api.exceptions import PredictorNotFoundError
from api.domain.predictor_factory import PredictorFactory
from api.domain.predictors import (
    BatchPredictor,
    SingleUserPredictor,
    TopKRecommendationPredictor,
)
from torch import nn


class MockModel(nn.Module):
    """Mock model for testing."""

    def __init__(self, num_users=100, num_items=50):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.model_name = "mock"

    def forward(self, user_ids, item_ids):
        """Forward pass returning zeros."""
        return torch.zeros_like(user_ids, dtype=torch.float32)


class TestPredictorFactory:
    """Tests for PredictorFactory."""

    def test_predictor_factory_registration(self):
        """Test that built-in predictors are registered."""
        available = PredictorFactory.available_predictors()
        assert "single_user" in available
        assert "top_k" in available
        assert "batch" in available

    def test_create_single_user_predictor(self):
        """Test creating a SingleUserPredictor."""
        model = MockModel(num_users=100, num_items=50)
        user2idx = {i: i for i in range(100)}
        item2idx = {i: i for i in range(50)}

        predictor = PredictorFactory.create(
            predictor_type="single_user",
            model=model,
            user2idx=user2idx,
            item2idx=item2idx,
        )

        assert isinstance(predictor, SingleUserPredictor)

    def test_create_top_k_predictor(self):
        """Test creating a TopKRecommendationPredictor."""
        model = MockModel(num_users=100, num_items=50)
        user2idx = {i: i for i in range(100)}
        item2idx = {i: i for i in range(50)}

        predictor = PredictorFactory.create(
            predictor_type="top_k",
            model=model,
            user2idx=user2idx,
            item2idx=item2idx,
        )

        assert isinstance(predictor, TopKRecommendationPredictor)

    def test_create_batch_predictor(self):
        """Test creating a BatchPredictor."""
        model = MockModel(num_users=100, num_items=50)
        user2idx = {i: i for i in range(100)}
        item2idx = {i: i for i in range(50)}

        predictor = PredictorFactory.create(
            predictor_type="batch",
            model=model,
            user2idx=user2idx,
            item2idx=item2idx,
        )

        assert isinstance(predictor, BatchPredictor)

    def test_create_invalid_predictor_type(self):
        """Test creating a predictor with invalid type raises error."""
        model = MockModel(num_users=100, num_items=50)
        user2idx = {i: i for i in range(100)}
        item2idx = {i: i for i in range(50)}

        with pytest.raises(PredictorNotFoundError):
            PredictorFactory.create(
                predictor_type="invalid_type",
                model=model,
                user2idx=user2idx,
                item2idx=item2idx,
            )

    def test_register_custom_predictor(self):
        """Test registering a custom predictor."""
        model = MockModel(num_users=100, num_items=50)
        user2idx = {i: i for i in range(100)}
        item2idx = {i: i for i in range(50)}

        # Register a custom predictor
        @PredictorFactory.register("custom")
        class CustomPredictor(BasePredictor):
            """Custom predictor for testing."""

            def predict(self, request):
                """Return dummy prediction."""
                from recommender.api.models import PredictionResponse
                return PredictionResponse(
                    user_id=request.user_id,
                    item_scores=dict.fromkeys(request.item_ids, 0.5),
                )

            def predict_batch(self, requests):
                """Return dummy batch prediction."""
                from recommender.api.models import PredictionResponse
                return [
                    PredictionResponse(
                        user_id=req.user_id,
                        item_scores=dict.fromkeys(req.item_ids, 0.5),
                    )
                    for req in requests
                ]

        # Create the custom predictor
        predictor = PredictorFactory.create(
            predictor_type="custom",
            model=model,
            user2idx=user2idx,
            item2idx=item2idx,
        )

        assert isinstance(predictor, CustomPredictor)
        assert "custom" in PredictorFactory.available_predictors()

    def test_predictor_factory_is_singleton(self):
        """Test that PredictorFactory maintains registry state."""
        # Get available predictors twice
        first_call = PredictorFactory.available_predictors()
        second_call = PredictorFactory.available_predictors()

        # Should return the same registry
        assert first_call == second_call

    def test_create_predictor_with_empty_mappings(self):
        """Test creating a predictor with empty user/item mappings."""
        model = MockModel(num_users=100, num_items=50)
        user2idx = {}
        item2idx = {}

        predictor = PredictorFactory.create(
            predictor_type="single_user",
            model=model,
            user2idx=user2idx,
            item2idx=item2idx,
        )

        assert isinstance(predictor, SingleUserPredictor)
