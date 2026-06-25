"""Unit tests for API predictors.

Tests predictor implementations for the prediction API.
"""

import pytest
from api.domain.base_predictor import BasePredictor
from api.domain.predictors import (
    BatchPredictor,
    SingleUserPredictor,
    TopKRecommendationPredictor,
)
from api.exceptions import InvalidInputError
from torch import nn


class MockModel(nn.Module):
    """Mock model for testing."""

    def __init__(self, num_users=100, num_items=50, embedding_dim=10):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.model_name = "mock"
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_ids, item_ids):
        """Forward pass returning dot product."""
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        return (user_emb * item_emb).sum(dim=-1)


class TestBasePredictor:
    """Tests for BasePredictor abstract class."""

    def test_base_predictor_is_abstract(self):
        """Test that BasePredictor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BasePredictor(MockModel(), {}, {})


class TestSingleUserPredictor:
    """Tests for SingleUserPredictor."""

    def test_single_user_predictor_initialization(self):
        """Test SingleUserPredictor initialization."""
        model = MockModel(num_users=100, num_items=50)
        user2idx = {i: i for i in range(100)}
        item2idx = {i: i for i in range(50)}

        predictor = SingleUserPredictor(model, user2idx, item2idx)

        assert predictor.user2idx == user2idx
        assert predictor.item2idx == item2idx

    def test_single_user_predictor_valid_prediction(self):
        """Test SingleUserPredictor with valid user and items."""
        model = MockModel(num_users=100, num_items=50)
        user2idx = {i: i for i in range(100)}
        item2idx = {i: i for i in range(50)}

        predictor = SingleUserPredictor(model, user2idx, item2idx)

        # Test with valid user and items using new API
        from api.models.schemas import PredictionRequest

        request = PredictionRequest(user_id=5, item_ids=[1, 2, 3])
        result = predictor.predict(request)

        assert result.user_id == 5
        assert len(result.item_scores) == 3
        assert all(score is not None for score in result.item_scores.values())

    def test_single_user_predictor_invalid_user(self):
        """Test SingleUserPredictor with invalid user ID."""
        model = MockModel(num_users=100, num_items=50)
        user2idx = {i: i for i in range(100)}
        item2idx = {i: i for i in range(50)}

        predictor = SingleUserPredictor(model, user2idx, item2idx)

        # Test with invalid user
        from api.models.schemas import PredictionRequest

        with pytest.raises(InvalidInputError):
            request = PredictionRequest(user_id=999, item_ids=[1, 2, 3])
            predictor.predict(request)

    def test_single_user_predictor_invalid_item(self):
        """Test SingleUserPredictor with invalid item ID."""
        model = MockModel(num_users=100, num_items=50)
        user2idx = {i: i for i in range(100)}
        item2idx = {i: i for i in range(50)}

        predictor = SingleUserPredictor(model, user2idx, item2idx)

        # Test with invalid item
        from api.models.schemas import PredictionRequest

        with pytest.raises(InvalidInputError):
            request = PredictionRequest(user_id=5, item_ids=[999])
            predictor.predict(request)

    def test_single_user_predictor_empty_items(self):
        """Test SingleUserPredictor with empty item list."""
        model = MockModel(num_users=100, num_items=50)
        user2idx = {i: i for i in range(100)}
        item2idx = {i: i for i in range(50)}

        predictor = SingleUserPredictor(model, user2idx, item2idx)

        # Test with empty item list
        from api.models.schemas import PredictionRequest

        with pytest.raises(InvalidInputError):
            request = PredictionRequest(user_id=5, item_ids=[])
            predictor.predict(request)


class TestTopKRecommendationPredictor:
    """Tests for TopKRecommendationPredictor."""

    def test_top_k_predictor_initialization(self):
        """Test TopKRecommendationPredictor initialization."""
        model = MockModel(num_users=100, num_items=50)
        user2idx = {i: i for i in range(100)}
        item2idx = {i: i for i in range(50)}

        predictor = TopKRecommendationPredictor(model, user2idx, item2idx)

        assert predictor.user2idx == user2idx
        assert predictor.item2idx == item2idx

    def test_top_k_predictor_valid_recommendation(self):
        """Test TopKRecommendationPredictor with valid user."""
        model = MockModel(num_users=100, num_items=50)
        user2idx = {i: i for i in range(100)}
        item2idx = {i: i for i in range(50)}

        predictor = TopKRecommendationPredictor(model, user2idx, item2idx)

        # Test with valid user
        result = predictor.recommend(user_id=5, k=10)

        assert result.user_id == 5
        assert len(result.recommendations) == 10
        assert all(len(rec) == 2 for rec in result.recommendations)

    def test_top_k_predictor_invalid_user(self):
        """Test TopKRecommendationPredictor with invalid user ID."""
        model = MockModel(num_users=100, num_items=50)
        user2idx = {i: i for i in range(100)}
        item2idx = {i: i for i in range(50)}

        predictor = TopKRecommendationPredictor(model, user2idx, item2idx)

        # Test with invalid user
        with pytest.raises(InvalidInputError):
            predictor.recommend(user_id=999, k=10)

    def test_top_k_predictor_invalid_k(self):
        """Test TopKRecommendationPredictor with invalid k."""
        model = MockModel(num_users=100, num_items=50)
        user2idx = {i: i for i in range(100)}
        item2idx = {i: i for i in range(50)}

        predictor = TopKRecommendationPredictor(model, user2idx, item2idx)

        # Test with invalid k (negative)
        with pytest.raises(InvalidInputError):
            predictor.recommend(user_id=5, k=-1)

        # Test with invalid k (zero)
        with pytest.raises(InvalidInputError):
            predictor.recommend(user_id=5, k=0)

    def test_top_k_predictor_k_greater_than_items(self):
        """Test TopKRecommendationPredictor with k greater than available items."""
        model = MockModel(num_users=100, num_items=50)
        user2idx = {i: i for i in range(100)}
        item2idx = {i: i for i in range(50)}

        predictor = TopKRecommendationPredictor(model, user2idx, item2idx)

        # Test with k greater than number of items
        result = predictor.recommend(user_id=5, k=100)

        assert result.user_id == 5
        assert len(result.recommendations) == 50  # Limited by available items


class TestBatchPredictor:
    """Tests for BatchPredictor."""

    def test_batch_predictor_initialization(self):
        """Test BatchPredictor initialization."""
        model = MockModel(num_users=100, num_items=50)
        user2idx = {i: i for i in range(100)}
        item2idx = {i: i for i in range(50)}

        predictor = BatchPredictor(model, user2idx, item2idx)

        assert predictor.user2idx == user2idx
        assert predictor.item2idx == item2idx

    def test_batch_predictor_valid_batch(self):
        """Test BatchPredictor with valid batch requests."""
        model = MockModel(num_users=100, num_items=50)
        user2idx = {i: i for i in range(100)}
        item2idx = {i: i for i in range(50)}

        predictor = BatchPredictor(model, user2idx, item2idx)

        from api.models.schemas import PredictionRequest

        requests = [
            PredictionRequest(user_id=5, item_ids=[1, 2, 3]),
            PredictionRequest(user_id=10, item_ids=[4, 5, 6]),
        ]

        result = predictor.predict_batch(requests)

        assert len(result) == 2
        assert result[0].user_id == 5
        assert result[1].user_id == 10

    def test_batch_predictor_invalid_user_in_batch(self):
        """Test BatchPredictor with invalid user in batch."""
        model = MockModel(num_users=100, num_items=50)
        user2idx = {i: i for i in range(100)}
        item2idx = {i: i for i in range(50)}

        predictor = BatchPredictor(model, user2idx, item2idx)

        from api.models.schemas import PredictionRequest

        requests = [
            PredictionRequest(user_id=5, item_ids=[1, 2, 3]),
            PredictionRequest(user_id=999, item_ids=[4, 5, 6]),
        ]

        # Current implementation raises InvalidInputError for invalid users
        with pytest.raises(InvalidInputError):
            predictor.predict_batch(requests)

    def test_batch_predictor_empty_batch(self):
        """Test BatchPredictor with empty batch."""
        model = MockModel(num_users=100, num_items=50)
        user2idx = {i: i for i in range(100)}
        item2idx = {i: i for i in range(50)}

        predictor = BatchPredictor(model, user2idx, item2idx)

        result = predictor.predict_batch([])

        assert len(result) == 0
