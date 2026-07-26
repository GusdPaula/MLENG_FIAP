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

    def test_base_predictor_cold_start_fallback(self):
        """Test base predictor methods with cold start fallback."""
        model = MockModel()

        # We need a concrete subclass to test BasePredictor methods
        class ConcretePredictor(BasePredictor):
            def predict(self, request):
                pass

            def predict_batch(self, requests):
                pass

        predictor = ConcretePredictor(
            model=model,
            user2idx={1: 0},
            item2idx={10: 0, 20: 1},
            popular_items={10: 5.0, 20: 3.0},
        )

        assert predictor.enable_cold_start_fallback is True

        # _get_user_idx fallback
        assert predictor._get_user_idx(999) is None
        assert predictor._get_user_idx(1) == 0

        # _get_popular_items
        popular = predictor._get_popular_items(k=1)
        assert popular == [(10, 5.0)]

        # _get_popular_item_scores
        scores = predictor._get_popular_item_scores([10, 20, 30])
        assert scores == {10: 5.0, 20: 3.0, 30: 0.0}

    def test_base_predictor_without_cold_start(self):
        """Test base predictor methods without cold start fallback."""
        model = MockModel()

        class ConcretePredictor(BasePredictor):
            def predict(self, request):
                pass

            def predict_batch(self, requests):
                pass

        predictor = ConcretePredictor(model=model, user2idx={1: 0}, item2idx={10: 0})

        assert predictor.enable_cold_start_fallback is False

        # _get_user_idx without fallback raises InvalidInputError
        from api.exceptions import InvalidInputError

        with pytest.raises(InvalidInputError):
            predictor._get_user_idx(999)

        assert predictor._get_popular_items() == []
        assert predictor._get_popular_item_scores([10]) == {10: 0.0}


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

    def test_single_user_predictor_cold_start(self):
        """Test SingleUserPredictor with unknown user and cold start."""
        model = MockModel(num_users=100, num_items=50)
        predictor = SingleUserPredictor(model, user2idx={1: 0}, item2idx={10: 0}, popular_items={10: 5.0})

        from api.models.schemas import PredictionRequest

        request = PredictionRequest(user_id=999, item_ids=[10])
        result = predictor.predict(request)

        assert result.user_id == 999
        assert result.item_scores == {10: 5.0}
        assert result.metadata.get("cold_start") is True


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

    def test_top_k_predictor_predict_method(self):
        """Test TopKRecommendationPredictor predict() routing."""
        model = MockModel(num_users=100, num_items=50)
        predictor = TopKRecommendationPredictor(model, user2idx={i: i for i in range(100)}, item2idx={i: i for i in range(50)}, popular_items={0: 5.0})

        from api.models.schemas import PredictionRequest

        # Test routing to _predict_top_k
        req_top_k = PredictionRequest(user_id=5, k=10)
        res_top_k = predictor.predict(req_top_k)
        assert len(res_top_k.item_scores) == 10

        # Test routing to _predict_specific_items
        req_specific = PredictionRequest(user_id=5, item_ids=[1, 2, 3])
        res_specific = predictor.predict(req_specific)
        assert len(res_specific.item_scores) == 3

        # Test missing both
        req_invalid = PredictionRequest(user_id=5)
        with pytest.raises(InvalidInputError):
            predictor.predict(req_invalid)

    def test_top_k_predictor_cold_start(self):
        """Test TopKRecommendationPredictor with unknown user and cold start."""
        model = MockModel(num_users=100, num_items=50)
        predictor = TopKRecommendationPredictor(model, user2idx={1: 0}, item2idx={10: 0}, popular_items={10: 5.0})

        # _predict_specific_items cold start
        from api.models.schemas import PredictionRequest

        req_specific = PredictionRequest(user_id=999, item_ids=[10])
        res_specific = predictor.predict(req_specific)
        assert res_specific.metadata.get("cold_start") is True

        # recommend() cold start
        res_rec = predictor.recommend(user_id=999, k=1)
        assert res_rec.metadata.get("cold_start") is True
        assert res_rec.recommendations == [(10, 5.0)]


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

    def test_batch_predictor_predict_method(self):
        """Test BatchPredictor predict() for single item."""
        model = MockModel(num_users=100, num_items=50)
        predictor = BatchPredictor(model, user2idx={1: 0}, item2idx={10: 0}, popular_items={10: 5.0})

        from api.models.schemas import PredictionRequest

        # Empty item_ids
        with pytest.raises(InvalidInputError):
            predictor.predict(PredictionRequest(user_id=1, item_ids=[]))

        # Valid prediction
        res = predictor.predict(PredictionRequest(user_id=1, item_ids=[10]))
        assert 10 in res.item_scores

        # Cold start
        res_cold = predictor.predict(PredictionRequest(user_id=999, item_ids=[10]))
        assert res_cold.metadata.get("cold_start") is True

    def test_batch_predictor_invalid_items(self):
        model = MockModel(num_users=100, num_items=50)
        predictor = BatchPredictor(model, {1: 0}, {10: 0})
        from api.models.schemas import PredictionRequest

        with pytest.raises(InvalidInputError):
            predictor.predict_batch([PredictionRequest(user_id=1, item_ids=[])])
