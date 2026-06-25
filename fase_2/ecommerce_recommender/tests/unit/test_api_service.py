"""Unit tests for API service.

Tests the PredictionService for model loading and prediction orchestration.
"""

import pytest
import torch
from api.exceptions import InvalidInputError, ModelLoadError
from api.models.schemas import PredictionRequest
from api.services.prediction_service import PredictionService
from torch import nn


class MockModel(nn.Module):
    """Mock model for testing."""

    def __init__(self, num_users=100, num_items=50):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.model_name = "mock"
        self.user_embedding = nn.Embedding(num_users, 10)
        self.item_embedding = nn.Embedding(num_items, 10)

    def forward(self, user_ids, item_ids):
        """Forward pass returning dot product."""
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        return (user_emb * item_emb).sum(dim=-1)


class TestPredictionService:
    """Tests for PredictionService."""

    def test_prediction_service_initialization(self, tmp_path):
        """Test PredictionService initialization with valid model."""
        # Create a mock model checkpoint
        model = MockModel(num_users=100, num_items=50)
        user2idx = {i: i for i in range(100)}
        item2idx = {i: i for i in range(50)}

        checkpoint = {
            "model_type": "mock",
            "num_users": 100,
            "num_items": 50,
            "hyperparams": {},
            "model_state_dict": model.state_dict(),
            "user2idx": user2idx,
            "item2idx": item2idx,
        }

        model_path = tmp_path / "model.pt"
        torch.save(checkpoint, model_path)

        # Mock the ModelFactory and PredictorFactory
        import api.domain.predictor_factory as factory_module
        from src.recommender.models.factory import ModelFactory

        original_model_create = ModelFactory.create
        original_predictor_create = factory_module.PredictorFactory.create

        def mock_model_create(model_type, num_users, num_items, **hyperparams):
            return model

        def mock_predictor_create(predictor_type, model, user2idx, item2idx, **kwargs):
            from api.domain.predictors import SingleUserPredictor

            return SingleUserPredictor(model, user2idx, item2idx)

        ModelFactory.create = mock_model_create
        factory_module.PredictorFactory.create = mock_predictor_create

        try:
            service = PredictionService(
                model_path=model_path,
                predictor_type="single_user",
                device="cpu",
                enable_monitoring=False,
            )

            assert service.model_path == model_path
            assert service.predictor_type == "single_user"
            assert service.device == "cpu"
            assert service._predictor is not None
        finally:
            ModelFactory.create = original_model_create
            factory_module.PredictorFactory.create = original_predictor_create

    def test_prediction_service_invalid_model_path(self, tmp_path):
        """Test PredictionService with invalid model path."""
        invalid_path = tmp_path / "nonexistent.pt"

        with pytest.raises(ModelLoadError):
            PredictionService(
                model_path=invalid_path,
                predictor_type="single_user",
                device="cpu",
                enable_monitoring=False,
            )

    def test_prediction_service_predict(self, tmp_path):
        """Test PredictionService predict method."""
        # Create a mock model checkpoint
        model = MockModel(num_users=100, num_items=50)
        user2idx = {i: i for i in range(100)}
        item2idx = {i: i for i in range(50)}

        checkpoint = {
            "model_type": "mock",
            "num_users": 100,
            "num_items": 50,
            "hyperparams": {},
            "model_state_dict": model.state_dict(),
            "user2idx": user2idx,
            "item2idx": item2idx,
        }

        model_path = tmp_path / "model.pt"
        torch.save(checkpoint, model_path)

        # Mock the ModelFactory and PredictorFactory
        import api.domain.predictor_factory as factory_module
        from src.recommender.models.factory import ModelFactory

        original_model_create = ModelFactory.create
        original_predictor_create = factory_module.PredictorFactory.create

        def mock_model_create(model_type, num_users, num_items, **hyperparams):
            return model

        def mock_predictor_create(predictor_type, model, user2idx, item2idx, **kwargs):
            from api.domain.predictors import SingleUserPredictor

            return SingleUserPredictor(model, user2idx, item2idx)

        ModelFactory.create = mock_model_create
        factory_module.PredictorFactory.create = mock_predictor_create

        try:
            service = PredictionService(
                model_path=model_path,
                predictor_type="single_user",
                device="cpu",
                enable_monitoring=False,
            )

            # Test prediction
            request = PredictionRequest(user_id=5, item_ids=[1, 2, 3])
            response = service.predict(request)

            assert response.user_id == 5
            assert len(response.item_scores) == 3
        finally:
            ModelFactory.create = original_model_create
            factory_module.PredictorFactory.create = original_predictor_create

    def test_prediction_service_predict_invalid_user(self, tmp_path):
        """Test PredictionService with invalid user ID."""
        # Create a mock model checkpoint
        model = MockModel(num_users=100, num_items=50)
        user2idx = {i: i for i in range(100)}
        item2idx = {i: i for i in range(50)}

        checkpoint = {
            "model_type": "mock",
            "num_users": 100,
            "num_items": 50,
            "hyperparams": {},
            "model_state_dict": model.state_dict(),
            "user2idx": user2idx,
            "item2idx": item2idx,
        }

        model_path = tmp_path / "model.pt"
        torch.save(checkpoint, model_path)

        # Mock the ModelFactory and PredictorFactory
        import api.domain.predictor_factory as factory_module
        from src.recommender.models.factory import ModelFactory

        original_model_create = ModelFactory.create
        original_predictor_create = factory_module.PredictorFactory.create

        def mock_model_create(model_type, num_users, num_items, **hyperparams):
            return model

        def mock_predictor_create(predictor_type, model, user2idx, item2idx, **kwargs):
            from api.domain.predictors import SingleUserPredictor

            return SingleUserPredictor(model, user2idx, item2idx)

        ModelFactory.create = mock_model_create
        factory_module.PredictorFactory.create = mock_predictor_create

        try:
            service = PredictionService(
                model_path=model_path,
                predictor_type="single_user",
                device="cpu",
                enable_monitoring=False,
            )

            # Test prediction with invalid user
            request = PredictionRequest(user_id=999, item_ids=[1, 2, 3])
            with pytest.raises(InvalidInputError):
                service.predict(request)
        finally:
            ModelFactory.create = original_model_create
            factory_module.PredictorFactory.create = original_predictor_create

    def test_prediction_service_predict_batch(self, tmp_path):
        """Test PredictionService predict_batch method."""
        # Create a mock model checkpoint
        model = MockModel(num_users=100, num_items=50)
        user2idx = {i: i for i in range(100)}
        item2idx = {i: i for i in range(50)}

        checkpoint = {
            "model_type": "mock",
            "num_users": 100,
            "num_items": 50,
            "hyperparams": {},
            "model_state_dict": model.state_dict(),
            "user2idx": user2idx,
            "item2idx": item2idx,
        }

        model_path = tmp_path / "model.pt"
        torch.save(checkpoint, model_path)

        # Mock the ModelFactory and PredictorFactory
        import api.domain.predictor_factory as factory_module
        from src.recommender.models.factory import ModelFactory

        original_model_create = ModelFactory.create
        original_predictor_create = factory_module.PredictorFactory.create

        def mock_model_create(model_type, num_users, num_items, **hyperparams):
            return model

        def mock_predictor_create(predictor_type, model, user2idx, item2idx, **kwargs):
            from api.domain.predictors import BatchPredictor

            return BatchPredictor(model, user2idx, item2idx)

        ModelFactory.create = mock_model_create
        factory_module.PredictorFactory.create = mock_predictor_create

        try:
            service = PredictionService(
                model_path=model_path,
                predictor_type="batch",
                device="cpu",
                enable_monitoring=False,
            )

            # Test batch prediction
            requests = [
                PredictionRequest(user_id=5, item_ids=[1, 2, 3]),
                PredictionRequest(user_id=10, item_ids=[4, 5, 6]),
            ]
            response = service.predict_batch(requests)

            assert len(response.predictions) == 2
            assert response.predictions[0].user_id == 5
            assert response.predictions[1].user_id == 10
        finally:
            ModelFactory.create = original_model_create
            factory_module.PredictorFactory.create = original_predictor_create

    def test_prediction_service_recommend(self, tmp_path):
        """Test PredictionService recommend method."""
        # Create a mock model checkpoint
        model = MockModel(num_users=100, num_items=50)
        user2idx = {i: i for i in range(100)}
        item2idx = {i: i for i in range(50)}

        checkpoint = {
            "model_type": "mock",
            "num_users": 100,
            "num_items": 50,
            "hyperparams": {},
            "model_state_dict": model.state_dict(),
            "user2idx": user2idx,
            "item2idx": item2idx,
        }

        model_path = tmp_path / "model.pt"
        torch.save(checkpoint, model_path)

        # Mock the ModelFactory and PredictorFactory
        import api.domain.predictor_factory as factory_module
        from src.recommender.models.factory import ModelFactory

        original_model_create = ModelFactory.create
        original_predictor_create = factory_module.PredictorFactory.create

        def mock_model_create(model_type, num_users, num_items, **hyperparams):
            return model

        def mock_predictor_create(predictor_type, model, user2idx, item2idx, **kwargs):
            from api.domain.predictors import TopKRecommendationPredictor

            return TopKRecommendationPredictor(model, user2idx, item2idx)

        ModelFactory.create = mock_model_create
        factory_module.PredictorFactory.create = mock_predictor_create

        try:
            service = PredictionService(
                model_path=model_path,
                predictor_type="top_k",
                device="cpu",
                enable_monitoring=False,
            )

            # Test recommendation
            response = service.recommend(user_id=5, k=10)

            assert response.user_id == 5
            assert len(response.recommendations) == 10
        finally:
            ModelFactory.create = original_model_create
            factory_module.PredictorFactory.create = original_predictor_create

    def test_prediction_service_with_monitoring_enabled(self, tmp_path):
        """Test PredictionService with monitoring enabled."""
        # Create a mock model checkpoint
        model = MockModel(num_users=100, num_items=50)
        user2idx = {i: i for i in range(100)}
        item2idx = {i: i for i in range(50)}

        checkpoint = {
            "model_type": "mock",
            "num_users": 100,
            "num_items": 50,
            "hyperparams": {},
            "model_state_dict": model.state_dict(),
            "user2idx": user2idx,
            "item2idx": item2idx,
        }

        model_path = tmp_path / "model.pt"
        torch.save(checkpoint, model_path)

        # Mock the ModelFactory and PredictorFactory
        import api.domain.predictor_factory as factory_module
        from src.recommender.models.factory import ModelFactory

        original_model_create = ModelFactory.create
        original_predictor_create = factory_module.PredictorFactory.create

        def mock_model_create(model_type, num_users, num_items, **hyperparams):
            return model

        def mock_predictor_create(predictor_type, model, user2idx, item2idx, **kwargs):
            from api.domain.predictors import SingleUserPredictor

            return SingleUserPredictor(model, user2idx, item2idx)

        ModelFactory.create = mock_model_create
        factory_module.PredictorFactory.create = mock_predictor_create

        try:
            service = PredictionService(
                model_path=model_path,
                predictor_type="single_user",
                device="cpu",
                enable_monitoring=True,
            )

            assert service._monitoring_service is not None
        finally:
            ModelFactory.create = original_model_create
            factory_module.PredictorFactory.create = original_predictor_create
