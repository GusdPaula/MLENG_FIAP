"""Tests for the Model Factory and concrete models."""
import pytest
import torch
from src.recommender.models import (
    GMFModel,
    MatrixFactorizationModel,
    ModelFactory,
    NCFModel,
)
from src.recommender.models.base import BaseRecommenderModel

# -- factory registration -----------------------------------------------


def test_factory_registers_builtin_models():
    available = ModelFactory.available_models()
    assert "ncf" in available
    assert "gmf" in available
    assert "matrix_factorization" in available


def test_factory_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model type"):
        ModelFactory.create("does_not_exist", num_users=10, num_items=5)


def test_factory_returns_base_instance():
    model = ModelFactory.create("ncf", num_users=10, num_items=5, embedding_dim=8)
    assert isinstance(model, BaseRecommenderModel)


# -- model-specific shape & range tests --------------------------------


@pytest.mark.parametrize(
    "model_type, extra",
    [
        ("ncf", {"hidden_layers": [32, 16]}),
        ("gmf", {}),
        ("matrix_factorization", {}),
    ],
)
def test_factory_models_forward_shape(model_type, extra):
    model = ModelFactory.create(
        model_type,
        num_users=20,
        num_items=10,
        embedding_dim=8,
        **extra,
    )
    users = torch.randint(0, 20, (4,))
    items = torch.randint(0, 10, (4,))
    out = model(users, items)
    assert out.shape == (4,)


@pytest.mark.parametrize(
    "model_type, extra",
    [
        ("ncf", {"hidden_layers": [32, 16]}),
        ("gmf", {}),
        ("matrix_factorization", {}),
    ],
)
def test_factory_models_output_in_range(model_type, extra):
    model = ModelFactory.create(
        model_type,
        num_users=20,
        num_items=10,
        embedding_dim=8,
        **extra,
    )
    users = torch.randint(0, 20, (16,))
    items = torch.randint(0, 10, (16,))
    out = model(users, items)
    assert torch.all(out >= 0)
    assert torch.all(out <= 1)


def test_each_model_exposes_model_name():
    assert NCFModel(num_users=2, num_items=2, embedding_dim=4).model_name == "ncf"
    assert GMFModel(num_users=2, num_items=2, embedding_dim=4).model_name == "gmf"
    assert (
        MatrixFactorizationModel(num_users=2, num_items=2, embedding_dim=4).model_name
        == "matrix_factorization"
    )


def test_factory_register_decorator():
    @ModelFactory.register("dummy_for_test")
    class _Dummy(BaseRecommenderModel):
        @property
        def model_name(self) -> str:
            return "dummy_for_test"

        def forward(self, user_ids, item_ids):
            return torch.zeros(user_ids.shape[0])

    model = ModelFactory.create("dummy_for_test", num_users=3, num_items=3)
    assert isinstance(model, _Dummy)
    # Cleanup the registry so other tests aren't polluted.
    ModelFactory._registry.pop("dummy_for_test", None)
