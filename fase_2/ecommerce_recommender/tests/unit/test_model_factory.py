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


def test_factory_registers_builtin_models() -> None:
    available = ModelFactory.available_models()
    assert "ncf" in available
    assert "gmf" in available
    assert "matrix_factorization" in available


def test_factory_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model type"):
        ModelFactory.create("does_not_exist", num_users=10, num_items=5)


def test_factory_returns_base_instance() -> None:
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
def test_factory_models_forward_shape(model_type: str, extra: dict) -> None:
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
def test_factory_models_output_in_range(model_type: str, extra: dict) -> None:
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


def test_each_model_exposes_model_name() -> None:
    assert NCFModel(num_users=2, num_items=2, embedding_dim=4).model_name == "ncf"
    assert GMFModel(num_users=2, num_items=2, embedding_dim=4).model_name == "gmf"
    assert (
        MatrixFactorizationModel(num_users=2, num_items=2, embedding_dim=4).model_name
        == "matrix_factorization"
    )


def test_factory_register_decorator() -> None:
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


# -- parameter filtering tests ----------------------------------------------


def test_factory_filters_invalid_params_gmf() -> None:
    """GMF should reject hidden_layers parameter."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = ModelFactory.create(
            "gmf",
            num_users=10,
            num_items=5,
            embedding_dim=8,
            hidden_layers=[32, 16],  # Invalid for GMF
        )
        # Should create model successfully
        assert isinstance(model, GMFModel)
        # Should warn about filtered parameter
        assert len(w) == 1
        assert "hidden_layers" in str(w[0].message)


def test_factory_filters_invalid_params_ncf() -> None:
    """NCF should reject projection_dim parameter."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = ModelFactory.create(
            "ncf",
            num_users=10,
            num_items=5,
            embedding_dim=8,
            projection_dim=16,  # Invalid for NCF
        )
        # Should create model successfully
        assert isinstance(model, NCFModel)
        # Should warn about filtered parameter
        assert len(w) == 1
        assert "projection_dim" in str(w[0].message)


def test_factory_filters_invalid_params_mf() -> None:
    """MatrixFactorization should reject hidden_layers and dropout parameters."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = ModelFactory.create(
            "matrix_factorization",
            num_users=10,
            num_items=5,
            embedding_dim=8,
            hidden_layers=[32, 16],  # Invalid for MF
            dropout=0.2,  # Invalid for MF
        )
        # Should create model successfully
        assert isinstance(model, MatrixFactorizationModel)
        # Should warn about filtered parameters
        assert len(w) == 1
        assert "hidden_layers" in str(w[0].message)
        assert "dropout" in str(w[0].message)


def test_factory_accepts_valid_params_gmf() -> None:
    """GMF should accept its valid parameters."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = ModelFactory.create(
            "gmf",
            num_users=10,
            num_items=5,
            embedding_dim=8,
            projection_dim=16,
            dropout=0.2,
        )
        # Should create model successfully
        assert isinstance(model, GMFModel)
        # Should NOT warn about any parameters
        assert len(w) == 0


def test_factory_accepts_valid_params_ncf() -> None:
    """NCF should accept its valid parameters."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = ModelFactory.create(
            "ncf",
            num_users=10,
            num_items=5,
            embedding_dim=8,
            hidden_layers=[32, 16],
            dropout=0.2,
        )
        # Should create model successfully
        assert isinstance(model, NCFModel)
        # Should NOT warn about any parameters
        assert len(w) == 0


def test_factory_accepts_valid_params_mf() -> None:
    """MatrixFactorization should accept its valid parameters."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = ModelFactory.create(
            "matrix_factorization",
            num_users=10,
            num_items=5,
            embedding_dim=8,
            global_bias=0.1,
        )
        # Should create model successfully
        assert isinstance(model, MatrixFactorizationModel)
        # Should NOT warn about any parameters
        assert len(w) == 0


def test_factory_param_map_coverage() -> None:
    """Ensure all registered models have parameter mappings."""
    for model_type in ModelFactory.available_models():
        assert model_type in ModelFactory.MODEL_PARAM_MAP, (
            f"Model type '{model_type}' is registered but not in MODEL_PARAM_MAP"
        )
