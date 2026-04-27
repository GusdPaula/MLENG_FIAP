import pickle
from pathlib import Path

import numpy as np
import pytest

from src.models.inference import ModelRegistry, PredictionService

# -----------------------------
# Fixtures & Mocks
# -----------------------------


class MockPipeline:
    def predict(self, X):
        return np.array([1] * len(X))

    def predict_proba(self, X):
        # Always return [prob_0, prob_1]
        return np.array([[0.2, 0.8]] * len(X))


class MockPreprocessor:
    def transform(self, X):
        return X * 2


class MockTelcoPipeline:
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"model_object": MockPipeline()}, f)


@pytest.fixture
def temp_pipeline_file(tmp_path):
    path = tmp_path / "pipeline.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model_object": MockPipeline()}, f)
    return path


@pytest.fixture
def temp_preprocessor_file(tmp_path):
    path = tmp_path / "preprocessor.pkl"
    with open(path, "wb") as f:
        pickle.dump(MockPreprocessor(), f)
    return path


# -----------------------------
# PredictionService Tests
# -----------------------------


def test_init_loads_pipeline(temp_pipeline_file):
    service = PredictionService(str(temp_pipeline_file))
    assert service.pipeline is not None


def test_init_raises_if_pipeline_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        PredictionService(str(tmp_path / "missing.pkl"))


def test_preprocess_applies_preprocessor(temp_pipeline_file, temp_preprocessor_file):
    service = PredictionService(
        str(temp_pipeline_file),
        preprocessor_path=str(temp_preprocessor_file),
    )

    X = np.array([[1, 2]])
    result = service.preprocess(X)

    assert np.array_equal(result, X * 2)


def test_predict_without_proba(temp_pipeline_file):
    service = PredictionService(str(temp_pipeline_file))

    X = np.array([[1, 2], [3, 4]])
    preds = service.predict(X)

    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == 2
    assert all(preds == 1)


def test_predict_with_proba(temp_pipeline_file):
    service = PredictionService(str(temp_pipeline_file))

    X = np.array([[1, 2]])
    result = service.predict(X, return_proba=True)

    assert "predictions" in result
    assert "probabilities" in result
    assert "confidence" in result

    assert result["predictions"][0] == 1
    assert result["probabilities"][0] == 0.8
    assert result["confidence"][0] == 0.8


def test_predict_batch(temp_pipeline_file):
    service = PredictionService(str(temp_pipeline_file))

    X = np.array([[i, i] for i in range(10)])
    preds = service.predict_batch(X, batch_size=3)

    assert len(preds) == 10
    assert all(preds == 1)


def test_predict_single(temp_pipeline_file):
    service = PredictionService(str(temp_pipeline_file))

    features = {"a": 1, "b": 2}
    result = service.predict_single(features)

    assert "churn_prediction" in result
    assert "churn_probability" in result
    assert "confidence" in result

    assert result["churn_prediction"] == 1
    assert result["churn_probability"] == 0.8


# -----------------------------
# ModelRegistry Tests
# -----------------------------


def test_register_model(tmp_path):
    registry = ModelRegistry(registry_dir=tmp_path)

    model = MockTelcoPipeline()
    metadata = {"accuracy": 0.9}

    model_dir = registry.register_model(
        model_name="test_model",
        version="1.0",
        pipeline=model,
        metadata=metadata,
    )

    assert Path(model_dir).exists()
    assert (Path(model_dir) / "pipeline.pkl").exists()
    assert (Path(model_dir) / "metadata.pkl").exists()


def test_load_model(tmp_path):
    registry = ModelRegistry(registry_dir=tmp_path)

    model = MockTelcoPipeline()
    metadata = {"accuracy": 0.9}

    registry.register_model("test_model", "1.0", model, metadata)

    service = registry.load_model("test_model", "1.0")

    assert isinstance(service, PredictionService)


def test_load_model_not_found(tmp_path):
    registry = ModelRegistry(registry_dir=tmp_path)

    with pytest.raises(FileNotFoundError):
        registry.load_model("unknown", "1.0")


def test_list_models(tmp_path):
    registry = ModelRegistry(registry_dir=tmp_path)

    model = MockTelcoPipeline()

    registry.register_model("model_a", "1.0", model, {})
    registry.register_model("model_a", "2.0", model, {})
    registry.register_model("model_b", "1.0", model, {})

    models = registry.list_models()

    assert "model_a" in models
    assert "model_b" in models
    assert set(models["model_a"]) == {"1.0", "2.0"}
