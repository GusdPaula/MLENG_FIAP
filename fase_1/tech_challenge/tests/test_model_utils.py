from unittest.mock import MagicMock, patch

import pytest

from src.api.model_utils import ModelManager


def test_load_from_local_path_disabled_when_env_is_missing(monkeypatch):
    monkeypatch.delenv("LOCAL_MODEL_PATH", raising=False)
    manager = ModelManager()

    assert manager._load_from_local_path() is False


def test_load_from_local_path_returns_false_when_path_does_not_exist(monkeypatch):
    monkeypatch.setenv("LOCAL_MODEL_PATH", "/tmp/path-that-does-not-exist-12345")
    manager = ModelManager()

    assert manager._load_from_local_path() is False


def test_load_from_local_path_success(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    monkeypatch.setenv("LOCAL_MODEL_PATH", str(model_dir))
    manager = ModelManager()

    fake_pipeline = MagicMock()
    with patch("src.api.model_utils.mlflow.sklearn.load_model", return_value=fake_pipeline) as mocked_load:
        assert manager._load_from_local_path() is True
        mocked_load.assert_called_once_with(str(model_dir))
        assert manager.pipeline is fake_pipeline


def test_load_from_local_path_handles_load_errors(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    monkeypatch.setenv("LOCAL_MODEL_PATH", str(model_dir))
    manager = ModelManager()

    with patch("src.api.model_utils.mlflow.sklearn.load_model", side_effect=RuntimeError("boom")):
        assert manager._load_from_local_path() is False


def test_is_mlflow_ready_returns_true_for_http_200():
    manager = ModelManager()

    with patch("src.api.model_utils.requests.get") as mocked_get:
        mocked_get.return_value.status_code = 200
        assert manager.is_mlflow_ready() is True


def test_is_mlflow_ready_returns_false_for_non_200():
    manager = ModelManager()

    with patch("src.api.model_utils.requests.get") as mocked_get:
        mocked_get.return_value.status_code = 503
        assert manager.is_mlflow_ready() is False


def test_is_mlflow_ready_returns_false_on_exception():
    manager = ModelManager()

    with patch("src.api.model_utils.requests.get", side_effect=Exception("network")):
        assert manager.is_mlflow_ready() is False


def test_load_from_mlflow_prefers_local_path():
    manager = ModelManager()

    with patch.object(manager, "_load_from_local_path", return_value=True), patch.object(
        manager, "is_mlflow_ready"
    ) as mocked_ready:
        assert manager.load_from_mlflow() is True
        mocked_ready.assert_not_called()


def test_load_from_mlflow_returns_false_when_mlflow_is_unavailable():
    manager = ModelManager()

    with patch.object(manager, "_load_from_local_path", return_value=False), patch.object(
        manager, "is_mlflow_ready", return_value=False
    ):
        assert manager.load_from_mlflow() is False


def test_load_from_mlflow_remote_success_sets_pipeline():
    manager = ModelManager()
    fake_pipeline = MagicMock()

    with patch.object(manager, "_load_from_local_path", return_value=False), patch.object(
        manager, "is_mlflow_ready", return_value=True
    ), patch("src.api.model_utils.mlflow.set_tracking_uri") as mocked_set_uri, patch(
        "src.api.model_utils.mlflow.sklearn.load_model", return_value=fake_pipeline
    ) as mocked_load:
        assert manager.load_from_mlflow() is True
        mocked_set_uri.assert_called_once_with(manager.tracking_uri)
        mocked_load.assert_called_once_with(f"models:/{manager.model_name}@{manager.alias}")
        assert manager.pipeline is fake_pipeline


def test_load_from_mlflow_remote_failure_returns_false():
    manager = ModelManager()

    with patch.object(manager, "_load_from_local_path", return_value=False), patch.object(
        manager, "is_mlflow_ready", return_value=True
    ), patch("src.api.model_utils.mlflow.set_tracking_uri"), patch(
        "src.api.model_utils.mlflow.sklearn.load_model", side_effect=RuntimeError("load failed")
    ):
        assert manager.load_from_mlflow() is False


def test_predict_raises_when_model_is_not_loaded():
    manager = ModelManager()

    with pytest.raises(RuntimeError, match="Model not loaded"):
        manager.predict([1, 2, 3])


def test_predict_delegates_to_pipeline_predict():
    manager = ModelManager()
    manager.pipeline = MagicMock()
    manager.pipeline.predict.return_value = [0]

    result = manager.predict([1, 2, 3])

    manager.pipeline.predict.assert_called_once_with([1, 2, 3])
    assert result == [0]
