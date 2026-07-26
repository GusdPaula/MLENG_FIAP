"""Tests for the MLflow toolkit wrapper."""

from __future__ import annotations

import sys
import types
from typing import Any

import pandas as pd
import pytest
from src.recommender.mlflow_toolkit import MLflowToolkit


def _install_dummy_mlflow(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:  # noqa: C901 - local test double with many stubs
    calls: dict[str, list] = {
        "set_tracking_uri": [],
        "set_registry_uri": [],
        "set_experiment": [],
        "get_experiment_by_name": [],
        "create_experiment": [],
        "start_run": [],
        "log_params": [],
        "log_metrics": [],
        "log_artifact": [],
        "log_input": [],
        "set_tag": [],
        "register_model": [],
    }

    class DummyRun:
        info = types.SimpleNamespace(run_id="run-123")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def set_tracking_uri(uri):
        calls["set_tracking_uri"].append(uri)

    def set_registry_uri(uri):
        calls["set_registry_uri"].append(uri)

    def set_experiment(name):
        calls["set_experiment"].append(name)

    def get_experiment_by_name(name):
        calls["get_experiment_by_name"].append(name)
        return None

    def create_experiment(name):
        calls["create_experiment"].append(name)
        return "exp-1"

    def start_run(run_name=None, tags=None, nested=False):
        calls["start_run"].append({"run_name": run_name, "tags": tags, "nested": nested})
        return DummyRun()

    def log_params(params):
        calls["log_params"].append(params)

    def log_metrics(metrics, step=None):
        calls["log_metrics"].append({"metrics": metrics, "step": step})

    def log_artifact(path):
        calls["log_artifact"].append(path)

    def log_input(dataset, context=None):
        calls["log_input"].append({"dataset": dataset, "context": context})

    def set_tag(key, value):
        calls["set_tag"].append((key, value))

    def register_model(model_uri, name):
        calls["register_model"].append((model_uri, name))
        return types.SimpleNamespace(name=name, model_uri=model_uri)

    dummy_mlflow = types.SimpleNamespace(
        set_tracking_uri=set_tracking_uri,
        set_registry_uri=set_registry_uri,
        set_experiment=set_experiment,
        get_experiment_by_name=get_experiment_by_name,
        create_experiment=create_experiment,
        start_run=start_run,
        log_params=log_params,
        log_metrics=log_metrics,
        log_artifact=log_artifact,
        log_input=log_input,
        set_tag=set_tag,
        register_model=register_model,
        data=types.SimpleNamespace(from_pandas=lambda df, name=None: {"df": df, "name": name}),
        pytorch=types.SimpleNamespace(log_model=lambda **kwargs: None),
    )
    monkeypatch.setitem(sys.modules, "mlflow", dummy_mlflow)
    return calls


def _install_failing_mlflow(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    calls = _install_dummy_mlflow(monkeypatch)
    state = {"tracking_uri": None}

    def set_tracking_uri(uri):
        state["tracking_uri"] = uri
        calls["set_tracking_uri"].append(uri)

    def set_experiment(name):
        calls["set_experiment"].append(name)
        if state["tracking_uri"] and str(state["tracking_uri"]).startswith("http"):
            raise ConnectionError("mlflow server unavailable")

    monkeypatch.setitem(
        sys.modules,
        "mlflow",
        types.SimpleNamespace(
            set_tracking_uri=set_tracking_uri,
            set_registry_uri=lambda uri: calls["set_registry_uri"].append(uri),
            set_experiment=set_experiment,
            get_experiment_by_name=lambda name: None,
            create_experiment=lambda name: "exp-offline",
            start_run=lambda run_name=None, tags=None, nested=False: types.SimpleNamespace(
                __enter__=lambda self=None: self,
                __exit__=lambda self, exc_type, exc, tb: False,
                info=types.SimpleNamespace(run_id="run-offline"),
            ),
            log_params=lambda params: calls["log_params"].append(params),
            log_metrics=lambda metrics, step=None: calls["log_metrics"].append({"metrics": metrics, "step": step}),
            log_artifact=lambda path: calls["log_artifact"].append(path),
            log_input=lambda dataset, context=None: calls["log_input"].append({"dataset": dataset, "context": context}),
            set_tag=lambda key, value: calls["set_tag"].append((key, value)),
            register_model=lambda model_uri, name: types.SimpleNamespace(name=name, model_uri=model_uri),
            data=types.SimpleNamespace(from_pandas=lambda df, name=None: {"df": df, "name": name}),
            pytorch=types.SimpleNamespace(log_model=lambda **kwargs: None),
        ),
    )
    return calls


def test_setup_and_experiment_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_dummy_mlflow(monkeypatch)

    # Mock config to return None so tracking_uri parameter is used
    dummy_settings = types.SimpleNamespace(mlflow_tracking_uri=None)
    monkeypatch.setitem(
        sys.modules,
        "src.recommender.config",
        types.SimpleNamespace(get_settings=lambda: dummy_settings),
    )

    toolkit = MLflowToolkit(
        tracking_uri="http://localhost:5000",
        experiment_name="demo-experiment",
        registry_uri="http://localhost:5000",
    )

    assert toolkit.setup() == "demo-experiment"
    assert toolkit.get_experiment_id() == "exp-1"
    assert calls["set_tracking_uri"] == ["http://localhost:5000"]
    assert calls["set_registry_uri"] == ["http://localhost:5000"]
    assert calls["set_experiment"] == ["demo-experiment"]
    assert calls["create_experiment"] == ["demo-experiment"]


def test_logging_helpers_and_model_registration(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_dummy_mlflow(monkeypatch)
    toolkit = MLflowToolkit(experiment_name="demo-experiment")

    with toolkit.start_run(run_name="run-a", tags={"env": "test"}):
        toolkit.log_params({"epochs": 3})
        toolkit.log_metrics({"auc_roc": 0.9}, step=1)
        toolkit.log_dataset(
            pd.DataFrame({"user": [1], "item": [10]}),
            name="interactions",
            source="unit-test",
            context="training",
        )
        registered = toolkit.register_model("runs:/run-123/model", "demo-model")

    assert calls["start_run"][0]["run_name"] == "run-a"
    assert calls["log_params"] == [{"epochs": 3}]
    assert calls["log_metrics"] == [{"metrics": {"auc_roc": 0.9}, "step": 1}]
    assert calls["log_input"][0]["context"] == "training"
    assert calls["register_model"] == [("runs:/run-123/model", "demo-model")]
    assert registered.name == "demo-model"


def test_setup_falls_back_to_local_store(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_failing_mlflow(monkeypatch)

    # Mock config to return None so tracking_uri parameter is used
    dummy_settings = types.SimpleNamespace(mlflow_tracking_uri=None)
    monkeypatch.setitem(
        sys.modules,
        "src.recommender.config",
        types.SimpleNamespace(get_settings=lambda: dummy_settings),
    )

    toolkit = MLflowToolkit(
        tracking_uri="http://localhost:5000",
        experiment_name="demo-experiment",
        offline_tracking_db="mlflow-test.db",
    )

    assert toolkit.setup() == "demo-experiment"
    assert toolkit.is_offline is True
    assert calls["set_tracking_uri"][0] == "http://localhost:5000"
    assert calls["set_tracking_uri"][-1] == "sqlite:///mlflow-test.db"
    assert calls["create_experiment"] == []


def test_get_model_version_by_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dummy_mlflow(monkeypatch)

    class DummyVersion:
        def __init__(self, run_id, version):
            self.run_id = run_id
            self.version = version

    class DummyClient:
        def search_model_versions(self, filter_string):
            return [DummyVersion("run-123", "1"), DummyVersion("run-456", "2")]

    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda: DummyClient())

    toolkit = MLflowToolkit()
    version = toolkit.get_model_version_by_run_id("model", "run-456")
    assert version == "2"

    version_none = toolkit.get_model_version_by_run_id("model", "run-789")
    assert version_none is None


def test_set_model_version_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dummy_mlflow(monkeypatch)

    calls = []

    class DummyClient:
        def set_registered_model_alias(self, name, alias, version):
            calls.append((name, alias, version))

    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda: DummyClient())

    toolkit = MLflowToolkit()
    toolkit.set_model_version_alias("model", "1", "staging")
    assert calls == [("model", "staging", "1")]


def test_get_version_by_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dummy_mlflow(monkeypatch)

    class DummyClient:
        def get_model_version_by_alias(self, name, alias):
            if alias == "staging":
                return types.SimpleNamespace(version="1", run_id="run-123")
            raise Exception("not found")

    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda: DummyClient())

    toolkit = MLflowToolkit()
    version = toolkit.get_version_by_alias("model", "staging")
    assert version.version == "1"

    version_none = toolkit.get_version_by_alias("model", "prod")
    assert version_none is None


def test_promote_best_to_staging(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dummy_mlflow(monkeypatch)

    class DummyVersion:
        def __init__(self, run_id, version):
            self.run_id = run_id
            self.version = version

    class DummyRun:
        def __init__(self, run_id, metric_value):
            self.data = types.SimpleNamespace(metrics={"auc_roc": metric_value})

    class DummyClient:
        def search_model_versions(self, filter_string):
            return [DummyVersion("new-run", "2")]

        def get_model_version_by_alias(self, name, alias):
            if alias == "staging":
                return DummyVersion("old-run", "1")
            raise Exception("not found")

        def get_run(self, run_id):
            if run_id == "new-run":
                return DummyRun("new-run", 0.95)
            if run_id == "old-run":
                return DummyRun("old-run", 0.90)

        def set_registered_model_alias(self, name, alias, version):
            self.promoted = version

    dummy_client = DummyClient()
    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda: dummy_client)

    toolkit = MLflowToolkit(experiment_name="test_exp")

    # Test promotion (new is better)
    promoted = toolkit.promote_best_to_staging("model", "new-run", "auc_roc", higher_is_better=True)
    assert promoted is True
    assert getattr(dummy_client, "promoted", None) == "2"


def test_promote_best_to_staging_no_staging(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dummy_mlflow(monkeypatch)

    class DummyVersion:
        def __init__(self, run_id, version):
            self.run_id = run_id
            self.version = version

    class DummyClient:
        def search_model_versions(self, filter_string):
            return [DummyVersion("new-run", "1")]

        def get_model_version_by_alias(self, name, alias):
            raise Exception("not found")

        def set_registered_model_alias(self, name, alias, version):
            self.promoted = version

    dummy_client = DummyClient()
    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda: dummy_client)

    toolkit = MLflowToolkit(experiment_name="test_exp")
    promoted = toolkit.promote_best_to_staging("model", "new-run", "auc_roc")
    assert promoted is True
    assert getattr(dummy_client, "promoted", None) == "1"


def test_promote_best_to_staging_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    toolkit = MLflowToolkit(experiment_name="test_exp")
    toolkit._is_offline = True
    promoted = toolkit.promote_best_to_staging("model", "new-run", "auc_roc")
    assert promoted is False
