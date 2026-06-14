"""Tests for the MLflow toolkit wrapper."""

from __future__ import annotations

import sys
import types

import pandas as pd
from src.recommender.mlflow_toolkit import MLflowToolkit


def _install_dummy_mlflow(monkeypatch):  # noqa: C901 - local test double with many stubs
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
        calls["start_run"].append(
            {"run_name": run_name, "tags": tags, "nested": nested}
        )
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
        data=types.SimpleNamespace(
            from_pandas=lambda df, name=None: {"df": df, "name": name}
        ),
        pytorch=types.SimpleNamespace(log_model=lambda **kwargs: None),
    )
    monkeypatch.setitem(sys.modules, "mlflow", dummy_mlflow)
    return calls


def _install_failing_mlflow(monkeypatch):
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
            start_run=lambda run_name=None, tags=None, nested=False: (
                types.SimpleNamespace(
                    __enter__=lambda self=None: self,
                    __exit__=lambda self, exc_type, exc, tb: False,
                    info=types.SimpleNamespace(run_id="run-offline"),
                )
            ),
            log_params=lambda params: calls["log_params"].append(params),
            log_metrics=lambda metrics, step=None: calls["log_metrics"].append(
                {"metrics": metrics, "step": step}
            ),
            log_artifact=lambda path: calls["log_artifact"].append(path),
            log_input=lambda dataset, context=None: calls["log_input"].append(
                {"dataset": dataset, "context": context}
            ),
            set_tag=lambda key, value: calls["set_tag"].append((key, value)),
            register_model=lambda model_uri, name: types.SimpleNamespace(
                name=name, model_uri=model_uri
            ),
            data=types.SimpleNamespace(
                from_pandas=lambda df, name=None: {"df": df, "name": name}
            ),
            pytorch=types.SimpleNamespace(log_model=lambda **kwargs: None),
        ),
    )
    return calls


def test_setup_and_experiment_creation(monkeypatch):
    calls = _install_dummy_mlflow(monkeypatch)
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


def test_logging_helpers_and_model_registration(tmp_path, monkeypatch):
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


def test_setup_falls_back_to_local_store(monkeypatch):
    calls = _install_failing_mlflow(monkeypatch)
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
