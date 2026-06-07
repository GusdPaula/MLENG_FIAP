import subprocess
import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def load_module_from_path(module_name: str, path: Path):
    spec = spec_from_file_location(module_name, str(path))
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class DummyRow(dict):
    def __getitem__(self, item):
        return super().__getitem__(item)


class DummyQueryResult:
    def __init__(self, schema, rows):
        self.schema = [types.SimpleNamespace(name=name) for name in schema]
        self._rows = [DummyRow(row) for row in rows]
        self.total_rows = len(rows)

    def __iter__(self):
        return iter(self._rows)


class DummyQueryJob:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class DummyClient:
    def __init__(self, project: str | None = None) -> None:
        self.project = project
        self.queries = []

    def query(self, query_text: str):
        self.queries.append(query_text)
        schema = ["event_id", "user_id"]
        rows = [{"event_id": 1, "user_id": 100}, {"event_id": 2, "user_id": 101}]
        return DummyQueryJob(DummyQueryResult(schema, rows))


def test_bigquery_query_extract_table_and_versions(tmp_path, monkeypatch):
    google_module = types.ModuleType("google")
    cloud_module = types.ModuleType("google.cloud")
    bigquery_module = types.ModuleType("google.cloud.bigquery")

    bigquery_module.Client = DummyClient

    google_module.__path__ = ["."]
    cloud_module.__path__ = ["."]
    cloud_module.bigquery = bigquery_module

    sys.modules["google"] = google_module
    sys.modules["google.cloud"] = cloud_module
    sys.modules["google.cloud.bigquery"] = bigquery_module

    module_path = Path(__file__).resolve().parents[2] / "data_pipeline" / "bigquery_query.py"
    query_module = load_module_from_path("bigquery_query", module_path)

    dvc_calls = []

    def fake_run(args, cwd, capture_output, text, check):
        dvc_calls.append((args, cwd))
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="added", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    query_client = query_module.BigQueryQuery(
        project_id="test-project",
        dataset_id="test_dataset",
        output_dir=tmp_path / "output",
        dvc_repo_path=tmp_path,
    )

    result_path = query_client.extract_table("events", destination_name="events_export.csv")

    assert result_path.exists()
    assert result_path.name == "events_export.csv"
    assert dvc_calls == [(["dvc", "add", str(result_path.relative_to(tmp_path))], str(tmp_path))]

    csv_lines = result_path.read_text().splitlines()
    assert csv_lines[0] == "event_id,user_id"
    assert csv_lines[1] == "1,100"
    assert csv_lines[2] == "2,101"


def test_bigquery_query_skips_existing_file(tmp_path, monkeypatch):
    google_module = types.ModuleType("google")
    cloud_module = types.ModuleType("google.cloud")
    bigquery_module = types.ModuleType("google.cloud.bigquery")

    bigquery_module.Client = DummyClient

    google_module.__path__ = ["."]
    cloud_module.__path__ = ["."]
    cloud_module.bigquery = bigquery_module

    sys.modules["google"] = google_module
    sys.modules["google.cloud"] = cloud_module
    sys.modules["google.cloud.bigquery"] = bigquery_module

    module_path = Path(__file__).resolve().parents[2] / "data_pipeline" / "bigquery_query.py"
    query_module = load_module_from_path("bigquery_query", module_path)

    dvc_calls = []

    def fake_run(args, cwd, capture_output, text, check):
        dvc_calls.append((args, cwd))
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="added", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    existing_path = tmp_path / "output" / "events_export.csv"
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    existing_path.write_text("event_id,user_id\n1,100\n")

    query_client = query_module.BigQueryQuery(
        project_id="test-project",
        dataset_id="test_dataset",
        output_dir=tmp_path / "output",
        dvc_repo_path=tmp_path,
    )

    result_path = query_client.extract_table("events", destination_name="events_export.csv")

    assert result_path == existing_path
    assert dvc_calls == []
    assert result_path.read_text() == "event_id,user_id\n1,100\n"
