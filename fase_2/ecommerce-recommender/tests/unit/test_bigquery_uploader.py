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


class DummyLoadJob:
    def __init__(self) -> None:
        self.completed = False

    def result(self) -> None:
        self.completed = True


class DummyClient:
    def __init__(self, project: str | None = None) -> None:
        self.project = project
        self.dataset_id = None
        self.table_name = None
        self.created_datasets = []
        self.loaded_files = []

    def dataset(self, dataset_id: str):
        self.dataset_id = dataset_id
        return self

    def table(self, table_name: str):
        self.table_name = table_name
        return f"{self.dataset_id}.{table_name}"

    def create_dataset(self, dataset_ref, exists_ok=True):
        self.created_datasets.append(dataset_ref)

    def load_table_from_file(self, source_file, table_ref, job_config=None):
        self.loaded_files.append((source_file, table_ref, job_config))
        return DummyLoadJob()


def test_bigquery_uploader_upload_csv(tmp_path):
    google_module = types.ModuleType("google")
    cloud_module = types.ModuleType("google.cloud")
    bigquery_module = types.ModuleType("google.cloud.bigquery")

    def fake_dataset(reference: str):
        dataset = types.SimpleNamespace(location=None)
        dataset.reference = reference
        return dataset

    bigquery_module.Client = DummyClient
    bigquery_module.Dataset = fake_dataset
    bigquery_module.LoadJobConfig = lambda autodetect, skip_leading_rows, source_format: types.SimpleNamespace(
        autodetect=autodetect,
        skip_leading_rows=skip_leading_rows,
        source_format=source_format,
    )
    bigquery_module.SourceFormat = types.SimpleNamespace(CSV="CSV")

    google_module.__path__ = ["."]
    cloud_module.__path__ = ["."]
    cloud_module.bigquery = bigquery_module

    sys.modules["google"] = google_module
    sys.modules["google.cloud"] = cloud_module
    sys.modules["google.cloud.bigquery"] = bigquery_module

    module_path = Path(__file__).resolve().parents[2] / "data-pipeline" / "bigquery_uploader.py"
    uploader_module = load_module_from_path("bigquery_uploader", module_path)

    source_file = tmp_path / "events.csv"
    source_file.write_text("event_id,user_id\n1,100\n")

    uploader = uploader_module.BigQueryUploader(project_id="test-project", dataset_id="test_dataset")
    tables = uploader.upload_files({"events": source_file})

    assert "events" in tables
    assert tables["events"] == "test-project.test_dataset.events"
    assert uploader.client.created_datasets
    assert uploader.client.loaded_files


def test_bigquery_uploader_missing_file(tmp_path):
    google_module = types.ModuleType("google")
    cloud_module = types.ModuleType("google.cloud")
    bigquery_module = types.ModuleType("google.cloud.bigquery")

    bigquery_module.Client = DummyClient
    bigquery_module.Dataset = lambda reference: types.SimpleNamespace(location=None)
    bigquery_module.LoadJobConfig = lambda autodetect, skip_leading_rows, source_format: types.SimpleNamespace(
        autodetect=autodetect,
        skip_leading_rows=skip_leading_rows,
        source_format=source_format,
    )
    bigquery_module.SourceFormat = types.SimpleNamespace(CSV="CSV")

    google_module.__path__ = ["."]
    cloud_module.__path__ = ["."]
    cloud_module.bigquery = bigquery_module

    sys.modules["google"] = google_module
    sys.modules["google.cloud"] = cloud_module
    sys.modules["google.cloud.bigquery"] = bigquery_module

    module_path = Path(__file__).resolve().parents[2] / "data-pipeline" / "bigquery_uploader.py"
    uploader_module = load_module_from_path("bigquery_uploader_missing", module_path)

    uploader = uploader_module.BigQueryUploader(project_id="test-project", dataset_id="test_dataset")

    missing_path = tmp_path / "missing.csv"
    try:
        uploader.upload_csv(missing_path, "missing_table")
    except FileNotFoundError as exc:
        assert "CSV file not found" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing CSV path")
