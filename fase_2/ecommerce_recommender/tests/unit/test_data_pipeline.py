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


def test_data_pipeline_runs_with_prefixed_tables(tmp_path):
    kaggle_module_path = (
        Path(__file__).resolve().parents[2] / "data_pipeline" / "kaggle_data_loader.py"
    )
    bigquery_module_path = (
        Path(__file__).resolve().parents[2] / "data_pipeline" / "bigquery_uploader.py"
    )
    pipeline_module_path = (
        Path(__file__).resolve().parents[2] / "data_pipeline" / "pipeline.py"
    )

    sys.modules["kagglehub"] = types.SimpleNamespace(
        dataset_download=lambda dataset_name: str(tmp_path)
    )

    load_module_from_path("kaggle_data_loader", kaggle_module_path)
    load_module_from_path("bigquery_uploader", bigquery_module_path)
    pipeline_module = load_module_from_path("pipeline_module", pipeline_module_path)

    class DummyKaggleLoader:
        def download_dataset(self):
            return tmp_path

        def combine_item_properties(self, download_path):
            combined_file = download_path / "item_properties.csv"
            combined_file.write_text("item_id,item_property\n1,A\n")
            return combined_file

    class DummyBigQueryUploader:
        def __init__(self):
            self.uploaded = {}

        def upload_files(self, source_files):
            self.uploaded = source_files
            return {name: f"project.dataset.{name}" for name in source_files}

    dummy_loader = DummyKaggleLoader()
    dummy_uploader = DummyBigQueryUploader()
    pipeline = pipeline_module.DataPipeline(
        kaggle_loader=dummy_loader,
        bigquery_uploader=dummy_uploader,
        table_prefix="testprefix",
    )

    results = pipeline.run()

    assert (
        results["testprefix_category_tree"]
        == "project.dataset.testprefix_category_tree"
    )
    assert "testprefix_events" in results
    assert "testprefix_item_properties" in results
    assert (
        dummy_uploader.uploaded["testprefix_item_properties"].name
        == "item_properties.csv"
    )
