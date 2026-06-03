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


def test_kaggle_data_loader_download_and_combine(tmp_path):
    fake_root = tmp_path / "dataset"
    fake_root.mkdir()

    (fake_root / "item_properties_part1.csv").write_text("item_id,item_property\n1,A\n")
    (fake_root / "item_properties_part2.csv").write_text("item_id,item_property\n2,B\n")
    (fake_root / "category_tree.csv").write_text("category_id,parent_id\n")
    (fake_root / "events.csv").write_text("event_id,user_id\n")

    fake_kagglehub = types.SimpleNamespace(
        dataset_download=lambda dataset_name: str(fake_root)
    )
    sys.modules["kagglehub"] = fake_kagglehub

    module_path = Path(__file__).resolve().parents[2] / "data-pipeline" / "kaggle_data_loader.py"
    loader_module = load_module_from_path("kaggle_data_loader", module_path)

    loader = loader_module.KaggleDataLoader()
    downloaded_path = loader.download_dataset()

    assert downloaded_path == fake_root

    combined_path = loader.combine_item_properties(downloaded_path)
    assert combined_path.exists()
    csv_text = combined_path.read_text().splitlines()
    assert csv_text[0] == "item_id,item_property"
    assert len(csv_text) == 3


def test_kaggle_data_loader_collect_files(tmp_path):
    fake_root = tmp_path / "dataset"
    fake_root.mkdir()
    (fake_root / "category_tree.csv").write_text("category_id,parent_id\n")
    (fake_root / "events.csv").write_text("event_id,user_id\n")
    (fake_root / "item_properties.csv").write_text("item_id,item_property\n1,A\n")

    module_path = Path(__file__).resolve().parents[2] / "data-pipeline" / "kaggle_data_loader.py"
    loader_module = load_module_from_path("kaggle_data_loader_for_collect", module_path)
    loader = loader_module.KaggleDataLoader()

    files = loader.collect_files(fake_root)
    assert files["category_tree"] == fake_root / "category_tree.csv"
    assert files["events"] == fake_root / "events.csv"
    assert files["item_properties"] == fake_root / "item_properties.csv"
