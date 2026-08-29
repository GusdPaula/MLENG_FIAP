"""
Unit tests for data/ingest.py — Ingestor component.

Covers:
  - Happy path: load_data() returns dict with exactly the three expected keys (Req 1.7)
  - Missing file: FileNotFoundError with the standardised message (Req 1.4)
  - kagglehub exception propagation: exceptions raised by kagglehub are re-raised
    unchanged, without any wrapping or modification (Req 1.5)
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csv_bytes(n_rows: int = 5) -> bytes:
    """Return the bytes of a minimal valid CSV (condition_label, medical_abstract)."""
    rows = "\n".join(
        f"{(i % 5) + 1},Abstract text for row {i}."
        for i in range(n_rows)
    )
    return f"condition_label,medical_abstract\n{rows}\n".encode()


def _make_labels_csv_bytes() -> bytes:
    content = (
        "condition_label,condition_name\n"
        "1,neoplasms\n"
        "2,digestive system diseases\n"
        "3,nervous system diseases\n"
        "4,cardiovascular diseases\n"
        "5,general pathological conditions\n"
    )
    return content.encode()


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestLoadDataHappyPath:
    """load_data() succeeds when kagglehub returns a path with all three CSVs."""

    def test_returns_dict_with_all_three_keys(self, tmp_path):
        """load_data() must return exactly the keys 'train', 'test', 'labels'."""
        # Arrange: create the three CSV files that load_data() expects
        (tmp_path / "medical_tc_train.csv").write_bytes(_make_csv_bytes(10))
        (tmp_path / "medical_tc_test.csv").write_bytes(_make_csv_bytes(5))
        (tmp_path / "medical_tc_labels.csv").write_bytes(_make_labels_csv_bytes())

        with patch("kagglehub.dataset_download", return_value=str(tmp_path)):
            from treino_modelo.data.ingest import load_data
            result = load_data()

        assert set(result.keys()) == {"train", "test", "labels"}, (
            "Expected exactly the keys 'train', 'test' and 'labels'"
        )

    def test_values_are_dataframes(self, tmp_path):
        """Each value in the returned dict must be a pandas DataFrame."""
        (tmp_path / "medical_tc_train.csv").write_bytes(_make_csv_bytes(10))
        (tmp_path / "medical_tc_test.csv").write_bytes(_make_csv_bytes(5))
        (tmp_path / "medical_tc_labels.csv").write_bytes(_make_labels_csv_bytes())

        with patch("kagglehub.dataset_download", return_value=str(tmp_path)):
            from treino_modelo.data.ingest import load_data
            result = load_data()

        for key in ("train", "test", "labels"):
            assert isinstance(result[key], pd.DataFrame), (
                f"result['{key}'] should be a DataFrame, got {type(result[key])}"
            )

    def test_dataframes_contain_expected_rows(self, tmp_path):
        """DataFrames should have the same row-count as the CSV files written."""
        (tmp_path / "medical_tc_train.csv").write_bytes(_make_csv_bytes(12))
        (tmp_path / "medical_tc_test.csv").write_bytes(_make_csv_bytes(7))
        (tmp_path / "medical_tc_labels.csv").write_bytes(_make_labels_csv_bytes())

        with patch("kagglehub.dataset_download", return_value=str(tmp_path)):
            from treino_modelo.data.ingest import load_data
            result = load_data()

        assert len(result["train"]) == 12
        assert len(result["test"]) == 7
        assert len(result["labels"]) == 5

    def test_kagglehub_is_called_with_correct_slug(self, tmp_path):
        """kagglehub.dataset_download must be called with the DATASET_SLUG constant."""
        (tmp_path / "medical_tc_train.csv").write_bytes(_make_csv_bytes())
        (tmp_path / "medical_tc_test.csv").write_bytes(_make_csv_bytes())
        (tmp_path / "medical_tc_labels.csv").write_bytes(_make_labels_csv_bytes())

        with patch("kagglehub.dataset_download", return_value=str(tmp_path)) as mock_dl:
            from treino_modelo.data.ingest import load_data, DATASET_SLUG
            load_data()

        mock_dl.assert_called_once_with(DATASET_SLUG)


# ---------------------------------------------------------------------------
# FileNotFoundError tests
# ---------------------------------------------------------------------------

class TestLoadDataMissingFiles:
    """load_data() raises FileNotFoundError when a CSV is absent from the path."""

    @pytest.mark.parametrize("missing_key,missing_filename", [
        ("train",  "medical_tc_train.csv"),
        ("test",   "medical_tc_test.csv"),
        ("labels", "medical_tc_labels.csv"),
    ])
    def test_raises_file_not_found_for_missing_csv(
        self, tmp_path, missing_key, missing_filename
    ):
        """FileNotFoundError is raised when any of the three expected CSVs is absent."""
        # Write the two CSVs that are NOT missing
        all_files = {
            "medical_tc_train.csv":  _make_csv_bytes(),
            "medical_tc_test.csv":   _make_csv_bytes(),
            "medical_tc_labels.csv": _make_labels_csv_bytes(),
        }
        for filename, content in all_files.items():
            if filename != missing_filename:
                (tmp_path / filename).write_bytes(content)

        with patch("kagglehub.dataset_download", return_value=str(tmp_path)):
            from treino_modelo.data.ingest import load_data
            with pytest.raises(FileNotFoundError):
                load_data()

    def test_error_message_contains_filename(self, tmp_path):
        """FileNotFoundError message must contain the name of the missing file."""
        # Only write train and labels; test CSV is absent
        (tmp_path / "medical_tc_train.csv").write_bytes(_make_csv_bytes())
        (tmp_path / "medical_tc_labels.csv").write_bytes(_make_labels_csv_bytes())
        # medical_tc_test.csv is intentionally NOT created

        with patch("kagglehub.dataset_download", return_value=str(tmp_path)):
            from treino_modelo.data.ingest import load_data
            with pytest.raises(FileNotFoundError) as exc_info:
                load_data()

        assert "medical_tc_test.csv" in str(exc_info.value)

    def test_error_message_contains_path(self, tmp_path):
        """FileNotFoundError message must contain the path that was checked."""
        # Only write train and test; labels CSV is absent
        (tmp_path / "medical_tc_train.csv").write_bytes(_make_csv_bytes())
        (tmp_path / "medical_tc_test.csv").write_bytes(_make_csv_bytes())
        # medical_tc_labels.csv is intentionally NOT created

        with patch("kagglehub.dataset_download", return_value=str(tmp_path)):
            from treino_modelo.data.ingest import load_data
            with pytest.raises(FileNotFoundError) as exc_info:
                load_data()

        # The standardised message includes the path returned by kagglehub
        assert str(tmp_path) in str(exc_info.value)

    def test_error_message_full_format(self, tmp_path):
        """FileNotFoundError message matches the exact standardised format."""
        # Only train is absent so the error is deterministic (EXPECTED_FILES is ordered)
        # Since EXPECTED_FILES iterates train → test → labels, omitting train triggers first.
        (tmp_path / "medical_tc_test.csv").write_bytes(_make_csv_bytes())
        (tmp_path / "medical_tc_labels.csv").write_bytes(_make_labels_csv_bytes())
        # medical_tc_train.csv is intentionally NOT created

        with patch("kagglehub.dataset_download", return_value=str(tmp_path)):
            from treino_modelo.data.ingest import load_data
            with pytest.raises(FileNotFoundError) as exc_info:
                load_data()

        expected_fragment = (
            f"Arquivo esperado não encontrado: 'medical_tc_train.csv' "
            f"no caminho '{tmp_path}'"
        )
        assert expected_fragment in str(exc_info.value)


# ---------------------------------------------------------------------------
# kagglehub exception propagation tests
# ---------------------------------------------------------------------------

class TestLoadDataKagglehubExceptions:
    """Exceptions from kagglehub.dataset_download are propagated without modification."""

    def test_connection_error_is_propagated(self):
        """ConnectionError from kagglehub reaches the caller unchanged."""
        original_exc = ConnectionError("Network unavailable")

        with patch("kagglehub.dataset_download", side_effect=original_exc):
            from treino_modelo.data.ingest import load_data
            with pytest.raises(ConnectionError) as exc_info:
                load_data()

        # The *same* exception object must propagate (not a wrapped copy)
        assert exc_info.value is original_exc

    def test_runtime_error_is_propagated(self):
        """RuntimeError from kagglehub (e.g. auth failure) propagates unchanged."""
        original_exc = RuntimeError("Authentication failed")

        with patch("kagglehub.dataset_download", side_effect=original_exc):
            from treino_modelo.data.ingest import load_data
            with pytest.raises(RuntimeError) as exc_info:
                load_data()

        assert exc_info.value is original_exc

    def test_permission_error_is_propagated(self):
        """PermissionError from kagglehub propagates unchanged."""
        original_exc = PermissionError("Access denied to dataset")

        with patch("kagglehub.dataset_download", side_effect=original_exc):
            from treino_modelo.data.ingest import load_data
            with pytest.raises(PermissionError) as exc_info:
                load_data()

        assert exc_info.value is original_exc

    def test_exception_message_is_not_modified(self):
        """The error message from kagglehub must not be altered by the Ingestor."""
        original_message = "Some very specific kagglehub error message"
        original_exc = Exception(original_message)

        with patch("kagglehub.dataset_download", side_effect=original_exc):
            from treino_modelo.data.ingest import load_data
            with pytest.raises(Exception) as exc_info:
                load_data()

        assert str(exc_info.value) == original_message
