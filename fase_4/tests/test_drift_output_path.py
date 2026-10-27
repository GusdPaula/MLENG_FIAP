import numpy as np
import pandas as pd
import pytest

from default_pred.models.drift import DEFAULT_REPORT_PATH, detect_data_drift


@pytest.fixture
def frames():
    rng = np.random.default_rng(0)
    reference = pd.DataFrame(
        {"a": rng.normal(0, 1, 200), "b": rng.normal(0, 1, 200)}
    )
    current = pd.DataFrame({"a": rng.normal(3, 1, 200), "b": rng.normal(0, 1, 200)})
    return reference, current


def test_default_report_path_is_unchanged():
    assert DEFAULT_REPORT_PATH == "data_drift_report.html"


def test_default_output_path_writes_in_working_directory(
    frames, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    reference, current = frames

    detect_data_drift(X_train=reference, X_test=current)

    assert (tmp_path / "data_drift_report.html").exists()


def test_custom_output_path_is_respected(frames, tmp_path):
    reference, current = frames
    target = tmp_path / "custom.html"

    snapshot = detect_data_drift(X_train=reference, X_test=current, output_path=target)

    assert snapshot is not None
    assert target.exists()
