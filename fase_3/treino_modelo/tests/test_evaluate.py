"""Unit and property-based tests for train/evaluate.py — Evaluator component.

Covers:
  - Accuracy calculation and logging (Req 4.3, 4.8)
  - Classification report with 5 clinical class names (Req 4.4)
  - Report persistence in artifacts/evaluation_report.txt (Req 4.5)
  - I/O error resilience on saving report (Req 4.6)
  - Return dictionary structure and types (Req 4.7)
  - Exception handling for prediction failure (Req 4.2)
  - Property 7: Predictions length equals test set length (Req 4.1)
  - Property 8: Report contains all five class names (Req 4.4)
  - Property 9: evaluate_model structure and boundaries (Req 4.7)
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sklearn.pipeline import Pipeline

from treino_modelo.train.evaluate import (
    ACCURACY_THRESHOLD,
    CLASS_NAMES,
    evaluate_model,
)


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_pipeline():
    """A mock sklearn pipeline returning predictable predictions."""
    pipeline = MagicMock(spec=Pipeline)
    pipeline.predict.side_effect = lambda X: np.array([(i % 5) + 1 for i in range(len(X))])
    return pipeline


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestEvaluateModelUnit:
    def test_evaluate_model_returns_dict_with_keys(self, mock_pipeline, df_test_valid, tmp_path):
        with patch("treino_modelo.train.evaluate.ARTIFACTS_DIR", tmp_path), \
             patch("treino_modelo.train.evaluate.REPORT_PATH", tmp_path / "evaluation_report.txt"):
            res = evaluate_model(mock_pipeline, df_test_valid)
            assert isinstance(res, dict)
            assert "accuracy" in res
            assert "report" in res
            assert isinstance(res["accuracy"], float)
            assert isinstance(res["report"], str)

    def test_evaluate_model_accuracy_calculation(self, tmp_path):
        # Known predictions vs ground truth: 3 out of 4 correct -> 0.75
        df = pd.DataFrame({
            "condition_label": [1, 2, 3, 4],
            "medical_abstract": ["Text 1", "Text 2", "Text 3", "Text 4"],
        })
        pipeline = MagicMock(spec=Pipeline)
        pipeline.predict.return_value = np.array([1, 2, 3, 5])  # 3 correct, 1 wrong

        with patch("treino_modelo.train.evaluate.ARTIFACTS_DIR", tmp_path), \
             patch("treino_modelo.train.evaluate.REPORT_PATH", tmp_path / "evaluation_report.txt"):
            res = evaluate_model(pipeline, df)
            assert res["accuracy"] == pytest.approx(0.75)

    def test_report_contains_all_class_names(self, mock_pipeline, df_test_valid, tmp_path):
        with patch("treino_modelo.train.evaluate.ARTIFACTS_DIR", tmp_path), \
             patch("treino_modelo.train.evaluate.REPORT_PATH", tmp_path / "evaluation_report.txt"):
            res = evaluate_model(mock_pipeline, df_test_valid)
            for class_name in CLASS_NAMES:
                assert class_name in res["report"]

    def test_saves_evaluation_report_file(self, mock_pipeline, df_test_valid, tmp_path):
        report_file = tmp_path / "evaluation_report.txt"
        with patch("treino_modelo.train.evaluate.ARTIFACTS_DIR", tmp_path), \
             patch("treino_modelo.train.evaluate.REPORT_PATH", report_file):
            res = evaluate_model(mock_pipeline, df_test_valid)
            assert report_file.exists()
            content = report_file.read_text(encoding="utf-8")
            assert content == res["report"]

    def test_low_accuracy_logs_warning(self, df_test_valid, tmp_path, caplog):
        pipeline = MagicMock(spec=Pipeline)
        # All predictions wrong -> accuracy = 0.0 < 0.50
        pipeline.predict.return_value = np.array([5 if label == 1 else 1 for label in df_test_valid["condition_label"]])

        with caplog.at_level(logging.WARNING), \
             patch("treino_modelo.train.evaluate.ARTIFACTS_DIR", tmp_path), \
             patch("treino_modelo.train.evaluate.REPORT_PATH", tmp_path / "evaluation_report.txt"):
            evaluate_model(pipeline, df_test_valid)
            assert any("abaixo do limiar mínimo aceitável" in record.message for record in caplog.records)

    def test_io_error_on_save_is_absorbed(self, mock_pipeline, df_test_valid, tmp_path, caplog):
        with caplog.at_level(logging.ERROR), \
             patch("treino_modelo.train.evaluate.ARTIFACTS_DIR", tmp_path), \
             patch("pathlib.Path.write_text", side_effect=OSError("Write permission denied")):
            res = evaluate_model(mock_pipeline, df_test_valid)
            # Result dictionary is still returned despite write failure
            assert "accuracy" in res
            assert "report" in res
            assert any("Falha ao salvar o relatório" in record.message for record in caplog.records)

    def test_predict_exception_logged_and_reraised(self, df_test_valid, tmp_path, caplog):
        pipeline = MagicMock(spec=Pipeline)
        pipeline.predict.side_effect = RuntimeError("Predict failure")

        with caplog.at_level(logging.ERROR), \
             patch("treino_modelo.train.evaluate.ARTIFACTS_DIR", tmp_path):
            with pytest.raises(RuntimeError, match="Predict failure"):
                evaluate_model(pipeline, df_test_valid)
            assert any("Erro durante a predição" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

_sample_texts = [
    "Clinical study of neoplastic cell line response.",
    "Digestive disorder assessment and gastric analysis.",
    "Neurological scan of central nervous system.",
    "Cardiovascular heart rate and arterial pressure.",
    "Pathological findings of inflammatory syndrome.",
]

HEALTH_CHECKS = [
    HealthCheck.large_base_example,
    HealthCheck.too_slow,
    HealthCheck.data_too_large,
]


@st.composite
def _test_df_strategy(draw) -> pd.DataFrame:
    n = draw(st.integers(min_value=10, max_value=50))
    sample_labels = [draw(st.integers(min_value=1, max_value=5)) for _ in range(5)]
    labels = [sample_labels[i % 5] for i in range(n)]
    sample_abstracts = [draw(st.sampled_from(_sample_texts)) for _ in range(5)]
    abstracts = [sample_abstracts[i % 5] + f" Id {i}" for i in range(n)]
    return pd.DataFrame({"condition_label": labels, "medical_abstract": abstracts})


# Property 7: Predições têm o mesmo comprimento que o conjunto de teste (Req 4.1)
@given(df_test=_test_df_strategy())
@settings(max_examples=25, suppress_health_check=HEALTH_CHECKS)
def test_prop7_predictions_length_matches_test_set(df_test):
    """Property 7: pipeline.predict() produces an array with the same length as the input df_test.

    **Validates: Requirement 4.1**
    """
    pipeline = MagicMock(spec=Pipeline)
    pipeline.predict.side_effect = lambda X: np.array([1] * len(X))
    preds = pipeline.predict(df_test["medical_abstract"])
    assert len(preds) == len(df_test)


# Property 8: Relatório contém os nomes das cinco classes (Req 4.4)
@given(df_test=_test_df_strategy())
@settings(max_examples=25, suppress_health_check=HEALTH_CHECKS)
def test_prop8_report_contains_all_five_classes(df_test, tmp_path_factory):
    """Property 8: evaluate_model() generates a report containing all 5 clinical class names.

    **Validates: Requirement 4.4**
    """
    tmp_path = tmp_path_factory.mktemp("rep")
    pipeline = MagicMock(spec=Pipeline)
    pipeline.predict.side_effect = lambda X: np.array([((i % 5) + 1) for i in range(len(X))])

    with patch("treino_modelo.train.evaluate.ARTIFACTS_DIR", tmp_path), \
         patch("treino_modelo.train.evaluate.REPORT_PATH", tmp_path / "evaluation_report.txt"):
        res = evaluate_model(pipeline, df_test)

    for name in CLASS_NAMES:
        assert name in res["report"], f"Expected class '{name}' in evaluation report."


# Property 9: evaluate_model() sempre retorna dicionário com estrutura correta (Req 4.7)
@given(df_test=_test_df_strategy())
@settings(max_examples=25, suppress_health_check=HEALTH_CHECKS)
def test_prop9_evaluate_model_structure_and_bounds(df_test, tmp_path_factory):
    """Property 9: evaluate_model() always returns dict with 'accuracy' in [0.0, 1.0]
    and non-empty 'report'.

    **Validates: Requirement 4.7**
    """
    tmp_path = tmp_path_factory.mktemp("eval")
    pipeline = MagicMock(spec=Pipeline)
    pipeline.predict.side_effect = lambda X: np.array([((i % 5) + 1) for i in range(len(X))])

    with patch("treino_modelo.train.evaluate.ARTIFACTS_DIR", tmp_path), \
         patch("treino_modelo.train.evaluate.REPORT_PATH", tmp_path / "evaluation_report.txt"):
        res = evaluate_model(pipeline, df_test)

    assert isinstance(res, dict)
    assert set(res.keys()) == {"accuracy", "report"}
    assert isinstance(res["accuracy"], float)
    assert 0.0 <= res["accuracy"] <= 1.0
    assert isinstance(res["report"], str)
    assert len(res["report"].strip()) > 0
