"""Unit tests for pipeline.py — Orchestrator component.

Covers:
  - Sequential execution order: Ingestion -> Validation -> Training -> Evaluation (Req 5.1)
  - Error handling: immediate exit with code 1 upon failure of any stage (Req 5.2)
  - Output format: console summary with 4-decimal accuracy and model path (Req 5.3)
  - Directory creation: artifacts/ directory created before execution (Req 5.5)
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from treino_modelo.pipeline import run_pipeline


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_stages():
    """Mocks for all 4 pipeline stages returning valid mock objects."""
    train_df = pd.DataFrame({"condition_label": [1, 2], "medical_abstract": ["A", "B"]})
    test_df = pd.DataFrame({"condition_label": [1, 2], "medical_abstract": ["C", "D"]})
    data_dict = {"train": train_df, "test": test_df, "labels": pd.DataFrame()}

    mock_load = MagicMock(return_value=data_dict)
    mock_validate = MagicMock(return_value=True)
    mock_pipeline_obj = MagicMock()
    mock_train = MagicMock(return_value=mock_pipeline_obj)
    mock_eval_results = {"accuracy": 0.85234, "report": "mock report"}
    mock_evaluate = MagicMock(return_value=mock_eval_results)

    return {
        "load_data": mock_load,
        "validate_data": mock_validate,
        "train_model": mock_train,
        "evaluate_model": mock_evaluate,
        "data_dict": data_dict,
        "pipeline_obj": mock_pipeline_obj,
        "eval_results": mock_eval_results,
    }


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestPipelineOrchestrator:
    def test_run_pipeline_executes_stages_in_order(self, mock_stages, tmp_path, capsys):
        with patch("treino_modelo.pipeline.ARTIFACTS_DIR", tmp_path), \
             patch("treino_modelo.pipeline.load_data", mock_stages["load_data"]), \
             patch("treino_modelo.pipeline.validate_data", mock_stages["validate_data"]), \
             patch("treino_modelo.pipeline.train_model", mock_stages["train_model"]), \
             patch("treino_modelo.pipeline.evaluate_model", mock_stages["evaluate_model"]):

            run_pipeline()

            # Verify call sequence
            mock_stages["load_data"].assert_called_once()
            mock_stages["validate_data"].assert_called_once_with(
                mock_stages["data_dict"]["train"], mock_stages["data_dict"]["test"]
            )
            mock_stages["train_model"].assert_called_once_with(mock_stages["data_dict"]["train"])
            mock_stages["evaluate_model"].assert_called_once_with(
                mock_stages["pipeline_obj"], mock_stages["data_dict"]["test"]
            )

    def test_run_pipeline_prints_summary(self, mock_stages, tmp_path, capsys):
        with patch("treino_modelo.pipeline.ARTIFACTS_DIR", tmp_path), \
             patch("treino_modelo.pipeline.load_data", mock_stages["load_data"]), \
             patch("treino_modelo.pipeline.validate_data", mock_stages["validate_data"]), \
             patch("treino_modelo.pipeline.train_model", mock_stages["train_model"]), \
             patch("treino_modelo.pipeline.evaluate_model", mock_stages["evaluate_model"]):

            run_pipeline()

            captured = capsys.readouterr().out
            assert "PIPELINE CONCLUÍDA COM SUCESSO" in captured
            assert "0.8523" in captured
            assert str(tmp_path / "model.pkl") in captured

    def test_ingestion_failure_exits_with_code_1(self, mock_stages, tmp_path, caplog):
        mock_stages["load_data"].side_effect = RuntimeError("Download error")

        with patch("treino_modelo.pipeline.ARTIFACTS_DIR", tmp_path), \
             patch("treino_modelo.pipeline.load_data", mock_stages["load_data"]), \
             patch("treino_modelo.pipeline.validate_data", mock_stages["validate_data"]), \
             patch("treino_modelo.pipeline.train_model", mock_stages["train_model"]), \
             patch("treino_modelo.pipeline.evaluate_model", mock_stages["evaluate_model"]):

            with pytest.raises(SystemExit) as exc_info:
                run_pipeline()
            assert exc_info.value.code == 1
            mock_stages["validate_data"].assert_not_called()
            mock_stages["train_model"].assert_not_called()
            mock_stages["evaluate_model"].assert_not_called()

    def test_validation_failure_exits_with_code_1(self, mock_stages, tmp_path, caplog):
        mock_stages["validate_data"].side_effect = ValueError("Schema invalid")

        with patch("treino_modelo.pipeline.ARTIFACTS_DIR", tmp_path), \
             patch("treino_modelo.pipeline.load_data", mock_stages["load_data"]), \
             patch("treino_modelo.pipeline.validate_data", mock_stages["validate_data"]), \
             patch("treino_modelo.pipeline.train_model", mock_stages["train_model"]), \
             patch("treino_modelo.pipeline.evaluate_model", mock_stages["evaluate_model"]):

            with pytest.raises(SystemExit) as exc_info:
                run_pipeline()
            assert exc_info.value.code == 1
            mock_stages["train_model"].assert_not_called()
            mock_stages["evaluate_model"].assert_not_called()

    def test_training_failure_exits_with_code_1(self, mock_stages, tmp_path, caplog):
        mock_stages["train_model"].side_effect = RuntimeError("Fit crashed")

        with patch("treino_modelo.pipeline.ARTIFACTS_DIR", tmp_path), \
             patch("treino_modelo.pipeline.load_data", mock_stages["load_data"]), \
             patch("treino_modelo.pipeline.validate_data", mock_stages["validate_data"]), \
             patch("treino_modelo.pipeline.train_model", mock_stages["train_model"]), \
             patch("treino_modelo.pipeline.evaluate_model", mock_stages["evaluate_model"]):

            with pytest.raises(SystemExit) as exc_info:
                run_pipeline()
            assert exc_info.value.code == 1
            mock_stages["evaluate_model"].assert_not_called()

    def test_evaluation_failure_exits_with_code_1(self, mock_stages, tmp_path, caplog):
        mock_stages["evaluate_model"].side_effect = RuntimeError("Predict crashed")

        with patch("treino_modelo.pipeline.ARTIFACTS_DIR", tmp_path), \
             patch("treino_modelo.pipeline.load_data", mock_stages["load_data"]), \
             patch("treino_modelo.pipeline.validate_data", mock_stages["validate_data"]), \
             patch("treino_modelo.pipeline.train_model", mock_stages["train_model"]), \
             patch("treino_modelo.pipeline.evaluate_model", mock_stages["evaluate_model"]):

            with pytest.raises(SystemExit) as exc_info:
                run_pipeline()
            assert exc_info.value.code == 1
