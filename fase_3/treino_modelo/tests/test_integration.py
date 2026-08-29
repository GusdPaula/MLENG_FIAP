"""Integration tests for the ML Training Pipeline.

Covers:
  - End-to-end pipeline run from mock Kaggle ingestion to evaluation report generation (Req 5.1, 5.2, 5.3)
  - Verification of created artifacts: model.pkl and evaluation_report.txt
"""

from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from treino_modelo.pipeline import run_pipeline


def _create_dataset_csvs(base_path: Path) -> None:
    """Create synthetic CSV files simulating the Kaggle dataset."""
    samples = [
        "Patient underwent surgical intervention for malignant neoplasm tumor.",
        "Clinical diagnosis revealed severe peptic ulcer and gastric disease.",
        "Magnetic resonance imaging showed progressive nervous system disorder.",
        "Cardiovascular exam indicated chronic myocardial infarction disease.",
        "Pathological biopsy showed standard chronic tissue inflammation.",
    ]
    # Training dataset: 1200 rows satisfying minimum 1000 rows
    train_labels = [((i % 5) + 1) for i in range(1200)]
    train_abstracts = [samples[i % 5] + f" Clinical record {i}." for i in range(1200)]
    df_train = pd.DataFrame({"condition_label": train_labels, "medical_abstract": train_abstracts})
    df_train.to_csv(base_path / "medical_tc_train.csv", index=False)

    # Testing dataset: 150 rows satisfying minimum 100 rows
    test_labels = [((i % 5) + 1) for i in range(150)]
    test_abstracts = [samples[i % 5] + f" Test evaluation record {i}." for i in range(150)]
    df_test = pd.DataFrame({"condition_label": test_labels, "medical_abstract": test_abstracts})
    df_test.to_csv(base_path / "medical_tc_test.csv", index=False)

    # Labels metadata
    df_labels = pd.DataFrame({
        "condition_label": [1, 2, 3, 4, 5],
        "condition_name": [
            "neoplasms",
            "digestive system diseases",
            "nervous system diseases",
            "cardiovascular diseases",
            "general pathological conditions",
        ],
    })
    df_labels.to_csv(base_path / "medical_tc_labels.csv", index=False)


class TestFullPipelineIntegration:
    def test_full_pipeline_run_creates_artifacts_and_succeeds(self, tmp_path, capsys):
        # Setup synthetic dataset directory
        dataset_dir = tmp_path / "kaggle_dataset"
        dataset_dir.mkdir()
        _create_dataset_csvs(dataset_dir)

        artifacts_dir = tmp_path / "artifacts"
        model_file = artifacts_dir / "model.pkl"
        report_file = artifacts_dir / "evaluation_report.txt"

        with patch("kagglehub.dataset_download", return_value=str(dataset_dir)), \
             patch("treino_modelo.pipeline.ARTIFACTS_DIR", artifacts_dir), \
             patch("treino_modelo.train.train.ARTIFACTS_DIR", artifacts_dir), \
             patch("treino_modelo.train.train.MODEL_PATH", model_file), \
             patch("treino_modelo.train.evaluate.ARTIFACTS_DIR", artifacts_dir), \
             patch("treino_modelo.train.evaluate.REPORT_PATH", report_file):

            run_pipeline()

        # Check artifacts
        assert model_file.exists(), "model.pkl was not created."
        assert report_file.exists(), "evaluation_report.txt was not created."

        # Verify model artifact is loadable and functional
        loaded_model = joblib.load(model_file)
        assert isinstance(loaded_model, Pipeline)
        test_pred = loaded_model.predict(["Sample medical text for test inference."])
        assert len(test_pred) == 1
        assert test_pred[0] in [1, 2, 3, 4, 5]

        # Verify report artifact contains class names and metrics
        report_content = report_file.read_text(encoding="utf-8")
        assert "neoplasms" in report_content
        assert "cardiovascular diseases" in report_content
        assert "accuracy" in report_content

        # Verify stdout output
        out = capsys.readouterr().out
        assert "PIPELINE CONCLUÍDA COM SUCESSO" in out
        assert str(model_file) in out
