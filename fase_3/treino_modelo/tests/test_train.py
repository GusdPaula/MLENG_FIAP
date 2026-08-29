"""Unit and property-based tests for train/train.py — Trainer component.

Covers:
  - build_pipeline configuration and steps (Req 3.1, 3.2, 3.3)
  - train_model fitting, persistence and logging (Req 3.4, 3.6, 3.7)
  - train_model error handling for missing columns, fit errors, and serialization errors (Req 3.5, 3.8)
  - Property 10: Determinism (Req 6.1)
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from treino_modelo.train.train import ARTIFACTS_DIR, MODEL_PATH, build_pipeline, train_model


# ---------------------------------------------------------------------------
# Unit tests — build_pipeline
# ---------------------------------------------------------------------------

class TestBuildPipeline:
    def test_returns_pipeline_instance(self):
        pipeline = build_pipeline()
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_has_expected_step_names(self):
        pipeline = build_pipeline()
        step_names = [name for name, _ in pipeline.steps]
        assert step_names == ["tfidf", "clf"]

    def test_tfidf_step_configuration(self):
        pipeline = build_pipeline()
        tfidf = pipeline.named_steps["tfidf"]
        assert isinstance(tfidf, TfidfVectorizer)
        assert tfidf.stop_words == "english"
        assert tfidf.ngram_range == (1, 1)
        assert tfidf.min_df == 5
        assert tfidf.max_df == 0.95
        assert tfidf.lowercase is True

    def test_clf_step_configuration(self):
        pipeline = build_pipeline()
        clf = pipeline.named_steps["clf"]
        assert isinstance(clf, LinearSVC)
        assert clf.C == 0.06
        assert clf.random_state == 42
        assert clf.dual is False


# ---------------------------------------------------------------------------
# Unit tests — train_model
# ---------------------------------------------------------------------------

class TestTrainModel:
    def test_train_model_returns_fitted_pipeline(self, df_train_valid, tmp_path):
        custom_model_path = tmp_path / "artifacts" / "model.pkl"
        with patch("treino_modelo.train.train.ARTIFACTS_DIR", tmp_path / "artifacts"), \
             patch("treino_modelo.train.train.MODEL_PATH", custom_model_path):
            pipeline = train_model(df_train_valid)
            assert isinstance(pipeline, Pipeline)
            assert hasattr(pipeline.named_steps["clf"], "classes_")

    def test_train_model_creates_model_file(self, df_train_valid, tmp_path):
        custom_model_path = tmp_path / "artifacts" / "model.pkl"
        with patch("treino_modelo.train.train.ARTIFACTS_DIR", tmp_path / "artifacts"), \
             patch("treino_modelo.train.train.MODEL_PATH", custom_model_path):
            train_model(df_train_valid)
            assert custom_model_path.exists()
            loaded = joblib.load(custom_model_path)
            assert isinstance(loaded, Pipeline)

    def test_train_model_logs_elapsed_time(self, df_train_valid, tmp_path, caplog):
        custom_model_path = tmp_path / "artifacts" / "model.pkl"
        with caplog.at_level(logging.INFO), \
             patch("treino_modelo.train.train.ARTIFACTS_DIR", tmp_path / "artifacts"), \
             patch("treino_modelo.train.train.MODEL_PATH", custom_model_path):
            train_model(df_train_valid)
            assert any("Treinamento concluído em" in record.message for record in caplog.records)
            assert any("Modelo salvo em" in record.message for record in caplog.records)

    def test_train_model_raises_key_error_for_missing_abstract(self, tmp_path, caplog):
        df_no_abstract = pd.DataFrame({"condition_label": [1, 2, 3]})
        with caplog.at_level(logging.ERROR), \
             patch("treino_modelo.train.train.ARTIFACTS_DIR", tmp_path / "artifacts"):
            with pytest.raises(KeyError, match="medical_abstract"):
                train_model(df_no_abstract)
            assert any("Coluna obrigatória ausente" in record.message for record in caplog.records)

    def test_train_model_raises_key_error_for_missing_label(self, tmp_path, caplog):
        df_no_label = pd.DataFrame({"medical_abstract": ["Text 1", "Text 2"]})
        with caplog.at_level(logging.ERROR), \
             patch("treino_modelo.train.train.ARTIFACTS_DIR", tmp_path / "artifacts"):
            with pytest.raises(KeyError, match="condition_label"):
                train_model(df_no_label)
            assert any("Coluna obrigatória ausente" in record.message for record in caplog.records)

    def test_train_model_fit_exception_logged_and_reraised(self, df_train_valid, tmp_path, caplog):
        with caplog.at_level(logging.ERROR), \
             patch("treino_modelo.train.train.Pipeline.fit", side_effect=ValueError("Simulated fit error")), \
             patch("treino_modelo.train.train.ARTIFACTS_DIR", tmp_path / "artifacts"):
            with pytest.raises(ValueError, match="Simulated fit error"):
                train_model(df_train_valid)
            assert any("Erro durante o ajuste da pipeline" in record.message for record in caplog.records)

    def test_train_model_joblib_exception_logged_and_reraised(self, df_train_valid, tmp_path, caplog):
        with caplog.at_level(logging.ERROR), \
             patch("treino_modelo.train.train.joblib.dump", side_effect=OSError("Disk full")), \
             patch("treino_modelo.train.train.ARTIFACTS_DIR", tmp_path / "artifacts"):
            with pytest.raises(OSError, match="Disk full"):
                train_model(df_train_valid)
            assert any("Erro durante a serialização do modelo" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Property-based test — Property 10: Determinism (Req 6.1)
# ---------------------------------------------------------------------------

@given(seed=st.integers(min_value=1, max_value=100))
@settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
def test_prop10_training_is_deterministic(seed, tmp_path_factory):
    """Property 10: For identical training data, two independent train_model calls
    must produce pipelines with identical predictions on test inputs.

    **Validates: Requirement 6.1**
    """
    tmp_path1 = tmp_path_factory.mktemp(f"det1_{seed}")
    tmp_path2 = tmp_path_factory.mktemp(f"det2_{seed}")

    rng = np.random.default_rng(seed)
    samples = [
        "Patient underwent chemotherapy treatment for carcinoma neoplasm.",
        "Endoscopic findings confirmed gastric ulcer and digestive pathology.",
        "Neurological examination showed cerebral infarction and stroke.",
        "Echocardiogram indicated severe myocardial infarction and coronary disease.",
        "Biopsy revealed chronic inflammatory reaction in connective tissue.",
    ]
    n_train = 50
    train_labels = [((i % 5) + 1) for i in range(n_train)]
    train_abstracts = [samples[i % 5] + f" Repeat {i} with key term {i % 3}." for i in range(n_train)]
    df_train = pd.DataFrame({"condition_label": train_labels, "medical_abstract": train_abstracts})

    test_abstracts = [samples[i % 5] + f" Test abstract {i}." for i in range(10)]

    with patch("treino_modelo.train.train.ARTIFACTS_DIR", tmp_path1), \
         patch("treino_modelo.train.train.MODEL_PATH", tmp_path1 / "model.pkl"):
        # Pipeline must handle min_df=1 for smaller synthetic test sets if needed,
        # but with 50 samples repeated words have df >= 5
        pipeline1 = train_model(df_train)

    with patch("treino_modelo.train.train.ARTIFACTS_DIR", tmp_path2), \
         patch("treino_modelo.train.train.MODEL_PATH", tmp_path2 / "model.pkl"):
        pipeline2 = train_model(df_train)

    preds1 = pipeline1.predict(test_abstracts)
    preds2 = pipeline2.predict(test_abstracts)

    np.testing.assert_array_equal(preds1, preds2)
