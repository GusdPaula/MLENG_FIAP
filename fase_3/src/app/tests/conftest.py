"""
Shared fixtures for the API test suite.

Fixtures include:
  1. FastAPI test client
  2. Clinical diagnosis data samples (similar to treino_modelo data)
  3. Mock inference service
  4. Test configuration
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app

# ---------------------------------------------------------------------------
# Clinical diagnosis data samples (based on treino_modelo data structure)
# ---------------------------------------------------------------------------

CLINICAL_DIAGNOSIS_SAMPLES = [
    # Neoplasms (label 1)
    "The patient presented with a malignant neoplasm of the colon requiring surgical resection.",
    "Histopathology confirmed adenocarcinoma of the lung with metastatic involvement.",
    "Biopsy revealed hepatocellular carcinoma with underlying cirrhosis.",
    # Digestive system diseases (label 2)
    "Gastroesophageal reflux disease was diagnosed after endoscopy examination showing esophagitis.",
    "Colonoscopy identified inflammatory bowel disease with ulcerative colitis.",
    "The patient has chronic pancreatitis with recurrent episodes of abdominal pain.",
    # Nervous system diseases (label 3)
    "MRI revealed a demyelinating lesion consistent with multiple sclerosis in the white matter.",
    "Neurological examination indicated signs of Parkinson's disease progression with tremor.",
    "EEG confirmed epileptiform activity consistent with temporal lobe epilepsy.",
    # Cardiovascular diseases (label 4)
    "Echocardiography confirmed severe aortic stenosis with reduced ejection fraction.",
    "Coronary angiography demonstrated significant stenosis of the left anterior descending artery.",
    "The patient presented with acute myocardial infarction with ST-segment elevation.",
    # General pathological conditions (label 5)
    "Laboratory findings were consistent with systemic lupus erythematosus with renal involvement.",
    "Histopathology showed chronic inflammatory changes with granuloma formation.",
    "The patient developed sepsis with multi-organ dysfunction syndrome.",
]

CLASS_LABELS = [
    "Neoplasms",
    "Digestive system diseases",
    "Nervous system diseases",
    "Cardiovascular diseases",
    "General pathological conditions",
]


# ---------------------------------------------------------------------------
# FastAPI test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_model_path(tmp_path):
    """Create a temporary path for the mock ONNX model."""
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"fake onnx model content")
    return str(model_file)


@pytest.fixture
def mock_inference_service():
    """Create a mock InferenceService for testing."""
    mock_service = MagicMock()

    # Mock predict_batch to return realistic responses
    def mock_predict_batch(texts):
        from app.schemas.classify import BatchClassifyResponse, PredictionResult

        results = []
        for _i, text in enumerate(texts):
            # Simulate predictions based on text content
            if "neoplasm" in text.lower() or "cancer" in text.lower():
                label = "Neoplasms"
            elif "digestive" in text.lower() or "gastro" in text.lower() or "colon" in text.lower():
                label = "Digestive system diseases"
            elif "nervous" in text.lower() or "brain" in text.lower() or "mri" in text.lower():
                label = "Nervous system diseases"
            elif "cardio" in text.lower() or "heart" in text.lower() or "coronary" in text.lower():
                label = "Cardiovascular diseases"
            else:
                label = "General pathological conditions"

            # Generate realistic confidence scores
            scores = {label: np.random.random() for label in CLASS_LABELS}
            scores[label] = 0.85 + np.random.random() * 0.14  # High confidence for predicted label

            results.append(PredictionResult(label=label, confidence=scores[label], scores=scores))

        return BatchClassifyResponse(
            results=results,
            model_version="tfidf-rf-v1.0-onnx",
            batch_size=len(texts),
            inference_ms=50.0 + len(texts) * 5.0,
        )

    mock_service.predict_batch = mock_predict_batch
    mock_service.is_ready.return_value = True
    mock_service.warm_up.return_value = None

    return mock_service


@pytest.fixture
def test_app(mock_inference_service, mock_model_path):
    """Create a FastAPI test application with mocked dependencies."""
    with patch("app.config.get_settings") as mock_settings:
        # Mock settings
        settings_mock = MagicMock()
        settings_mock.app_name = "Medical Classification API"
        settings_mock.app_version = "1.0.0"
        settings_mock.model_path = mock_model_path
        settings_mock.model_version = "tfidf-rf-v1.0-onnx"
        settings_mock.class_labels = CLASS_LABELS
        settings_mock.log_level = "INFO"
        settings_mock.max_batch_size = 100
        settings_mock.max_text_length = 10000
        mock_settings.return_value = settings_mock

        app = create_app()

        # Manually set the inference service on app state
        app.state.inference_service = mock_inference_service
        app.state.settings = settings_mock

    return app


@pytest.fixture
def client(test_app):
    """Create a test client for the FastAPI application."""
    return TestClient(test_app)


# ---------------------------------------------------------------------------
# Clinical diagnosis data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_clinical_diagnosis():
    """Return a single clinical diagnosis sample."""
    return CLINICAL_DIAGNOSIS_SAMPLES[0]


@pytest.fixture
def sample_clinical_diagnoses():
    """Return multiple clinical diagnosis samples."""
    return CLINICAL_DIAGNOSIS_SAMPLES[:5]


@pytest.fixture
def sample_clinical_diagnoses_batch():
    """Return a larger batch of clinical diagnosis samples."""
    return CLINICAL_DIAGNOSIS_SAMPLES


@pytest.fixture
def clinical_diagnosis_by_category():
    """Return clinical diagnoses organized by category."""
    return {
        "neoplasms": [
            "The patient presented with a malignant neoplasm of the colon requiring surgical resection.",
            "Histopathology confirmed adenocarcinoma of the lung with metastatic involvement.",
            "Biopsy revealed hepatocellular carcinoma with underlying cirrhosis.",
        ],
        "digestive_system_diseases": [
            "Gastroesophageal reflux disease was diagnosed after endoscopy examination showing esophagitis.",
            "Colonoscopy identified inflammatory bowel disease with ulcerative colitis.",
            "The patient has chronic pancreatitis with recurrent episodes of abdominal pain.",
        ],
        "nervous_system_diseases": [
            "MRI revealed a demyelinating lesion consistent with multiple sclerosis in the white matter.",
            "Neurological examination indicated signs of Parkinson's disease progression with tremor.",
            "EEG confirmed epileptiform activity consistent with temporal lobe epilepsy.",
        ],
        "cardiovascular_diseases": [
            "Echocardiography confirmed severe aortic stenosis with reduced ejection fraction.",
            "Coronary angiography demonstrated significant stenosis of the left anterior descending artery.",
            "The patient presented with acute myocardial infarction with ST-segment elevation.",
        ],
        "general_pathological_conditions": [
            "Laboratory findings were consistent with systemic lupus erythematosus with renal involvement.",
            "Histopathology showed chronic inflammatory changes with granuloma formation.",
            "The patient developed sepsis with multi-organ dysfunction syndrome.",
        ],
    }


# ---------------------------------------------------------------------------
# Edge case fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_diagnosis():
    """Return an empty diagnosis for testing validation."""
    return ""


@pytest.fixture
def whitespace_only_diagnosis():
    """Return a whitespace-only diagnosis for testing validation."""
    return "   "


@pytest.fixture
def very_long_diagnosis():
    """Return a very long diagnosis for testing length validation."""
    return "This is a very long medical abstract. " * 1000


@pytest.fixture
def special_characters_diagnosis():
    """Return a diagnosis with special characters."""
    return "Patient with @#$% special characters & symbols! in diagnosis."


@pytest.fixture
def multilingual_diagnosis():
    """Return a diagnosis with multilingual content."""
    return "Patient présenté avec néoplasme malin du côlon nécessitant une résection chirurgicale."
