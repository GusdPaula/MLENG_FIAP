"""
Shared fixtures for the ML Training Pipeline test suite.

Fixtures are organised into three groups:
  1. Valid fixtures  — satisfy all constraints (used in happy-path tests)
  2. Invalid fixtures — violate exactly one constraint each (used in error-path tests)
  3. Trained-pipeline fixture — a minimal sklearn Pipeline fitted on synthetic data
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_LABELS = [1, 2, 3, 4, 5]

_SAMPLE_ABSTRACTS = [
    "The patient presented with a malignant neoplasm of the colon requiring surgery.",
    "Gastroesophageal reflux disease was diagnosed after endoscopy examination.",
    "MRI revealed a lesion consistent with multiple sclerosis in the white matter.",
    "Echocardiography confirmed severe aortic stenosis with reduced ejection fraction.",
    "Histopathology showed chronic inflammatory changes with granuloma formation.",
    "A biopsy of the hepatic tissue revealed hepatocellular carcinoma.",
    "The colonoscopy identified polyps associated with Crohn's disease.",
    "Neurological examination indicated signs of Parkinson's disease progression.",
    "Coronary angiography demonstrated significant stenosis of the left anterior descending artery.",
    "Laboratory findings were consistent with systemic lupus erythematosus.",
]


def _make_valid_dataframe(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame with n_rows rows satisfying all validation constraints."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(1, 6, size=n_rows)  # uniform over {1, 2, 3, 4, 5}
    abstracts = [
        _SAMPLE_ABSTRACTS[i % len(_SAMPLE_ABSTRACTS)] + f" Case number {i}."
        for i in range(n_rows)
    ]
    return pd.DataFrame({
        "condition_label": labels.astype("int64"),
        "medical_abstract": abstracts,
    })


# ---------------------------------------------------------------------------
# Valid fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_train_valid() -> pd.DataFrame:
    """Valid training DataFrame with 1 200 rows (above the 1 000-row minimum)."""
    return _make_valid_dataframe(n_rows=1_200, seed=0)


@pytest.fixture
def df_test_valid() -> pd.DataFrame:
    """Valid test DataFrame with 150 rows (above the 100-row minimum)."""
    return _make_valid_dataframe(n_rows=150, seed=1)


@pytest.fixture
def df_labels_valid() -> pd.DataFrame:
    """Valid labels DataFrame with the five class definitions."""
    return pd.DataFrame({
        "condition_label": [1, 2, 3, 4, 5],
        "condition_name": [
            "neoplasms",
            "digestive system diseases",
            "nervous system diseases",
            "cardiovascular diseases",
            "general pathological conditions",
        ],
    })


# ---------------------------------------------------------------------------
# Invalid fixtures — missing columns
# ---------------------------------------------------------------------------

@pytest.fixture
def df_missing_label_column() -> pd.DataFrame:
    """DataFrame without 'condition_label' column (1 200 rows)."""
    df = _make_valid_dataframe(1_200)
    return df.drop(columns=["condition_label"])


@pytest.fixture
def df_missing_abstract_column() -> pd.DataFrame:
    """DataFrame without 'medical_abstract' column (1 200 rows)."""
    df = _make_valid_dataframe(1_200)
    return df.drop(columns=["medical_abstract"])


# ---------------------------------------------------------------------------
# Invalid fixtures — bad labels
# ---------------------------------------------------------------------------

@pytest.fixture
def df_invalid_labels_train() -> pd.DataFrame:
    """Training DataFrame where some condition_label values are outside {1-5}."""
    df = _make_valid_dataframe(1_200)
    # Inject invalid labels at fixed positions
    df.loc[0, "condition_label"] = 0
    df.loc[1, "condition_label"] = 6
    df.loc[2, "condition_label"] = -1
    return df


@pytest.fixture
def df_invalid_labels_test() -> pd.DataFrame:
    """Test DataFrame where some condition_label values are outside {1-5}."""
    df = _make_valid_dataframe(150)
    df.loc[0, "condition_label"] = 99
    return df


# ---------------------------------------------------------------------------
# Invalid fixtures — bad abstracts
# ---------------------------------------------------------------------------

@pytest.fixture
def df_null_abstract() -> pd.DataFrame:
    """DataFrame with a null value in medical_abstract (1 200 rows)."""
    df = _make_valid_dataframe(1_200)
    df.loc[5, "medical_abstract"] = None
    return df


@pytest.fixture
def df_empty_abstract() -> pd.DataFrame:
    """DataFrame with an empty string in medical_abstract (1 200 rows)."""
    df = _make_valid_dataframe(1_200)
    df.loc[5, "medical_abstract"] = ""
    return df


@pytest.fixture
def df_whitespace_abstract() -> pd.DataFrame:
    """DataFrame with a whitespace-only string in medical_abstract (1 200 rows)."""
    df = _make_valid_dataframe(1_200)
    df.loc[5, "medical_abstract"] = "   "
    return df


# ---------------------------------------------------------------------------
# Invalid fixtures — insufficient size
# ---------------------------------------------------------------------------

@pytest.fixture
def df_train_too_small() -> pd.DataFrame:
    """Valid-content training DataFrame with only 50 rows (below the 1 000-row minimum)."""
    return _make_valid_dataframe(n_rows=50, seed=10)


@pytest.fixture
def df_test_too_small() -> pd.DataFrame:
    """Valid-content test DataFrame with only 10 rows (below the 100-row minimum)."""
    return _make_valid_dataframe(n_rows=10, seed=11)


# ---------------------------------------------------------------------------
# Trained pipeline fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def trained_pipeline(df_train_valid: pd.DataFrame) -> Pipeline:
    """A minimal Pipeline (TF-IDF + LinearSVC) fitted on df_train_valid.

    This fixture is used by evaluator tests and integration tests so they
    don't have to re-train a model from scratch.
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 1),
            min_df=1,       # min_df=1 to work with small synthetic data
            max_df=0.95,
            lowercase=True,
        )),
        ("clf", LinearSVC(
            C=0.06,
            random_state=42,
            dual=False,
        )),
    ])
    pipeline.fit(
        df_train_valid["medical_abstract"],
        df_train_valid["condition_label"],
    )
    return pipeline
