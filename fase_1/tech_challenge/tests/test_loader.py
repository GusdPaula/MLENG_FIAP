import numpy as np
import pandas as pd
import pytest

from src.data.loader import TelcoDataLoader


@pytest.fixture
def sample_csv_path(tmp_path):
    """
    Creates a dummy Telco Churn CSV.
    CRITICAL: Every category value appears at least TWICE to ensure
    they exist in both Train and Test sets after a split.
    """
    path = tmp_path / "dummy_telco.csv"

    data = {
        "customerid": [str(i) for i in range(10)],
        "count": [1] * 10,
        "country": ["Brazil"] * 10,
        "state": ["SP"] * 10,
        "city": ["Sampa"] * 10,
        "zip_code": [12345] * 10,
        "lat_long": ["0,0"] * 10,
        "latitude": [0.0] * 10,
        "longitude": [0.0] * 10,
        # Ensure categories have at least 2 instances
        "gender": ["Female", "Male"] * 5,
        "senior_citizen": ["No", "Yes"] * 5,
        "partner": ["Yes", "No"] * 5,
        "dependents": ["No", "No"] * 5,
        "tenure_months": [1, 24, 12, 6, 72, 1, 24, 12, 6, 72],
        "phone_service": ["Yes"] * 10,
        "multiple_lines": ["No", "Yes"] * 5,
        "internet_service": ["Fiber optic", "DSL"] * 5,
        "online_security": ["No", "Yes"] * 5,
        "online_backup": ["Yes", "No"] * 5,
        "device_protection": ["No", "Yes"] * 5,
        "tech_support": ["No", "Yes"] * 5,
        "streaming_tv": ["No", "Yes"] * 5,
        "streaming_movies": ["No", "Yes"] * 5,
        "contract": ["Month-to-month", "One year"] * 5,
        "paperless_billing": ["Yes", "No"] * 5,
        "payment_method": ["Electronic check", "Mailed check"] * 5,
        "monthly_charges": [70.0, 50.0] * 5,
        "total_charges": [70.0, 1200.0] * 5,
        "churn_label": ["Yes", "No"] * 5,
        "Churn Value": [1, 0] * 5,  # Renamed to churn_value by loader
        "churn_score": [80, 20] * 5,
        "cltv": [2000, 5000] * 5,
        "churn_reason": ["Competitor", None] * 5,
    }
    pd.DataFrame(data).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def loader(sample_csv_path):
    return TelcoDataLoader(sample_csv_path)


def test_pipeline_completo(loader):
    """
    Test the full split and transform logic.
    Using 10 rows ensures the stratified split (80/20) will put
    representative labels in both sets.
    """
    # X_train, X_test, y_train, y_test
    res = loader.pipeline_completo(test_size=0.2)
    X_train, X_test, _y_train, _y_test = res

    # 1. Check if categorization worked (objects converted to numbers)
    assert not X_train.select_dtypes(include=["object"]).columns.any()
    assert not X_test.select_dtypes(include=["object"]).columns.any()

    # 2. Check shapes (8 rows train, 2 rows test)
    assert X_train.shape[0] == 8
    assert X_test.shape[0] == 2

    # 3. Check scaling (StandardScaler output)
    # The mean of a scaled column in the training set should be nearly 0
    assert np.isclose(X_train["monthly_charges"].mean(), 0, atol=1e-5)


def test_inference_flow(loader):
    """Verifies that fit_for_inference sets the internal state correctly."""
    loader.fit_for_inference()
    assert loader._fitted_for_inference is True
    assert len(loader.feature_names) == 19

    # Test a single prediction transform
    sample = {
        "gender": "Female",
        "senior_citizen": "No",
        "partner": "Yes",
        "dependents": "No",
        "tenure_months": 12,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 75.0,
        "total_charges": 900.0,
    }

    X_inference = loader.transform_single(sample)
    assert X_inference.shape == (1, 19)
    assert isinstance(X_inference, np.ndarray)
