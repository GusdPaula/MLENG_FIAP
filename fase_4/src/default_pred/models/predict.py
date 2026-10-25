import logging
import os

import joblib
import numpy as np
import pandas as pd

from default_pred.config import MODEL_DIR
from default_pred.features.engineering import feature_engineering

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict(df: pd.DataFrame) -> np.ndarray:
    """Load the trained model and make a prediction for a single entry."""

    logger.info(f"Loading model from {MODEL_DIR}/logistic_regression_model.pkl")
    model_path = os.path.join(MODEL_DIR, "logistic_regression_model.pkl")
    model = joblib.load(model_path)

    logger.info("Loading scaler and feature columns for preprocessing")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    scaler = joblib.load(scaler_path)

    logger.info("Loading feature columns for preprocessing")
    feature_columns_path = os.path.join(MODEL_DIR, "feature_columns.pkl")
    feature_columns = joblib.load(feature_columns_path)

    logger.info("Performing feature engineering on the input data")
    df = feature_engineering(df)

    # Align to the exact training schema before model inference.
    model_columns = list(getattr(model, "feature_names_in_", feature_columns))
    df = df.reindex(columns=model_columns, fill_value=0)

    # Only the scaler-managed columns were transformed; rebuild the frame in a
    # float-safe way so integer-encoded dummy columns remain valid.
    scaler_columns = list(getattr(scaler, "feature_names_in_", []))
    if scaler_columns:
        scaled_values = scaler.transform(df[scaler_columns])
        scaled_df = pd.DataFrame(scaled_values, columns=scaler_columns, index=df.index)

        other_columns = [col for col in model_columns if col not in scaler_columns]
        df = pd.concat([df[other_columns], scaled_df], axis=1)
        df = df.reindex(columns=model_columns)

    try:
        from default_pred.models.drift import detect_data_drift

        logger.info("Checking for data drift before making predictions")
        _result = detect_data_drift(X_test=df)
    except Exception:
        logger.warning(
            "Data drift check failed; continuing with prediction without it.",
            exc_info=True,
        )

    logger.info("Making prediction for the input data")
    # Make prediction
    prediction = model.predict(df)
    logger.info(f"Prediction: {prediction}")

    return prediction
