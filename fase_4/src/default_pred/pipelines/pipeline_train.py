import os
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main() -> None:
    from default_pred.config import (
        MODEL_DIR,
        PROCESSED_DATA_DIR,
        RANDOM_STATE,
        RAW_DATA_DIR,
        TARGET_COLUMN,
        TEST_SIZE,
    )
    from default_pred.data.load import load_dataset
    from default_pred.features.engineering import feature_engineering
    from default_pred.features.preprocessing import split_and_scale
    from default_pred.models.train import train_logistic_regression

    # Load and preprocess the dataset
    df = load_dataset(os.path.join(RAW_DATA_DIR, "loan_data.csv"))
    df = feature_engineering(df)
    df.to_pickle(os.path.join(PROCESSED_DATA_DIR, "loan_data_processed.pkl"))

    # Split the dataset into training and testing sets and scale the features
    X_train, X_test, y_train, y_test, scaler, feature_columns = split_and_scale(
        df, target=TARGET_COLUMN, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Save the scaler for future use
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    # Save the feature columns for future use
    feature_columns_path = os.path.join(MODEL_DIR, "feature_columns.pkl")
    joblib.dump(feature_columns, feature_columns_path)

    # Train the logistic regression model and save it
    model, _metrics, _class_report, _conf_matrix = train_logistic_regression(
        X_train, X_test, y_train, y_test, max_iter=1000, class_weight=None
    )
    model_path = os.path.join(MODEL_DIR, "logistic_regression_model.pkl")
    joblib.dump(model, model_path)


if __name__ == "__main__":
    main()
