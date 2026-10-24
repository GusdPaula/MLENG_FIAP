from data.load import load_dataset
from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
)
import os
from features.engineering import feature_engineering
from features.preprocessing import split_and_scale
from models.train import train_logistic_regression
import joblib

df = load_dataset(os.path.join(RAW_DATA_DIR, "loan_data.csv"))
df = feature_engineering(df)
df.to_pickle(os.path.join(PROCESSED_DATA_DIR, "loan_data_processed.pkl"))

X_train, X_test, y_train, y_test, scaler = split_and_scale(
    df, target=TARGET_COLUMN, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

model, _metrics, _class_report, _conf_matrix = train_logistic_regression(
    X_train, X_test, y_train, y_test, max_iter=1000, class_weight=None
)
model_path = os.path.join(MODEL_DIR, "logistic_regression_model.pkl")
joblib.dump(model, model_path)
