from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"

TARGET_COLUMN = "loan_status"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Observability (Stage 3)
LOG_DIR = PROJECT_ROOT / "logs"

MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_ARTIFACT_ROOT = PROJECT_ROOT / "mlruns"
EXPERIMENT_TRAINING = "credit_training"
EXPERIMENT_DRIFT = "credit_drift_monitoring"
EXPERIMENT_INFERENCE = "credit_inference"

DRIFT_ALERT_SHARE_THRESHOLD = 0.5
DRIFT_REPORT_PATH = PROJECT_ROOT / "data_drift_report.html"
