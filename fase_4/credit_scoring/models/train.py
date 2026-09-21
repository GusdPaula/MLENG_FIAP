"""Treinamento do modelo baseline de Credit Scoring.

Modelos: LogisticRegression (baseline) e XGBClassifier (campeão).
Rastreamento via MLflow. Artefato salvo em artifacts/model.pkl.
"""

import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
FEATURE_COLS = [
    "idade",
    "anos_emprego",
    "renda_mensal",
    "valor_emprestimo",
    "score_credito",
    "possui_conta_poupanca",
    "historico_cod",
    "finalidade_cod",
]
TARGET_COL = "inadimplente"


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hist_map = {"excelente": 4, "bom": 3, "regular": 2, "ruim": 1}
    df["historico_cod"] = df["historico_pagamentos"].map(hist_map).fillna(0).astype(int)

    le = LabelEncoder()
    df["finalidade_cod"] = le.fit_transform(df["finalidade"].fillna("desconhecido"))
    return df


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_proba),
    }


def train_baseline(df: pd.DataFrame, experiment_name: str = "credit_scoring_baseline") -> dict:
    """Treina LogisticRegression e XGBClassifier, loga no MLflow, salva campeão."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df_enc = _encode_categoricals(df)
    X = df_enc[FEATURE_COLS]
    y = df_enc[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment(experiment_name)

    results = {}

    # --- Logistic Regression baseline ---
    with mlflow.start_run(run_name="logistic_regression"):
        lr_pipeline = Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42))]
        )
        lr_pipeline.fit(X_train, y_train)
        y_pred_lr = lr_pipeline.predict(X_test)
        y_proba_lr = lr_pipeline.predict_proba(X_test)[:, 1]

        metrics_lr = _compute_metrics(y_test.values, y_pred_lr, y_proba_lr)
        mlflow.log_params({"model": "LogisticRegression", "max_iter": 1000})
        mlflow.log_metrics(metrics_lr)
        mlflow.sklearn.log_model(lr_pipeline, "model")

        results["logistic_regression"] = {"pipeline": lr_pipeline, "metrics": metrics_lr}
        logger.info("LR — accuracy=%.3f, AUC=%.3f", metrics_lr["accuracy"], metrics_lr["auc_roc"])

    # --- XGBoost ---
    with mlflow.start_run(run_name="xgboost"):
        xgb_params = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "random_state": 42,
            "eval_metric": "logloss",
        }
        xgb = XGBClassifier(**xgb_params)
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        y_proba_xgb = xgb.predict_proba(X_test)[:, 1]

        metrics_xgb = _compute_metrics(y_test.values, y_pred_xgb, y_proba_xgb)
        mlflow.log_params(xgb_params)
        mlflow.log_metrics(metrics_xgb)
        mlflow.sklearn.log_model(xgb, "model")

        fi = dict(zip(FEATURE_COLS, xgb.feature_importances_.tolist(), strict=False))
        mlflow.log_dict(fi, "feature_importances.json")

        results["xgboost"] = {"pipeline": xgb, "metrics": metrics_xgb}
        logger.info("XGB — accuracy=%.3f, AUC=%.3f", metrics_xgb["accuracy"], metrics_xgb["auc_roc"])

    # Seleciona campeão pelo AUC
    champion_name = max(results, key=lambda k: results[k]["metrics"]["auc_roc"])
    champion = results[champion_name]["pipeline"]
    champion_path = ARTIFACTS_DIR / "model.pkl"
    joblib.dump({"model": champion, "feature_cols": FEATURE_COLS, "name": champion_name}, champion_path)
    logger.info("Campeão: %s — salvo em %s", champion_name, champion_path)

    return {
        "champion_name": champion_name,
        "champion_path": champion_path,
        "champion_model": champion,
        "metrics": results[champion_name]["metrics"],
        "all_results": results,
        "feature_cols": FEATURE_COLS,
        "X_test": X_test,
        "y_test": y_test,
    }


def predict(model, df: pd.DataFrame) -> np.ndarray:
    """Gera predições para um DataFrame (aplica encoding primeiro)."""
    df_enc = _encode_categoricals(df)
    feature_cols = [c for c in FEATURE_COLS if c in df_enc.columns]
    return model.predict(df_enc[feature_cols])
