import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath('')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
from sklearn.linear_model import LogisticRegression

from data.loader import TelcoDataLoader
from baseline import BaselineExperiment

import pandas as pd
import mlflow



data_path = "data/processed/telco_churn_processed.csv"

loader = TelcoDataLoader(data_path)
X_train, X_test, y_train, y_test = loader.pipeline_completo()

print(f"Dataset shape:")
print(f"  X_train: {X_train.shape}")
print(f"  X_test: {X_test.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  y_test: {y_test.shape}")
print(f"\nChurn rate (train): {y_train.mean():.2%}")
print(f"Churn rate (test): {y_test.mean():.2%}")



print(f"Dados preparados:")
print(f"  Features: {X_train.shape[1]}")
print(f"  Train samples: {X_train.shape[0]}")
print(f"  Test samples: {X_test.shape[0]}")
print(f"  Seed: 42")
print(f"  Stratified: True")



if mlflow.active_run():
    mlflow.end_run()

exp = BaselineExperiment("Telco - Baseline com Scaling")



# 3. LogisticRegression (balanced)
print("\n[3/7] Treinando LogisticRegression (balanced)...")
logreg_balanced = LogisticRegression(
    max_iter=1000, class_weight="balanced", random_state=42
)
modelo, metricas = exp.treinar_modelo(
    logreg_balanced,
    X_train,
    X_test,
    y_train,
    y_test,
    nome_modelo="LogisticRegression-balanced",
)


print("\nMétricas LogisticRegression (balanced):")
print(f"  AUC-ROC: {metricas['test_auc_roc']:.4f}")
print(f"  PR-AUC: {metricas['test_pr_auc']:.4f}")
print(f"  Acurácia: {metricas['test_accuracy']:.4f}")       
print(f"  Precisão: {metricas['test_precision']:.4f}")
print(f"  Recall: {metricas['test_recall']:.4f}")
print(f"  F1-Score: {metricas['test_f1_score']:.4f}")



### Salvando modelo no MLflow
mlflow.sklearn.log_model(modelo, "logistic_regression_balanced")
print("\nModelo LogisticRegression (balanced) salvo no MLflow.")
