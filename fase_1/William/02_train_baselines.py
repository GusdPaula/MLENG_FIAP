#!/usr/bin/env python
"""
Script de Treinamento de Baselines

Etapas:
1. Carregar e preparar dados
2. Definir métricas técnicas e de negócio
3. Treinar DummyClassifier e LogisticRegression
4. Registrar experimentos no MLflow
5. Comparar resultados
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.data.loader import TelcoDataLoader
from src.evaluation.metrics import TelcoMetrics
from src.models.baseline import BaselineExperiment
import pandas as pd


def main():
    print("\n" + "="*70)
    print("ETAPA 3: TREINAMENTO DE BASELINES COM MLFLOW")
    print("="*70)

    # ========== PASSO 1: CARREGAR DADOS ==========
    print("\n[1/5] Carregando e preparando dados...")
    data_path = "data/processed/telco_churn_processed.csv"

    loader = TelcoDataLoader(data_path)
    X_train, X_test, y_train, y_test = loader.pipeline_completo()

    # ========== PASSO 2: DEFINIR METRICAS ==========
    print("\n[2/5] Metricas definidas:")
    print("\n  TECNICAS:")
    print("    [OK] AUC-ROC (Area Under ROC Curve)")
    print("    [OK] PR-AUC (Precision-Recall AUC)")
    print("    [OK] F1-Score")
    print("    [OK] Recall (Sensibilidade)")
    print("    [OK] Precision (Especificidade)")
    print("    [OK] Acuracia")

    print("\n  DE NEGOCIO:")
    print("    [OK] Churn Avoided Revenue: Receita economizada ao reter cliente")
    print(f"      - Lifetime Value por cliente: ${TelcoMetrics.CUSTOMER_LTV}")
    print("    [OK] False Positive Cost: Custo de campanhas ineficazes")
    print(f"      - Custo por campanha: ${TelcoMetrics.FALSE_POSITIVE_COST}")
    print("    [OK] Retention Cost: Total gasto em retencao")
    print(f"      - Custo por retencao: ${TelcoMetrics.RETENTION_COST}")
    print("    [OK] Net Benefit: Lucro liquido")
    print("    [OK] ROI: Retorno sobre investimento")

    # ========== PASSO 3: TREINAR BASELINES ==========
    print("\n[3/5] Treinando baselines com MLflow...")

    exp = BaselineExperiment(experiment_name="Telco Churn - Baselines")

    # DummyClassifier
    dummy_model, dummy_metrics = exp.treinar_dummy_baseline(
        X_train, X_test, y_train, y_test, strategy='stratified'
    )

    # LogisticRegression
    lr_model, lr_metrics = exp.treinar_logistic_regression(
        X_train, X_test, y_train, y_test
    )

    # ========== PASSO 4: CALCULAR METRICAS DE NEGOCIO ==========
    print("\n[4/5] Calculando metricas de negocio...")

    # Adicionar metricas de negocio para cada modelo
    dummy_pred = dummy_model.predict(X_test)
    dummy_business_metrics = TelcoMetrics.calcular_metricas_negocio(y_test, dummy_pred)

    lr_pred = lr_model.predict(X_test)
    lr_business_metrics = TelcoMetrics.calcular_metricas_negocio(y_test, lr_pred)

    # ========== PASSO 5: COMPARAR RESULTADOS ==========
    print("\n[5/5] Comparando e resumindo resultados...")

    resultados = {
        'DummyClassifier': dummy_metrics,
        'LogisticRegression': lr_metrics,
    }

    df_comparacao = exp.comparar_baselines(resultados)

    # ========== RESUMO FINAL ==========
    print("\n" + "="*70)
    print("RESUMO DE NEGOCIO - MODELOS BASELINE")
    print("="*70 + "\n")

    print("DummyClassifier (Stratified):")
    print(f"  Churn Evitado (Receita): ${dummy_business_metrics['churn_avoided_revenue']:,.2f}")
    print(f"  Custo de False Positives: ${dummy_business_metrics['false_positive_cost']:,.2f}")
    print(f"  Custo de Retencao: ${dummy_business_metrics['retention_cost']:,.2f}")
    print(f"  Lucro Liquido: ${dummy_business_metrics['net_benefit']:,.2f}")
    print(f"  ROI: {dummy_business_metrics['roi_percent']:.2f}%")

    print("\nLogisticRegression:")
    print(f"  Churn Evitado (Receita): ${lr_business_metrics['churn_avoided_revenue']:,.2f}")
    print(f"  Custo de False Positives: ${lr_business_metrics['false_positive_cost']:,.2f}")
    print(f"  Custo de Retencao: ${lr_business_metrics['retention_cost']:,.2f}")
    print(f"  Lucro Liquido: ${lr_business_metrics['net_benefit']:,.2f}")
    print(f"  ROI: {lr_business_metrics['roi_percent']:.2f}%")

    print("\n" + "="*70)
    print("[OK] Treinamento concluido com sucesso!")
    print(f"[OK] Experimentos registrados no MLflow: ./mlruns/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
