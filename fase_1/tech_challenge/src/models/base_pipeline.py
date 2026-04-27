"""Orquestrador de treinamento com MLflow.

Integra model_training.py (lógica pura) com MLflow para:
- Logging automático de parâmetros e métricas
- Registro de modelos no MLflow Model Registry
- Rastreamento de experimentos

Uso:
    python src/models/base_pipeline.py
    # OU
    MLFLOW_TRACKING_URI=http://localhost:5000 python src/models/base_pipeline.py
"""

import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.loader import TelcoDataLoader
from src.evaluation.metrics import TelcoMetrics
from src.models.baseline import BaselineExperiment


def train_production_model():
    """Treina o modelo LogisticRegression balanceado para produção.
    
    Returns:
        tuple: (modelo, X_test, y_test, metricas)
    """
    print("\n" + "="*70)
    print("TELCO CHURN: Pipeline de Treinamento com MLflow")
    print("="*70)
    
    # 1. Configurar MLflow
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI", 
        "http://localhost:5000"
    )
    mlflow.set_tracking_uri(tracking_uri)
    print(f"\n📊 MLflow Tracking URI: {tracking_uri}")
    
    # Criar experimento ou usar existente
    experiment_name = "Telco_Churn_Production"
    mlflow.set_experiment(experiment_name)
    print(f"📁 Experimento: {experiment_name}")
    
    # 2. Carregar dados
    print("\n📂 Carregando dados...")
    data_path = "data/processed/telco_churn_processed.csv"
    loader = TelcoDataLoader(data_path)
    X_train, X_test, y_train, y_test = loader.pipeline_completo()
    
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  Churn rate (train): {y_train.mean():.2%}")
    print(f"  Churn rate (test): {y_test.mean():.2%}")
    
    # 3. Iniciar MLflow run
    print("\n🚀 Iniciando MLflow run...")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"  Run ID: {run_id}")
        
        # 4. Treinar modelo
        print("\n🤖 Treinando LogisticRegression (balanced)...")
        logreg = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
            n_jobs=-1
        )
        
        # Usar BaselineExperiment para consistência
        exp = BaselineExperiment("Telco - Production Pipeline")
        modelo, metricas = exp.treinar_modelo(
            logreg,
            X_train,
            X_test,
            y_train,
            y_test,
            nome_modelo="LogisticRegression-Production"
        )
        
        print("\n✅ Modelo treinado com sucesso!")
        print(f"  Tipo: LogisticRegression")
        print(f"  Classe: {type(modelo).__name__}")
        
        # 5. Log de parâmetros
        print("\n📋 Logging parâmetros...")
        params = {
            "model_type": "LogisticRegression",
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": 42,
            "solver": "lbfgs",
            "n_jobs": -1,
            "train_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
            "n_features": X_train.shape[1],
        }
        mlflow.log_params(params)
        print(f"  {len(params)} parâmetros registrados")
        
        # 6. Log de métricas
        print("\n📊 Logging métricas...")
        metrics_to_log = {
            "test_auc_roc": float(metricas.get("test_auc_roc", 0)),
            "test_pr_auc": float(metricas.get("test_pr_auc", 0)),
            "test_accuracy": float(metricas.get("test_accuracy", 0)),
            "test_precision": float(metricas.get("test_precision", 0)),
            "test_recall": float(metricas.get("test_recall", 0)),
            "test_f1_score": float(metricas.get("test_f1_score", 0)),
        }
        
        # Adicionar métricas de negócio se disponíveis
        if "business_net_benefit" in metricas:
            metrics_to_log["business_net_benefit"] = float(
                metricas["business_net_benefit"]
            )
        if "business_roi" in metricas:
            metrics_to_log["business_roi"] = float(metricas["business_roi"])
        
        mlflow.log_metrics(metrics_to_log)
        print(f"  {len(metrics_to_log)} métricas registradas")
        
        # 7. Exibir métricas
        print("\n📈 Métricas de Teste:")
        print(f"  AUC-ROC:   {metricas.get('test_auc_roc', 0):.4f}")
        print(f"  PR-AUC:    {metricas.get('test_pr_auc', 0):.4f}")
        print(f"  Acurácia:  {metricas.get('test_accuracy', 0):.4f}")
        print(f"  Precisão:  {metricas.get('test_precision', 0):.4f}")
        print(f"  Recall:    {metricas.get('test_recall', 0):.4f}")
        print(f"  F1-Score:  {metricas.get('test_f1_score', 0):.4f}")
        
        if "business_net_benefit" in metricas:
            print(f"\n💰 Métricas de Negócio:")
            print(f"  Net Benefit: ${metricas['business_net_benefit']:.2f}")
            print(f"  ROI: {metricas.get('business_roi', 0):.2f}%")
        
        # 8. Log do modelo no MLflow
        print("\n💾 Logging modelo no MLflow...")
        mlflow.sklearn.log_model(
            modelo,
            "logistic_regression_model",
            registered_model_name="TelcoChurnLogisticRegression"
        )
        print("  Modelo logged com sucesso!")
        
        # 9. Tag para identificação
        mlflow.set_tag("model_type", "LogisticRegression")
        mlflow.set_tag("environment", "production")
        mlflow.set_tag("team", "MLOps")
        
        print("\n" + "="*70)
        print("✅ PIPELINE CONCLUÍDO COM SUCESSO!")
        print("="*70)
        print(f"\n📊 Acesse MLflow UI: {tracking_uri}")
        print(f"   Run ID: {run_id}")
        print(f"   Experimento: {experiment_name}")
        print("\n💡 Próximas etapas:")
        print("   1. Validar métricas no MLflow UI")
        print("   2. Transicionar modelo para 'Staging'")
        print("   3. Executar testes de inferência")
        print("   4. Promover para 'Production'")
        print("\n")
        
        return modelo, X_test, y_test, metricas


if __name__ == "__main__":
    try:
        train_production_model()
    except Exception as e:
        print(f"\n❌ Erro durante treinamento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
