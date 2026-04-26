"""Modelos baseline para Telco Churn com MLflow tracking."""

import os

import mlflow
import mlflow.data
import mlflow.sklearn
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier


class MLPWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper sklearn-compatível para MLP PyTorch.
    Permite usar MLP junto com sklearn Pipeline.
    """

    def __init__(
        self,
        input_size=19,
        hidden_sizes=None,
        dropout_rates=None,
        epochs=100,
        learning_rate=0.001,
        early_stopping_patience=10,
        batch_size=32,
        random_state=42,
    ):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes if hidden_sizes is not None else [128, 64, 32]
        self.dropout_rates = dropout_rates if dropout_rates is not None else [0.3, 0.2, 0.0]
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_network(self):
        """Constrói arquitetura MLP."""
        layers = []
        prev_size = self.input_size

        for hidden_size, dropout_rate in zip(self.hidden_sizes, self.dropout_rates, strict=False):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        """Treina MLP com Early Stopping."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Preparar dados
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(
            y_train.values if hasattr(y_train, "values") else y_train, dtype=torch.long
        )
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(
            y_val.values if hasattr(y_val, "values") else y_val, dtype=torch.long
        )

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Construir modelo
        self.model = self._build_network().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        pos_weight = torch.tensor(
            np.sum(y_train == 0) / np.sum(y_train == 1), dtype=torch.float32
        ).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Treinar com Early Stopping
        best_val_loss = float("inf")
        patience_counter = 0

        for _epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).unsqueeze(1).float()

                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)

            train_loss /= len(train_loader.dataset)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device).unsqueeze(1).float()
                    logits = self.model(X_batch)
                    loss = criterion(logits, y_batch)
                    val_loss += loss.item() * X_batch.size(0)

            val_loss /= len(val_loader.dataset)

            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                break

        return self

    def predict(self, X):
        """Prediz classes (0 ou 1)."""
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            proba = torch.sigmoid(logits).cpu().numpy().flatten()
        return (proba > 0.5).astype(int)

    def predict_proba(self, X):
        """Retorna probabilidades para ambas classes."""
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            proba_class_1 = torch.sigmoid(logits).cpu().numpy().flatten()
        proba_class_0 = 1 - proba_class_1
        return np.column_stack([proba_class_0, proba_class_1])

    def get_params(self, deep=True):
        """Retorna parâmetros (compatível com sklearn)."""
        return {
            "input_size": self.input_size,
            "hidden_sizes": self.hidden_sizes,
            "dropout_rates": self.dropout_rates,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "early_stopping_patience": self.early_stopping_patience,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        """Define parâmetros (compatível com sklearn)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class BaselineExperiment:
    """Treina e registra modelos baseline com MLflow."""

    def __init__(self, experiment_name="Telco Churn - Baselines", mlflow_uri="../mlruns"):
        """
        Inicializa experimento MLflow.

        Args:
            experiment_name: Nome do experimento
            mlflow_uri: URI para armazenar artefatos MLflow
        """
        # Criar pasta se não existir
        os.makedirs(mlflow_uri, exist_ok=True)

        self.experiment_name = experiment_name
        self.mlflow_uri = mlflow_uri
        mlflow.set_tracking_uri(mlflow_uri)

        # Obter ou criar experimento
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)

        mlflow.set_experiment(experiment_name)
        self.resultados = {}
        self._modelos = {}  # Armazenar modelos treinados

    def treinar_modelo(
        self,
        modelo,
        X_train,
        X_test,
        y_train,
        y_test,
        nome_modelo: str,
        X_train_modificado=None,
        y_train_modificado=None,
    ) -> tuple:
        """
        Método genérico para treinar qualquer modelo.

        Args:
            modelo: Instância do modelo (sklearn, xgboost, etc)
            X_train: Features de treino
            X_test: Features de teste
            y_train: Labels de treino
            y_test: Labels de teste
            nome_modelo: Nome descritivo do modelo
            X_train_modificado: Features de treino modificadas (ex: SMOTE) - opcional
            y_train_modificado: Labels de treino modificados (ex: SMOTE) - opcional

        Returns:
            (modelo_treinado, metricas)
        """
        # Encerrar run ativo se existir
        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run(run_name=nome_modelo):
            # Usar dados modificados se fornecidos (ex: SMOTE)
            X_treino = X_train_modificado if X_train_modificado is not None else X_train
            y_treino = y_train_modificado if y_train_modificado is not None else y_train

            # Log de parâmetros do modelo
            parametros = modelo.get_params()
            for param_name, param_value in parametros.items():
                mlflow.log_param(param_name, param_value)

            # Dataset info
            mlflow.log_dict(
                {
                    "churn_rate_train": float(y_treino.mean()),
                    "churn_rate_test": float(y_test.mean()),
                    "n_train_samples": X_treino.shape[0],
                    "n_test_samples": X_test.shape[0],
                    "n_features": X_treino.shape[1],
                },
                artifact_file="data_info.json",
            )

            # Treinar modelo
            modelo.fit(X_treino, y_treino)

            # Predicoes
            y_pred = modelo.predict(X_test)
            y_pred_proba = modelo.predict_proba(X_test)[:, 1]

            # Calcular métricas
            metricas = {
                "test_auc_roc": roc_auc_score(y_test, y_pred_proba),
                "test_pr_auc": average_precision_score(y_test, y_pred_proba),
                "test_accuracy": accuracy_score(y_test, y_pred),
                "test_precision": precision_score(y_test, y_pred),
                "test_recall": recall_score(y_test, y_pred),
                "test_f1_score": f1_score(y_test, y_pred),
            }

            mlflow.log_metrics(metricas)

            # Adding model info to MLflow artifacts
            mlflow.log_dict(
                {
                    "model_type": type(modelo).__name__,
                    "parameters": parametros,
                    "metrics": metricas,
                },
                artifact_file="model_info.json",
            )

            mlflow.log_dict(
                {"classification_report": classification_report(y_test, y_pred, output_dict=True)},
                artifact_file="classification_report.json",
            )

            mlflow.log_dict(
                {"confusion_matrix": confusion_matrix(y_test, y_pred).tolist()},
                artifact_file="confusion_matrix.json",
            )

            # 1. Ensure X_train is a DataFrame
            if not isinstance(X_train, pd.DataFrame):
                train_df = pd.DataFrame(X_train)
            else:
                train_df = X_train.copy()

            # 2. Assign the target safely
            if hasattr(y_train, "values"):
                train_df["churn_value"] = y_train.values
            else:
                train_df["churn_value"] = y_train

            # Repeat the same logic for test_df
            if not isinstance(X_test, pd.DataFrame):
                test_df = pd.DataFrame(X_test)
            else:
                test_df = X_test.copy()

            test_df["churn_value"] = y_test.values if hasattr(y_test, "values") else y_test
            dataset_train = mlflow.data.from_pandas(
                train_df, targets="churn_value", name="churn_train"
            )
            dataset_test = mlflow.data.from_pandas(
                test_df, targets="churn_value", name="churn_test"
            )

            mlflow.log_input(dataset_train, context="training")
            mlflow.log_input(dataset_test, context="testing")

            # Save to local temporary CSVs
            pd.DataFrame(X_train).to_csv("X_train.csv", index=False)
            pd.DataFrame(X_test).to_csv("X_test.csv", index=False)
            pd.DataFrame(y_train).to_csv("y_train.csv", index=False)
            pd.DataFrame(y_test).to_csv("y_test.csv", index=False)

            # Log the files to MLflow
            mlflow.log_artifact("X_train.csv", artifact_path="data")
            mlflow.log_artifact("X_test.csv", artifact_path="data")
            mlflow.log_artifact("y_train.csv", artifact_path="data")
            mlflow.log_artifact("y_test.csv", artifact_path="data")

            # Remove files after logging
            os.remove("X_train.csv")
            os.remove("X_test.csv")
            os.remove("y_train.csv")
            os.remove("y_test.csv")

            # Exibir resultados
            print(f"\n[OK] {nome_modelo} treinado")
            print(f"  - AUC-ROC: {metricas['test_auc_roc']:.4f}")
            print(f"  - PR-AUC: {metricas['test_pr_auc']:.4f}")
            print(f"  - Acurácia: {metricas['test_accuracy']:.4f}")
            print(f"  - Precision: {metricas['test_precision']:.4f}")
            print(f"  - Recall: {metricas['test_recall']:.4f}")
            print(f"  - F1-Score: {metricas['test_f1_score']:.4f}")

            # Log de feature importance (se disponível)
            if hasattr(modelo, "feature_importances_"):
                feature_importance = pd.DataFrame(
                    {"feature": range(X_treino.shape[1]), "importance": modelo.feature_importances_}
                ).sort_values("importance", ascending=False)

                mlflow.log_dict(
                    feature_importance.to_dict(), artifact_file="feature_importance.json"
                )

                print("  - Top 5 features (importância):")
                for _idx, row in feature_importance.head(5).iterrows():
                    print(f"    Feature {row['feature']}: {row['importance']:.4f}")

            # Log de coeficientes (se disponível)
            elif hasattr(modelo, "coef_"):
                coeficientes = pd.DataFrame(
                    {"feature": range(X_treino.shape[1]), "coef": modelo.coef_[0]}
                ).sort_values("coef", ascending=False)

                mlflow.log_dict(coeficientes.to_dict(), artifact_file="top_features.json")

                print("  - Top 5 features (coeficientes):")
                for _idx, row in coeficientes.head(5).iterrows():
                    print(f"    Feature {row['feature']}: {row['coef']:.4f}")

            # Registrar modelo
            mlflow.sklearn.log_model(modelo, artifact_path="model")
            self.resultados[nome_modelo] = metricas

            # Armazenar modelo em memória para análise posterior
            self._modelos[nome_modelo] = modelo

            return modelo, metricas

    def treinar_esteira_completa(
        self, X_train, X_test, y_train, y_test, include_tuning=False, n_iter_tuning=5, cv_tuning=3
    ):
        """
        Treina todos os modelos em uma esteira usando método genérico.

        Args:
            X_train, X_test, y_train, y_test: Dados de treino e teste
            include_tuning: Se True, inclui modelos com tuning (mais lento)
            n_iter_tuning: Número de iterações para RandomizedSearchCV
            cv_tuning: Número de folds para cross-validation

        Returns:
            DataFrame com comparação de todos os modelos
        """
        print("\n" + "=" * 70)
        print("INICIANDO ESTEIRA DE MODELOS")
        print("=" * 70)

        # 1. DummyClassifier (most_frequent)
        print("\n[1/6] Treinando DummyClassifier (most_frequent)...")
        dummy_clf = DummyClassifier(strategy="most_frequent", random_state=42)
        self.treinar_modelo(
            dummy_clf, X_train, X_test, y_train, y_test, nome_modelo="DummyClassifier-most_frequent"
        )

        # 2. LogisticRegression (simples)
        print("\n[2/6] Treinando LogisticRegression (simples)...")
        logreg_clf = LogisticRegression(max_iter=1000, random_state=42)
        self.treinar_modelo(
            logreg_clf, X_train, X_test, y_train, y_test, nome_modelo="LogisticRegression-simples"
        )

        # 3. LogisticRegression (balanced)
        print("\n[3/6] Treinando LogisticRegression (balanced)...")
        logreg_balanced = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        )
        self.treinar_modelo(
            logreg_balanced,
            X_train,
            X_test,
            y_train,
            y_test,
            nome_modelo="LogisticRegression-balanced",
        )

        # 4. LogisticRegression (com SMOTE)
        print("\n[4/6] Treinando LogisticRegression com SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        logreg_smote = LogisticRegression(max_iter=1000, random_state=42)
        self.treinar_modelo(
            logreg_smote,
            X_train,
            X_test,
            y_train,
            y_test,
            nome_modelo="LogisticRegression-SMOTE",
            X_train_modificado=X_train_smote,
            y_train_modificado=y_train_smote,
        )

        # 5. RandomForest
        print("\n[5/6] Treinando RandomForestClassifier...")
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.treinar_modelo(
            rf_clf, X_train, X_test, y_train, y_test, nome_modelo="RandomForestClassifier"
        )

        # 6. XGBoost
        print("\n[6/6] Treinando XGBoostClassifier...")
        xgb_clf = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        self.treinar_modelo(
            xgb_clf, X_train, X_test, y_train, y_test, nome_modelo="XGBoostClassifier"
        )

        # 7. Tuning (opcional)
        if include_tuning:
            print("\n[7a] Treinando RandomForestClassifier com tuning...")
            self._treinar_com_tuning_rf(
                X_train, X_test, y_train, y_test, n_iter=n_iter_tuning, cv=cv_tuning
            )

            print("\n[7b] Treinando XGBoostClassifier com tuning...")
            self._treinar_com_tuning_xgb(
                X_train, X_test, y_train, y_test, n_iter=n_iter_tuning, cv=cv_tuning
            )

        return self.comparar_baselines(self.resultados)

    def _treinar_com_tuning_rf(self, X_train, X_test, y_train, y_test, n_iter=10, cv=3):
        """Treina RandomForest com tuning de hiperparâmetros."""
        with mlflow.start_run(run_name="RandomForestClassifier-Tuned"):
            param_dist = {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "bootstrap": [True, False],
            }

            print("[INFO] Executando RandomizedSearchCV (pode levar alguns minutos)...")
            rf = RandomForestClassifier(random_state=42)
            random_search = RandomizedSearchCV(
                estimator=rf,
                param_distributions=param_dist,
                n_iter=n_iter,
                cv=cv,
                random_state=42,
                verbose=1,
            )
            random_search.fit(X_train, y_train)

            best_model = random_search.best_estimator_

            # Log dos melhores parâmetros
            for param_name, param_value in random_search.best_params_.items():
                mlflow.log_param(f"best_{param_name}", param_value)

            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]

            metricas = {
                "test_auc_roc": roc_auc_score(y_test, y_pred_proba),
                "test_pr_auc": average_precision_score(y_test, y_pred_proba),
                "test_accuracy": accuracy_score(y_test, y_pred),
                "test_precision": precision_score(y_test, y_pred),
                "test_recall": recall_score(y_test, y_pred),
                "test_f1_score": f1_score(y_test, y_pred),
            }

            mlflow.log_metrics(metricas)
            mlflow.sklearn.log_model(best_model, artifact_path="model")

            print("\n[OK] RandomForestClassifier-Tuned treinado")
            print(f"  - AUC-ROC: {metricas['test_auc_roc']:.4f}")
            print(f"  - Melhores parâmetros: {random_search.best_params_}")

            self.resultados["RandomForestClassifier-Tuned"] = metricas

    def _treinar_com_tuning_xgb(self, X_train, X_test, y_train, y_test, n_iter=10, cv=3):
        """Treina XGBoost com tuning de hiperparâmetros."""
        with mlflow.start_run(run_name="XGBoostClassifier-Tuned"):
            param_dist = {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            }

            print("[INFO] Executando RandomizedSearchCV (pode levar alguns minutos)...")
            xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")
            random_search = RandomizedSearchCV(
                estimator=xgb,
                param_distributions=param_dist,
                n_iter=n_iter,
                cv=cv,
                random_state=42,
                verbose=1,
            )
            random_search.fit(X_train, y_train)

            best_model = random_search.best_estimator_

            # Log dos melhores parâmetros
            for param_name, param_value in random_search.best_params_.items():
                mlflow.log_param(f"best_{param_name}", param_value)

            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]

            metricas = {
                "test_auc_roc": roc_auc_score(y_test, y_pred_proba),
                "test_pr_auc": average_precision_score(y_test, y_pred_proba),
                "test_accuracy": accuracy_score(y_test, y_pred),
                "test_precision": precision_score(y_test, y_pred),
                "test_recall": recall_score(y_test, y_pred),
                "test_f1_score": f1_score(y_test, y_pred),
            }

            mlflow.log_metrics(metricas)
            mlflow.sklearn.log_model(best_model, artifact_path="model")

            print("\n[OK] XGBoostClassifier-Tuned treinado")
            print(f"  - AUC-ROC: {metricas['test_auc_roc']:.4f}")
            print(f"  - Melhores parâmetros: {random_search.best_params_}")

            print(metricas)

            self.resultados["XGBoostClassifier-Tuned"] = metricas

    def comparar_baselines(self, resultados: dict) -> pd.DataFrame:
        """
        Compara resultados de todos os baselines.

        Args:
            resultados: Dicionário com resultados de cada modelo

        Returns:
            DataFrame com comparação
        """
        print("\n" + "=" * 70)
        print("COMPARAÇÃO DE BASELINES")
        print("=" * 70 + "\n")

        df_comparacao = pd.DataFrame(resultados).T
        df_comparacao = df_comparacao.sort_values("test_auc_roc", ascending=False)

        print(df_comparacao.to_string())
        print("\n" + "=" * 70)

        return df_comparacao

    def treinar_esteira_controlada(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        aplicar_scaling=True,
        include_mlp=True,
        include_xgb_tuned=False,
    ):
        """
        Treina modelos baseline em condições controladas (mesmos dados, mesmo scaling).

        Args:
            X_train, X_test, y_train, y_test: Dados não-escalados
            aplicar_scaling: Se True, aplica StandardScaler
            include_mlp: Se True, inclui MLPWrapper na comparação
            include_xgb_tuned: Se True, inclui XGBoost com hiperparâmetros tuned

        Returns:
            Tupla (DataFrame com modelos, scaler_object)
        """

        # Encerrar qualquer run ativo
        if mlflow.active_run():
            mlflow.end_run()

        # Limpar resultados anteriores
        self.resultados = {}
        self._modelos = {}  # Limpar modelos anteriores

        # Preparar dados
        scaler = None
        if aplicar_scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            scaling_info = "com StandardScaler"
        else:
            scaling_info = "sem scaling"

        print("\n" + "=" * 70)
        print("INICIANDO ESTEIRA CONTROLADA DE MODELOS")
        print(f"Scaling: {scaling_info}")
        print("=" * 70)

        # Calcular número total de modelos
        num_modelos = 6  # Base: Dummy, 3x LogReg, RandomForest, XGBoost
        if include_mlp:
            num_modelos += 1
        if include_xgb_tuned:
            num_modelos += 1

        # 1. DummyClassifier
        print("\n[1/7] Treinando DummyClassifier (most_frequent)...")
        dummy_clf = DummyClassifier(strategy="most_frequent", random_state=42)
        self.treinar_modelo(
            dummy_clf, X_train, X_test, y_train, y_test, nome_modelo="DummyClassifier-most_frequent"
        )

        # 2. LogisticRegression (simples)
        print("\n[2/7] Treinando LogisticRegression (simples)...")
        logreg_clf = LogisticRegression(max_iter=1000, random_state=42)
        self.treinar_modelo(
            logreg_clf, X_train, X_test, y_train, y_test, nome_modelo="LogisticRegression-simples"
        )

        # 3. LogisticRegression (balanced)
        print("\n[3/7] Treinando LogisticRegression (balanced)...")
        logreg_balanced = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        )
        self.treinar_modelo(
            logreg_balanced,
            X_train,
            X_test,
            y_train,
            y_test,
            nome_modelo="LogisticRegression-balanced",
        )

        # 4. LogisticRegression (SMOTE)
        print("\n[4/7] Treinando LogisticRegression com SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        logreg_smote = LogisticRegression(max_iter=1000, random_state=42)
        self.treinar_modelo(
            logreg_smote,
            X_train,
            X_test,
            y_train,
            y_test,
            nome_modelo="LogisticRegression-SMOTE",
            X_train_modificado=X_train_smote,
            y_train_modificado=y_train_smote,
        )

        # 5. RandomForest
        print("\n[5/7] Treinando RandomForestClassifier...")
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.treinar_modelo(
            rf_clf, X_train, X_test, y_train, y_test, nome_modelo="RandomForestClassifier"
        )

        # 6. XGBoost
        print("\n[6/7] Treinando XGBoostClassifier...")
        xgb_clf = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        self.treinar_modelo(
            xgb_clf, X_train, X_test, y_train, y_test, nome_modelo="XGBoostClassifier"
        )

        # 7. MLP (opcional)
        if include_mlp:
            print(f"\n[7/{num_modelos}] Treinando MLPWrapper (PyTorch)...")
            mlp_clf = MLPWrapper(input_size=X_train.shape[1], random_state=42)
            self.treinar_modelo(
                mlp_clf, X_train, X_test, y_train, y_test, nome_modelo="MLPWrapper-PyTorch"
            )

        # 8. XGBoost Tuned (opcional)
        if include_xgb_tuned:
            model_idx = 8 if include_mlp else 7
            print(f"\n[{model_idx}/{num_modelos}] Treinando XGBoostClassifier (tuned)...")
            xgb_tuned = XGBClassifier(
                n_estimators=150,  # Aumentado
                max_depth=6,  # Aumentado
                learning_rate=0.05,  # Reduzido (mais conservador)
                subsample=0.8,  # Subsampling para regularização
                colsample_bytree=0.8,  # Feature subsampling
                min_child_weight=1,  # Min samples in leaf
                gamma=0.1,  # L1 regularization
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
            )
            self.treinar_modelo(
                xgb_tuned, X_train, X_test, y_train, y_test, nome_modelo="XGBoostClassifier-tuned"
            )

        # Retornar comparação e scaler
        df_resultados = self.comparar_baselines(self.resultados)
        return df_resultados, scaler
