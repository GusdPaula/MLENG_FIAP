"""Serviço de inferência e predição para uso em produção.

Carrega pipeline treinado e fornece métodos para predição de churn
em modo single ou batch. Suporta MLflow Model Registry com fallback
para arquivo local (.pkl)."""

import logging
import os
import pickle
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np

from .pipeline import TelcoPipeline

logger = logging.getLogger(__name__)


class PredictionService:
    """Serviço de predição de churn.

    Carrega pipeline treinado (sklearn Pipeline com StandardScaler + classificador)
    e fornece métodos para predição single e batch com suporte a probabilidades.

    Suporta carregamento de:
    - MLflow Model Registry (recomendado para produção)
    - Arquivo local (.pkl) como fallback
    """

    _model = None  # Cache de modelo em memória
    _model_source = None  # "mlflow" ou "local"

    def __init__(
        self,
        pipeline_path: str = None,
        scaler_path: str | None = None,
        preprocessor_path: str | None = None,
        use_mlflow: bool = True,
    ):
        """
        Inicializa serviço de predição.

        Args:
            pipeline_path: Caminho para o arquivo do pipeline treinado (fallback local)
            scaler_path: Caminho para o scaler (opcional)
            preprocessor_path: Caminho para o preprocessor (opcional)
            use_mlflow: Se True, tenta carregar do MLflow primeiro
        """
        self.pipeline = None
        self.scaler = None
        self.preprocessor = None
        self.use_mlflow = use_mlflow

        # Tentar carregar do MLflow se configurado
        if use_mlflow:
            self._load_from_mlflow()

        # Se não carregou do MLflow, tentar arquivo local
        if self.pipeline is None and pipeline_path:
            self._load_from_local(pipeline_path, scaler_path, preprocessor_path)

        if self.pipeline is None:
            raise RuntimeError(
                "Não foi possível carregar o modelo. "
                "Configure MLFLOW_TRACKING_URI ou forneça pipeline_path."
            )

    @classmethod
    def load_model(
        cls,
        pipeline_path: str = None,
        scaler_path: str | None = None,
        preprocessor_path: str | None = None,
    ) -> "PredictionService":
        """Factory method para criar e cachear serviço de predição.

        Reutiliza instância em cache se disponível (boa prática para produção).

        Args:
            pipeline_path: Caminho para arquivo local (fallback)
            scaler_path: Caminho para scaler (opcional)
            preprocessor_path: Caminho para preprocessor (opcional)

        Returns:
            PredictionService: Instância do serviço
        """
        if cls._model is None:
            cls._model = cls(
                pipeline_path=pipeline_path,
                scaler_path=scaler_path,
                preprocessor_path=preprocessor_path,
                use_mlflow=True,
            )
            logger.info(f"✓ Modelo carregado via {cls._model_source}")
        return cls._model

    def _load_from_mlflow(self) -> bool:
        """Tenta carregar modelo do MLflow Model Registry.

        Returns:
            bool: True se carregou com sucesso, False caso contrário
        """
        try:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            mlflow.set_tracking_uri(tracking_uri)

            # Tentar carregar modelo "Production" do registry
            model_uri = "models://TelcoChurnLogisticRegression/Production"
            logger.info(f"Tentando carregar modelo: {model_uri}")

            self.pipeline = mlflow.pyfunc.load_model(model_uri)
            PredictionService._model_source = "MLflow"
            logger.info(f"✓ Modelo carregado via MLflow: {model_uri}")
            return True

        except Exception as e:
            logger.warning(f"⚠ Não foi possível carregar do MLflow: {e}")
            return False

    def _load_from_local(
        self,
        pipeline_path: str,
        scaler_path: str | None = None,
        preprocessor_path: str | None = None,
    ):
        """Carrega modelo do arquivo local (.pkl).

        Args:
            pipeline_path: Caminho para o arquivo do pipeline
            scaler_path: Caminho para o scaler (opcional)
            preprocessor_path: Caminho para o preprocessor (opcional)
        """
        # Carregar pipeline treinado
        if Path(pipeline_path).exists():
            with open(pipeline_path, "rb") as f:
                # TODO: REMOVE
                self.pipeline = pickle.load(f)["model_object"]
            PredictionService._model_source = "local"
            logger.info(f"[OK] Pipeline carregado de {pipeline_path}")
        else:
            raise FileNotFoundError(f"Pipeline não encontrado em {pipeline_path}")

        # Carregar scaler adicional (se fornecido)
        if scaler_path and Path(scaler_path).exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            logger.info(f"[OK] Scaler carregado de {scaler_path}")

        # Carregar preprocessor adicional (se fornecido)
        if preprocessor_path and Path(preprocessor_path).exists():
            with open(preprocessor_path, "rb") as f:
                self.preprocessor = pickle.load(f)
            print(f"[OK] Preprocessor carregado de {preprocessor_path}")

    def preprocess(self, X: np.ndarray) -> np.ndarray:
        """Aplica preprocessamento adicional (se disponível).

        Na maioria dos casos, o preprocessamento já foi feito no feature_transformer.
        """
        if self.preprocessor is not None:
            return self.preprocessor.transform(X)
        return X

    def predict(self, X: np.ndarray, return_proba: bool = False) -> np.ndarray | dict:
        """Prediz churn para amostras de entrada.

        Args:
            X: Array (n_samples, n_features) com features codificadas e normalizadas
            return_proba: Se True, retorna probabilidades e confiança

        Returns:
            predictions: Array com 0 (sem churn) ou 1 (com churn)
            Se return_proba: Dict com predictions, probabilities e confidence
        """
        # Aplicar preprocessamento se necessário
        X_processed = self.preprocess(X)

        # Realizar predição
        y_pred = self.pipeline.predict(X_processed)

        if return_proba:
            y_proba = self.pipeline.predict_proba(X_processed)
            return {
                "predictions": y_pred,
                "probabilities": y_proba[:, 1],  # Probabilidade da classe 1 (churn)
                "confidence": np.max(y_proba, axis=1),
            }

        return y_pred

    def predict_batch(self, X: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        """Prediz churn em lotes para datasets grandes.

        Ûtil para processar muitos exemplos sem sobrecarregar memória.
        """
        predictions = []

        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size]
            batch_pred = self.predict(batch)
            predictions.append(batch_pred)

        return np.concatenate(predictions)

    def predict_single(self, features: dict) -> dict:
        """
        Predição para um único exemplo.

        Args:
            features: Dict com nome da coluna -> valor

        Returns:
            Dict com predição e probabilidade
        """
        # Converter dict para array (ordem importa!)
        feature_array = np.array([list(features.values())])

        pred = self.predict(feature_array, return_proba=True)

        return {
            "churn_prediction": int(pred["predictions"][0]),
            "churn_probability": float(pred["probabilities"][0]),
            "confidence": float(pred["confidence"][0]),
        }


class ModelRegistry:
    """Registro centralizado de modelos treinados.

    Permite salvar, carregar e consultar versões de modelos com metadados
    (métricas, data de treinamento, etc).
    """

    def __init__(self, registry_dir: str = "models/registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def register_model(
        self, model_name: str, version: str, pipeline: TelcoPipeline, metadata: dict
    ) -> str:
        """Registra modelo no registry com metadados.

        Args:
            model_name: Nome do modelo
            version: Versão (ex: "1.0.0")
            pipeline: Pipeline treinado (sklearn Pipeline)
            metadata: Dict com métricas, data, notas, etc

        Returns:
            Caminho do diretório do modelo registrado
        """
        model_dir = self.registry_dir / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Salvar pipeline
        pipeline_path = model_dir / "pipeline.pkl"
        pipeline.save(str(pipeline_path))

        # Salvar metadata
        metadata_path = model_dir / "metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        print(f"✅ Modelo registrado: {model_name}/{version}")
        return str(model_dir)

    def load_model(self, model_name: str, version: str) -> PredictionService:
        """Carrega modelo do registry."""
        model_dir = self.registry_dir / model_name / version

        if not model_dir.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {model_name}/{version}")

        pipeline_path = model_dir / "pipeline.pkl"
        return PredictionService(str(pipeline_path))

    def list_models(self) -> dict:
        """Lista todos os modelos registrados."""
        models = {}
        for model_dir in self.registry_dir.iterdir():
            if model_dir.is_dir():
                versions = [v.name for v in model_dir.iterdir() if v.is_dir()]
                models[model_dir.name] = versions
        return models
