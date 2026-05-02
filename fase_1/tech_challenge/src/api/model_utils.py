import logging
import os

import mlflow
import requests

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self):
        self.pipeline = None
        # Agora configurado para buscar o alias 'champion' por padrão
        self.model_name = os.getenv("MLFLOW_MODEL_NAME", "TelcoChurnPipeline")
        self.alias = os.getenv("MLFLOW_MODEL_ALIAS", "champion")
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    def is_mlflow_ready(self):
        """Verifica se o servidor MLflow está acessível."""
        try:
            response = requests.get(f"{self.tracking_uri}/health", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def load_from_mlflow(self):
        """Carrega o modelo 'champion' utilizando a sintaxe @."""
        if not self.is_mlflow_ready():
            logger.error(f"❌ MLflow inacessível em {self.tracking_uri}.")
            return False

        try:
            mlflow.set_tracking_uri(self.tracking_uri)

            # URI usando o arroba para Aliases
            model_uri = f"models:/{self.model_name}@{self.alias}"
            logger.info(f"📊 Buscando modelo Champion: {model_uri}")

            self.pipeline = mlflow.sklearn.load_model(model_uri)
            logger.info("✅ Modelo Champion carregado com sucesso!")
            return True
        except Exception as e:
            logger.warning(f"⚠ Não foi possível carregar o alias '{self.alias}': {e}")
            return False

    def predict(self, data):
        if self.pipeline is None:
            raise RuntimeError("Model not loaded.")
        return self.pipeline.predict(data)
