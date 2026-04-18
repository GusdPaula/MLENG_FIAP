"""Classe de inferência para uso em produção."""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Union
from .pipeline import TelcoPipeline
from .transformers import *


class PredictionService:
    """Serviço de predição para produção."""
    
    def __init__(self, 
                 pipeline_path: str,
                 scaler_path: Optional[str] = None,
                 preprocessor_path: Optional[str] = None):
        """
        Inicializa serviço de predição.
        
        Args:
            pipeline_path: Caminho para o arquivo do pipeline treinado
            scaler_path: Caminho para o scaler (opcional)
            preprocessor_path: Caminho para o preprocessor (opcional)
        """
        self.pipeline = None
        self.scaler = None
        self.preprocessor = None
        
        # Carregar pipeline
        if Path(pipeline_path).exists():
            with open(pipeline_path, 'rb') as f:
                self.pipeline = pickle.load(f)
            print(f"✅ Pipeline carregado de {pipeline_path}")
        else:
            raise FileNotFoundError(f"Pipeline não encontrado em {pipeline_path}")
        
        # Carregar scaler se fornecido
        if scaler_path and Path(scaler_path).exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✅ Scaler carregado de {scaler_path}")
        
        # Carregar preprocessor se fornecido
        if preprocessor_path and Path(preprocessor_path).exists():
            with open(preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            print(f"✅ Preprocessor carregado de {preprocessor_path}")
    
    def preprocess(self, X: np.ndarray) -> np.ndarray:
        """Aplica preprocessamento (se disponível)."""
        if self.preprocessor is not None:
            return self.preprocessor.transform(X)
        return X
    
    def predict(self, X: np.ndarray, return_proba: bool = False) -> Union[np.ndarray, Dict]:
        """
        Faz predição.
        
        Args:
            X: Features de entrada
            return_proba: Se True, retorna probabilidades também
        
        Returns:
            Predições ou dict com predições e probabilidades
        """
        # Preprocessar se necessário
        X_processed = self.preprocess(X)
        
        # Predições
        y_pred = self.pipeline.predict(X_processed)
        
        if return_proba:
            y_proba = self.pipeline.predict_proba(X_processed)
            return {
                'predictions': y_pred,
                'probabilities': y_proba[:, 1],  # Probabilidade da classe 1 (churn)
                'confidence': np.max(y_proba, axis=1)
            }
        
        return y_pred
    
    def predict_batch(self, X: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        """Predição em lotes (útil para datasets grandes)."""
        predictions = []
        
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            batch_pred = self.predict(batch)
            predictions.append(batch_pred)
        
        return np.concatenate(predictions)
    
    def predict_single(self, features: Dict) -> Dict:
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
            'churn_prediction': int(pred['predictions'][0]),
            'churn_probability': float(pred['probabilities'][0]),
            'confidence': float(pred['confidence'][0])
        }


class ModelRegistry:
    """Registro de modelos treinados."""
    
    def __init__(self, registry_dir: str = "models/registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
    
    def register_model(self, 
                      model_name: str,
                      version: str,
                      pipeline: TelcoPipeline,
                      metadata: Dict) -> str:
        """
        Registra modelo no registry.
        
        Args:
            model_name: Nome do modelo
            version: Versão (ex: "1.0.0")
            pipeline: Pipeline treinado
            metadata: Dict com informações do modelo (métricas, data, etc)
        
        Returns:
            Caminho do modelo registrado
        """
        model_dir = self.registry_dir / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Salvar pipeline
        pipeline_path = model_dir / "pipeline.pkl"
        pipeline.save(str(pipeline_path))
        
        # Salvar metadata
        metadata_path = model_dir / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
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
    
    def list_models(self) -> Dict:
        """Lista todos os modelos registrados."""
        models = {}
        for model_dir in self.registry_dir.iterdir():
            if model_dir.is_dir():
                versions = [v.name for v in model_dir.iterdir() if v.is_dir()]
                models[model_dir.name] = versions
        return models
