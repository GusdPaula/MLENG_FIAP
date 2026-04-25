"""Aplicação FastAPI para Telco Churn Prediction."""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging
from typing import Optional
import numpy as np
from datetime import datetime

from .schemas import (
    HealthCheckResponse,
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    ModelInfoResponse,
)
from src.data.preprocessing import TelcoDataPreprocessor
from src.models import PredictionService
from src.config import get_config, APIConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app(model_path: Optional[str] = None) -> FastAPI:
    """
    Factory function para criar a aplicação FastAPI.

    Args:
        model_path: Caminho para o modelo treinado

    Returns:
        Aplicação FastAPI configurada
    """
    app = FastAPI(
        title="Telco Churn Prediction API",
        description="API para predição de churn de clientes de telecom",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Middleware de latência
    @app.middleware("http")
    async def add_latency_header(request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

    # State da aplicação
    app.state.model_service: Optional[PredictionService] = None
    app.state.preprocessor: Optional[TelcoDataPreprocessor] = None
    app.state.config = get_config()

    # Inicializar preprocessor para inferência
    try:
        preprocessor = TelcoDataPreprocessor()
        preprocessor.fit_for_inference('data/processed/telco_churn_processed.csv')
        app.state.preprocessor = preprocessor
        logger.info(f"[OK] Preprocessador inicializado para inferência")
    except Exception as e:
        logger.error(f"[ERROR] Erro ao inicializar preprocessador: {e}")

    # Carregar modelo se fornecido
    if model_path:
        try:
            app.state.model_service = PredictionService(model_path)
            logger.info(f"[OK] Modelo carregado: {model_path}")
        except Exception as e:
            logger.error(f"[ERROR] Erro ao carregar modelo: {e}")

    # ============ HEALTH CHECK ============
    @app.get("/health", tags=["Health"])
    @app.get("/api/health", tags=["Health"])
    async def health_check() -> HealthCheckResponse:
        """
        Health check da API.

        Returns:
            Status da aplicação
        """
        return HealthCheckResponse(
            status="healthy" if app.state.model_service else "degraded",
            version="0.1.0",
            model_loaded=app.state.model_service is not None,
        )

    # ============ PREDIÇÕES ============
    @app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
    @app.post("/api/predict", response_model=PredictionResponse, tags=["Predictions"])
    async def predict(request: PredictionRequest) -> PredictionResponse:
        """
        Fazer predição de churn para um cliente.

        Args:
            request: Features nomeadas do cliente

        Returns:
            Predição com probabilidade e confiança
        """
        if app.state.model_service is None:
            raise HTTPException(
                status_code=503,
                detail="Modelo não foi carregado"
            )

        start_time = time.time()

        try:
            # Transformar dicionário de features para array ordenado
            X = app.state.preprocessor.transform_single(request.features)

            logger.info(f"Features transformadas shape: {X.shape}")

            # Realizar predição com/sem probabilidade
            if request.return_probability:
                result = app.state.model_service.predict(X, return_proba=True)
                pred_result = {
                    'prediction': int(result['predictions'][0]),
                    'probability': float(result['probabilities'][0]),
                    'confidence': float(result['confidence'][0]),
                }
                logger.info(f"Predicton: {pred_result['prediction']}, Probability: {pred_result['probability']:.4f}")
            else:
                pred = app.state.model_service.predict(X)
                pred_result = {
                    'prediction': int(pred[0]),
                    'probability': None,
                    'confidence': None,
                }

            processing_time = (time.time() - start_time) * 1000

            return PredictionResponse(
                **pred_result,
                processing_time_ms=processing_time
            )

        except ValueError as e:
            logger.error(f"Erro na validação de features: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Erro ao validar features: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Erro ao processar predição: {str(e)}"
            )

    # ============ PREDIÇÕES EM LOTE ============
    @app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Predictions"])
    @app.post("/api/predict-batch", response_model=BatchPredictionResponse, tags=["Predictions"])
    async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
        """
        Fazer predições em lote.

        Args:
            request: Lista de dicionários com features nomeadas

        Returns:
            Predições em lote
        """
        if app.state.model_service is None:
            raise HTTPException(
                status_code=503,
                detail="Modelo não foi carregado"
            )

        start_time = time.time()

        try:
            # Transformar lista de dicionários para array ordenado
            X = app.state.preprocessor.transform_batch(request.samples)

            if request.return_probabilities:
                result = app.state.model_service.predict(X, return_proba=True)
                preds = result['predictions']
                probas = result['probabilities'].tolist()
            else:
                preds = app.state.model_service.predict(X)
                probas = None

            processing_time = (time.time() - start_time) * 1000

            return BatchPredictionResponse(
                predictions=preds.tolist(),
                probabilities=probas,
                batch_size=len(request.samples),
                processing_time_ms=processing_time
            )

        except ValueError as e:
            logger.error(f"Erro na validação de features: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Erro ao validar features: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Erro na predição em lote: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Erro ao processar lote: {str(e)}"
            )

    # ============ MODELO INFO ============
    @app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
    @app.get("/api/model-info", response_model=ModelInfoResponse, tags=["Model"])
    async def get_model_info() -> ModelInfoResponse:
        """
        Informações do modelo carregado.

        Returns:
            Informações do modelo
        """
        if app.state.model_service is None:
            raise HTTPException(
                status_code=503,
                detail="Modelo não foi carregado"
            )

        # Retornar informações do modelo com 30 features
        return ModelInfoResponse(
            model_type="Logistic Regression / Random Forest / XGBoost",
            model_version="0.1.0",
            n_features=30,
            features_used=app.state.preprocessor.feature_names
        )

    # ============ ROOT ============
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint com informações da API."""
        return {
            "message": "Telco Churn Prediction API",
            "version": "0.1.0",
            "docs": "/api/docs",
            "health": "/health",
        }

    # ============ ERROR HANDLERS ============
    @app.exception_handler(ValueError)
    async def value_error_handler(request, exc):
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=str(exc),
                status_code=400,
                timestamp=datetime.now().isoformat()
            ).dict()
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        logger.error(f"Erro não tratado: {exc}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Erro interno do servidor",
                status_code=500,
                timestamp=datetime.now().isoformat()
            ).dict()
        )

    return app


# Aplicação padrão
app = create_app(".\\models\\best_model_with_metadata.pkl")
