"""Aplicação FastAPI para Telco Churn Prediction - REFATORADA.

API com suporte a MLflow Model Registry:
- Carrega Pipeline completo do MLflow (preprocessamento + modelo)
- Recebe dados brutos em formato JSON
- Configure MLFLOW_TRACKING_URI para ativar integração MLflow
"""

import logging
import os
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthCheckResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowPipelineLoader:
    """Carregador simples de Pipeline do MLflow com fallback para arquivo local."""

    def __init__(self):
        self.pipeline = None
        self.categorical_columns = None
        self.numerical_columns = None

    def load_from_mlflow(self, model_name: str = "TelcoChurnPipeline", stage: str = "Production"):
        """
        Carrega Pipeline do MLflow Model Registry.

        Args:
            model_name: Nome do modelo registrado no MLflow
            stage: Stage do modelo (Production, Staging, etc)

        Returns:
            True se carregou com sucesso, False caso contrário
        """
        try:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
            logger.info(f"📊 Configurando MLflow Tracking URI: {tracking_uri}")
            mlflow.set_tracking_uri(tracking_uri)

            model_uri = f"models:/{model_name}@{stage}"
            logger.info(f"📊 Tentando carregar pipeline: {model_uri}")

            self.pipeline = mlflow.sklearn.load_model(model_uri)
            logger.info(f"✓ Pipeline carregado via MLflow: {model_uri}")          
            
            return True

        except Exception as e:
            logger.warning(f"⚠ Não foi possível carregar do MLflow: {e}")
            return False

    def is_loaded(self) -> bool:
        logger.info("Verificando se pipeline foi carregado...")
        """Verifica se o pipeline foi carregado com sucesso."""
        return self.pipeline is not None

    def get_feature_names(self) -> list[str]:
        """Retorna nomes das features esperadas pelo pipeline."""
        logger.info("Obtendo nomes das features do pipeline...")
        # Extrai nomes das features do preprocessor
        if hasattr(self.pipeline, "named_steps"):
            preprocessor = self.pipeline.named_steps.get("preprocessor")
            if preprocessor and hasattr(preprocessor, "get_feature_names_out"):
                try:
                    return preprocessor.get_feature_names_out().tolist()
                except Exception:
                    pass
        return []


def create_app(model_path: str | None = None) -> FastAPI:
    """
    Factory function para criar a aplicação FastAPI.

    Args:
        model_path: Caminho para o modelo treinado (fallback local)

    Returns:
        Aplicação FastAPI configurada com suporte a MLflow
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifecycle manager: startup e shutdown.

        Startup:
        - Carrega Pipeline completo do MLflow (preprocessador + modelo)

        Shutdown:
        - Limpeza de recursos
        """
        # ============ STARTUP ============
        logger.info("=" * 70)
        logger.info("Iniciando Telco Churn Prediction API")
        logger.info("=" * 70)

        # Carregar Pipeline completo (preprocessador + modelo integrados)
        try:
            loader = MLflowPipelineLoader()

            # Tentar carregar do MLflow primeiro
            mlflow_loaded = loader.load_from_mlflow(
                model_name="TelcoChurnPipeline", stage="Production"
            )

            if loader.is_loaded():
                app.state.pipeline = loader.pipeline
                app.state.feature_names = loader.get_feature_names()
                logger.info(f"✓ Pipeline carregado com sucesso do MLflow! Features: {app.state.feature_names}")
                logger.info("✓ Pipeline carregado com sucesso")
                preprocessor = app.state.pipeline.named_steps['preprocessor']
                for name, transformer, columns in preprocessor.transformers_:
                    logger.info(f"🔍 O transformer '{name}' está esperando estas colunas: {columns}")
            else:
                raise RuntimeError(
                    "Não foi possível carregar o Pipeline. "
                    "Configure MLFLOW_TRACKING_URI ou forneça model_path."
                )

        except Exception as e:
            logger.error(f"✗ Erro ao carregar Pipeline: {e}")
            logger.error("  API iniciada em modo DEGRADADO (sem predições)")
            app.state.pipeline = None
            app.state.feature_names = []

        logger.info("API iniciada com sucesso!")
        logger.info("=" * 70)

        yield

        # ============ SHUTDOWN ============
        logger.info("Encerrando API...")

    app = FastAPI(
        title="Telco Churn Prediction API",
        description="API para predição de churn de clientes de telecom",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        lifespan=lifespan,
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
    app.state.pipeline = None  # Pipeline sklearn (preprocessador + modelo)
    app.state.feature_names = []  # Nomes das features esperadas

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
            status="healthy" if app.state.pipeline else "degraded",
            version="0.1.0",
            model_loaded=app.state.pipeline is not None,
        )

    # ============ PREDIÇÕES ============
    @app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
    @app.post("/api/predict", response_model=PredictionResponse, tags=["Predictions"])
    async def predict(request: PredictionRequest) -> PredictionResponse:
        """
        Fazer predição de churn para um cliente.

        O Pipeline sklearn já contém o preprocessamento,
        então apenas convertemos o JSON para DataFrame e passamos ao pipeline.

        Args:
            request: Features nomeadas do cliente (em formato texto/bruto)

        Returns:
            Predição com probabilidade e confiança
        """
        if app.state.pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline não foi carregado")

        start_time = time.time()
        logger.info("Recebendo requisição de predição...")

        try:
            # 1. Converter objeto Pydantic para dict
            features_dict = request.features.model_dump() if hasattr(request.features, "model_dump") else request.features.dict()
            
            logger.info(f"Features recebidas (raw): {features_dict}")

            # 2. Converter dict para DataFrame de 1 linha
            # O pipeline sklearn já espera as features na ordem correta
            X = pd.DataFrame([features_dict])

            logger.info(X.head())

            logger.info(f"Features convertidas para DataFrame: {X.shape} - Colunas: {X.columns.tolist()}")

            # 3. Passar para o pipeline (que já faz o preprocessamento)
            y_pred = app.state.pipeline.predict(X)
            logger.info(f"Predição bruta do pipeline: {y_pred}")

            if request.return_probability:
                y_proba = app.state.pipeline.predict_proba(X)
                pred_result = {
                    "prediction": int(y_pred[0]),
                    "probability": float(y_proba[0, 1]),  # Probabilidade da classe 1 (churn)
                    "confidence": float(y_proba.max()),
                }
                logger.info(
                    f"Predição: {pred_result['prediction']}, Probabilidade: {pred_result['probability']:.4f}, Confiança: {pred_result['confidence']:.4f}"
                )
            else:
                pred_result = {
                    "prediction": int(y_pred[0]),
                    "probability": None,
                    "confidence": None,
                }

            processing_time = (time.time() - start_time) * 1000

            return PredictionResponse(**pred_result, processing_time_ms=processing_time)

        except ValueError as e:
            logger.error(f"Erro na validação de features: {e}")
            raise HTTPException(status_code=400, detail=f"Erro ao validar features: {e!s}") from e

        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Erro ao processar predição: {e!s}") from e

    # ============ PREDIÇÕES EM LOTE ============
    @app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Predictions"])
    @app.post("/api/predict-batch", response_model=BatchPredictionResponse, tags=["Predictions"])
    async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
        """
        Fazer predições em lote.

        O Pipeline sklearn já contém o preprocessamento integrado.

        Args:
            request: Lista de dicionários com features nomeadas

        Returns:
            Predições em lote com probabilidades opcionais
        """
        if app.state.pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline não foi carregado")

        start_time = time.time()

        try:
            # 1. Converter lista de objetos Pydantic para lista de dicts
            samples_list = [
                s.model_dump() if hasattr(s, "model_dump") else s.dict()
                for s in request.samples
            ]

            # 2. Converter para DataFrame
            X = pd.DataFrame(samples_list)

            logger.info(f"Batch recebido: {X.shape} - Colunas: {X.columns.tolist()}")

            # 3. Passar para o pipeline (que já faz o preprocessamento)
            y_pred = app.state.pipeline.predict(X)

            if request.return_probabilities:
                y_proba = app.state.pipeline.predict_proba(X)
                probas = y_proba[:, 1].tolist()  # Probabilidade da classe 1 (churn)
            else:
                probas = None

            processing_time = (time.time() - start_time) * 1000

            return BatchPredictionResponse(
                predictions=y_pred.tolist(),
                probabilities=probas,
                batch_size=len(request.samples),
                processing_time_ms=processing_time,
            )

        except ValueError as e:
            logger.error(f"Erro na validação de features: {e}")
            raise HTTPException(status_code=400, detail=f"Erro ao validar features: {e!s}") from e

        except Exception as e:
            logger.error(f"Erro na predição em lote: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Erro ao processar lote: {e!s}") from e

    # ============ MODELO INFO ============
    @app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
    @app.get("/api/model-info", response_model=ModelInfoResponse, tags=["Model"])
    async def get_model_info() -> ModelInfoResponse:
        """
        Informações do Pipeline carregado.

        Returns:
            Informações do Pipeline
        """
        if app.state.pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline não foi carregado")

        # Tentar extrair nomes das features do pipeline
        feature_names = app.state.feature_names
        if not feature_names and hasattr(app.state.pipeline, "named_steps"):
            preprocessor = app.state.pipeline.named_steps.get("preprocessor")
            if preprocessor and hasattr(preprocessor, "get_feature_names_out"):
                try:
                    feature_names = preprocessor.get_feature_names_out().tolist()
                except Exception:
                    feature_names = []

        return ModelInfoResponse(
            model_type="LogisticRegression (com ColumnTransformer)",
            model_version="1.0.0",
            n_features=len(feature_names) if feature_names else 0,
            features_used=feature_names,
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
                error=str(exc), status_code=400, timestamp=datetime.now().isoformat()
            ).dict(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        logger.error(f"Erro não tratado: {exc}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Erro interno do servidor",
                status_code=500,
                timestamp=datetime.now().isoformat(),
            ).dict(),
        )

    return app


# Aplicação padrão
# Tentar carregar modelo do arquivo local, caso não exista, criar app degraded
_default_model_path = "models/best_model_with_metadata.pkl"
if os.path.exists(_default_model_path):
    app = create_app(_default_model_path)
else:
    # Em desenvolvimento/testes, criar app sem modelo (modo degraded)
    app = create_app(model_path=None)
