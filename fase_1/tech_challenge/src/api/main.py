"""
Aplicação FastAPI para Telco Churn Prediction - Versão Final Integrada.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Importações internas
from .model_utils import ModelManager
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthCheckResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

# Configuração de Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Instância global do ModelManager
model_manager = ModelManager()


class ScheduleUpdateRequest(BaseModel):
    """Schema para receber a data/hora do agendamento de atualização."""

    target_datetime: datetime


# ============ LÓGICA DE BACKGROUND PARA HOT-SWAP ============


async def wait_and_update_model(target_time: datetime, app: FastAPI):
    """
    Calcula o tempo de espera e atualiza o modelo no ModelManager e no app.state.
    """
    now = datetime.now(timezone.utc)

    if target_time.tzinfo is None:
        target_time = target_time.replace(tzinfo=timezone.utc)

    delay_seconds = (target_time - now).total_seconds()

    if delay_seconds > 0:
        logger.info(
            f"⏰ Atualização agendada para {target_time}. Aguardando {int(delay_seconds)}s."
        )
        await asyncio.sleep(delay_seconds)
    else:
        logger.warning("⚠ A data informada já passou! Atualizando imediatamente.")

    logger.info("🚀 Iniciando download do novo modelo via ModelManager...")

    # O ModelManager já encapsula a lógica de conexão com MLflow
    success = model_manager.load_from_mlflow()

    if success:
        # Atualiza o estado da aplicação atomicamente
        app.state.pipeline = model_manager.pipeline
        logger.info("✅ Hot-Swap concluído! API atualizada com o novo modelo.")
    else:
        logger.error("❌ Falha ao carregar o novo modelo agendado.")


# ============ FACTORY DA APLICAÇÃO ============


def create_app() -> FastAPI:
    """Cria e configura a instância do FastAPI."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Gerencia o startup e shutdown da aplicação."""
        logger.info("=" * 50)
        logger.info("Iniciando Telco Churn Prediction API")

        # Tenta carregar o modelo inicial
        if model_manager.load_from_mlflow():
            app.state.pipeline = model_manager.pipeline
            logger.info("✓ Modelo carregado com sucesso no startup.")
        else:
            app.state.pipeline = None
            logger.warning("⚠ API iniciada sem modelo (Modo Degradado).")

        logger.info("=" * 50)
        yield
        logger.info("Encerrando API...")

    app = FastAPI(
        title="Telco Churn Prediction API",
        description="API para predição de churn com integração MLflow",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        lifespan=lifespan,
    )

    # Middlewares
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_process_time_header(request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

    # ============ ENDPOINTS DE GERENCIAMENTO ============

    @app.post("/api/schedule-update", tags=["Model Management"])
    async def schedule_update(request: ScheduleUpdateRequest, background_tasks: BackgroundTasks):
        """Agenda uma atualização de modelo para o futuro."""
        background_tasks.add_task(wait_and_update_model, request.target_datetime, app)
        return JSONResponse(
            status_code=202,
            content={
                "status": "accepted",
                "message": f"Atualização agendada para {request.target_datetime}",
            },
        )

    @app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
    @app.get("/api/health", response_model=HealthCheckResponse, tags=["Health"])
    async def health_check():
        """Verifica a saúde da API e se o modelo está pronto."""
        is_ready = app.state.pipeline is not None
        return HealthCheckResponse(
            status="healthy" if is_ready else "degraded",
            version="0.1.0",
            model_loaded=is_ready,
        )

    # ============ ENDPOINTS DE PREDIÇÃO ============

    @app.post("/api/predict", response_model=PredictionResponse, tags=["Predictions"])
    async def predict(request: PredictionRequest):
        """Predição individual para um cliente."""
        if app.state.pipeline is None:
            # Tentativa de auto-recuperação caso o modelo não esteja carregado
            if model_manager.load_from_mlflow():
                app.state.pipeline = model_manager.pipeline
            else:
                raise HTTPException(status_code=503, detail="Modelo não disponível no momento.")

        start_time = time.time()
        try:
            # Conversão de Pydantic para DataFrame
            input_df = pd.DataFrame([request.features.model_dump()])

            # Predição usando o pipeline (preprocessamento incluso)
            prediction = model_manager.predict(input_df)[0]

            # Cálculo de probabilidades se disponível
            prob = 0.0
            conf = 0.0
            if hasattr(app.state.pipeline, "predict_proba"):
                y_proba = app.state.pipeline.predict_proba(input_df)
                prob = float(y_proba[0, 1])
                conf = float(y_proba.max())

            return PredictionResponse(
                prediction=int(prediction),
                probability=prob if request.return_probability else None,
                confidence=conf if request.return_probability else None,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            raise HTTPException(status_code=500, detail=f"Erro interno: {e!s}") from e

    @app.post("/api/predict-batch", response_model=BatchPredictionResponse, tags=["Predictions"])
    async def predict_batch(request: BatchPredictionRequest):
        """Predição em lote (batch)."""
        if app.state.pipeline is None:
            raise HTTPException(status_code=503, detail="Modelo não disponível.")

        start_time = time.time()
        try:
            samples = [s.model_dump() for s in request.samples]
            X = pd.DataFrame(samples)

            predictions = model_manager.predict(X).tolist()

            probabilities = None
            if request.return_probabilities and hasattr(app.state.pipeline, "predict_proba"):
                probabilities = app.state.pipeline.predict_proba(X)[:, 1].tolist()

            return BatchPredictionResponse(
                predictions=predictions,
                probabilities=probabilities,
                batch_size=len(request.samples),
                processing_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            logger.error(f"Erro no processamento batch: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # ============ ENDPOINT DE INFO ============

    @app.get("/api/model-info", response_model=ModelInfoResponse, tags=["Model"])
    async def get_model_info():
        """Retorna detalhes sobre as features e versão do modelo."""
        if app.state.pipeline is None:
            raise HTTPException(status_code=503, detail="Modelo não carregado.")

        # Tenta extrair nomes de features do preprocessor se existir
        features = []
        if hasattr(app.state.pipeline, "named_steps"):
            prep = app.state.pipeline.named_steps.get("preprocessor")
            if prep and hasattr(prep, "get_feature_names_out"):
                features = prep.get_feature_names_out().tolist()

        return ModelInfoResponse(
            model_type=str(type(app.state.pipeline.named_steps.get("model"))),
            model_version="MLflow Production",
            n_features=len(features),
            features_used=features,
        )

    # ============ ERROR HANDLERS ============

    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "detail": str(exc),
                "timestamp": datetime.now().isoformat(),
            },
        )

    return app


# Instanciação da aplicação para o Uvicorn
app = create_app()
