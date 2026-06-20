"""FastAPI application for prediction API.

This module provides a REST API wrapper around the PredictionService,
exposing prediction endpoints with proper error handling and validation.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from ..exceptions import (
    InvalidInputError,
    ModelLoadError,
    PredictionError,
    PredictorNotFoundError,
)
from ..models.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
    RecommendationResponse,
)
from ..services.prediction_service import PredictionService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# API Key configuration
API_KEY = os.getenv("API_KEY", "default-api-key-change-in-production")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Model configuration
MODEL_PATH = os.getenv("MODEL_PATH", "ecommerce_recommender/models/mlflow_experiments/gmf_binary.pt")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME")
MLFLOW_MODEL_VERSION = os.getenv("MLFLOW_MODEL_VERSION")
MLFLOW_MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")


async def verify_api_key(api_key: str = Depends(api_key_header)) -> str:
    """Verify the API key.

    Args:
        api_key: The API key from the header.

    Returns:
        The verified API key.

    Raises:
        HTTPException: If the API key is missing or invalid.
    """
    if api_key is None:
        logger.warning("API key missing from request")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key missing",
        )

    if api_key != API_KEY:
        logger.warning("Invalid API key provided")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return api_key

# Global prediction service instance
prediction_service: PredictionService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for the FastAPI application.

    Args:
        app: The FastAPI application instance.

    Yields:
        None
    """
    # Startup
    global prediction_service
    logger.info("Starting prediction API service")
    logger.info(f"API key authentication enabled (key: {API_KEY[:8]}...)" if len(API_KEY) > 8 else "API key authentication enabled (using default key - CHANGE IN PRODUCTION)")

    # Initialize prediction service
    model_path = Path(MODEL_PATH)
    logger.info(f"Loading model from {model_path}")
    if not model_path.exists():
        logger.warning(f"Model file not found at {model_path}, service will be initialized but predictions will fail")

    try:
        prediction_service = PredictionService(
            model_path=model_path,
            predictor_type="top_k",
            device="cpu",
            enable_monitoring=True,
            mlflow_tracking_uri=MLFLOW_TRACKING_URI,
            mlflow_model_name=MLFLOW_MODEL_NAME,
            mlflow_model_version=MLFLOW_MODEL_VERSION,
            mlflow_model_alias=MLFLOW_MODEL_ALIAS,
        )
        logger.info("Prediction service initialized successfully")
    except ModelLoadError as e:
        logger.error(f"Failed to initialize prediction service: {e}")
        prediction_service = None

    yield

    # Shutdown
    logger.info("Shutting down prediction API service")
    prediction_service = None


# Create FastAPI application
app = FastAPI(
    title="E-commerce Recommendation API",
    description="API for generating recommendations using trained recommender models",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check(api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    """Health check endpoint.

    Args:
        api_key: Verified API key.

    Returns:
        Dictionary with service health status.
    """
    return {
        "status": "healthy" if prediction_service is not None else "unhealthy",
        "service": "prediction_api",
    }


@app.get("/model/info")
async def get_model_info(api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    """Get information about the loaded model.

    Args:
        api_key: Verified API key.

    Returns:
        Dictionary with model metadata.

    Raises:
        HTTPException: If prediction service is not initialized.
    """
    if prediction_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service not initialized",
        )

    try:
        info = prediction_service.get_model_info()
        return info
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}",
        ) from e


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, api_key: str = Depends(verify_api_key)) -> PredictionResponse:
    """Generate predictions for a single user.

    Args:
        request: The prediction request.
        api_key: Verified API key.

    Returns:
        Prediction response with item scores.

    Raises:
        HTTPException: If prediction fails.
    """
    try:
        if prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service not initialized",
            )
        response = prediction_service.predict(request)
        return response
    except InvalidInputError as e:
        logger.warning(f"Invalid input: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        ) from e


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(requests: BatchPredictionRequest) -> BatchPredictionResponse:
    """Generate predictions for multiple users.

    Args:
        requests: Batch prediction request.

    Returns:
        Batch prediction response.

    Raises:
        HTTPException: If prediction fails.
    """
    if prediction_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service not initialized",
        )

    try:
        # Convert to list of PredictionRequest
        prediction_requests = [
            PredictionRequest(
                user_id=user_id,
                item_ids=item_ids,
                k=requests.k,
            )
            for user_id, item_ids in requests.user_item_pairs
        ]

        response = prediction_service.predict_batch(prediction_requests)
        return response
    except InvalidInputError as e:
        logger.warning(f"Invalid input: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        ) from e


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
async def recommend(user_id: int, k: int = 10) -> RecommendationResponse:
    """Generate top-k recommendations for a user.

    Args:
        user_id: The user ID to generate recommendations for.
        k: Number of recommendations to return.

    Returns:
        Recommendation response with top-k items and scores.

    Raises:
        HTTPException: If recommendation fails.
    """
    if prediction_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service not initialized",
        )

    try:
        response = prediction_service.recommend(user_id=user_id, k=k)
        return response
    except InvalidInputError as e:
        logger.warning(f"Invalid input: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recommendation failed: {str(e)}",
        ) from e


@app.post("/monitoring/baselines")
async def set_monitoring_baselines(api_key: str = Depends(verify_api_key)) -> dict[str, str]:
    """Set monitoring baselines based on current prediction history.

    Args:
        api_key: Verified API key.

    Returns:
        Dictionary with status message.

    Raises:
        HTTPException: If operation fails.
    """
    if prediction_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service not initialized",
        )

    try:
        prediction_service.set_monitoring_baselines()
        return {"status": "baselines set successfully"}
    except RuntimeError as e:
        logger.warning(f"Failed to set baselines: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Failed to set baselines: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set baselines: {str(e)}",
        ) from e


@app.get("/monitoring/check")
async def check_shifts() -> dict[str, Any]:
    """Check for model and data shifts.

    Returns:
        Dictionary with shift detection results.

    Raises:
        HTTPException: If operation fails.
    """
    if prediction_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service not initialized",
        )

    try:
        results = prediction_service.check_shifts()
        return results
    except RuntimeError as e:
        logger.warning(f"Failed to check shifts: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Failed to check shifts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check shifts: {str(e)}",
        ) from e


@app.get("/monitoring/summary")
async def get_monitoring_summary(api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    """Get monitoring status summary.

    Args:
        api_key: Verified API key.

    Returns:
        Dictionary with monitoring statistics.

    Raises:
        HTTPException: If operation fails.
    """
    if prediction_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service not initialized",
        )

    try:
        summary = prediction_service.get_monitoring_summary()
        return summary
    except RuntimeError as e:
        logger.warning(f"Failed to get monitoring summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Failed to get monitoring summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get monitoring summary: {str(e)}",
        ) from e


@app.exception_handler(ModelLoadError)
async def model_load_error_handler(request, exc: ModelLoadError) -> JSONResponse:
    """Handle ModelLoadError exceptions.

    Args:
        request: The request object.
        exc: The ModelLoadError exception.

    Returns:
        JSONResponse with error details.
    """
    logger.error(f"Model load error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)},
    )


@app.exception_handler(InvalidInputError)
async def invalid_input_error_handler(request, exc: InvalidInputError) -> JSONResponse:
    """Handle InvalidInputError exceptions.

    Args:
        request: The request object.
        exc: The InvalidInputError exception.

    Returns:
        JSONResponse with error details.
    """
    logger.warning(f"Invalid input error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


@app.exception_handler(PredictorNotFoundError)
async def predictor_not_found_error_handler(request, exc: PredictorNotFoundError) -> JSONResponse:
    """Handle PredictorNotFoundError exceptions.

    Args:
        request: The request object.
        exc: The PredictorNotFoundError exception.

    Returns:
        JSONResponse with error details.
    """
    logger.error(f"Predictor not found error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)},
    )


@app.exception_handler(PredictionError)
async def prediction_error_handler(request, exc: PredictionError) -> JSONResponse:
    """Handle generic PredictionError exceptions.

    Args:
        request: The request object.
        exc: The PredictionError exception.

    Returns:
        JSONResponse with error details.
    """
    logger.error(f"Prediction error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
