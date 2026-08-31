"""Top-level API router.

Health is mounted without a version prefix - it's an infrastructure
concern (used by Docker/orchestrator readiness probes), not part of the
versioned business API, so it shouldn't break if /v1 ever becomes /v2.
Everything else lives under /v1.
"""

from fastapi import APIRouter

from app.api.v1 import v1_router
from app.api.v1.health import router as health_router

api_router = APIRouter()

api_router.include_router(health_router, tags=["health"])
api_router.include_router(v1_router, prefix="/v1")
