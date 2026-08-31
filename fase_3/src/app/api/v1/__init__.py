"""v1 API surface — combines classify and model_info into one router
mounted at /v1 by api/router.py.
"""

from fastapi import APIRouter

from app.api.v1 import classify, model_info

v1_router = APIRouter()

v1_router.include_router(classify.router, tags=["classify"])
v1_router.include_router(model_info.router, tags=["model"])
