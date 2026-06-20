"""Main entry point for the FastAPI application.

This module initializes and configures the FastAPI application with all routes.
"""

from .controllers.routes import app

__all__ = ["app"]
