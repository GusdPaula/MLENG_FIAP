"""Pydantic Settings for environment configuration.

Loads settings from environment variables and .env files with validation
and type coercion. Replaces raw os.environ.get() calls and python-dotenv
load_dotenv() scattered across pipeline modules.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_env_file() -> Path | None:
    """Walk up from CWD to locate a .env file."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
    return None


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # MLflow
    mlflow_tracking_uri: str = Field(
        default="sqlite:///mlflow.db",
        description="MLflow tracking server URI.",
    )

    # AWS
    aws_default_region: str = Field(
        default="us-east-1",
        description="Default AWS region for SDK calls.",
    )
    aws_region: str = Field(
        default="us-east-1",
        description="AWS region (used by some services).",
    )
    aws_profile: str = Field(
        default="default",
        description="Named AWS CLI profile to use.",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    env_file = _find_env_file()
    if env_file:
        return Settings(_env_file=env_file)
    return Settings()
