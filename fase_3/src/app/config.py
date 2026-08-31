"""Application configuration.

All environment-dependent values live here, loaded from environment
variables (or a local .env file) via Pydantic Settings. Nothing else in
the codebase should call os.environ directly - this is the single source
of truth for configuration, and it gives us validation + typing for free.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="APP_",
        extra="ignore",
    )

    # App metadata
    app_name: str = "Triage API"
    app_version: str = "0.1.0"
    environment: str = "local"  # local | staging | production

    # Model artifact (produced by ml/convert_to_onnx.py, consumed at startup)
    model_path: str = "treino_modelo/artifacts/model.onnx"
    model_version: str = "tfidf-rf-v1.0-onnx"
    class_labels: list[str] = [
        "Neoplasms",
        "Digestive system diseases",
        "Nervous system diseases",
        "Cardiovascular diseases",
        "General pathological conditions",
    ]

    # Request limits — enforced in schemas/classify.py
    max_batch_size: int = 64
    max_text_length: int = 5000

    # Logging
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Returns a cached Settings instance.

    lru_cache means env vars are parsed once per process, not on every
    call — settings are read via Depends(get_settings) in routes, so this
    matters once request volume goes up.
    """
    return Settings()
