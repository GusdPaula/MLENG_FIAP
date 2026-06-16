"""Tests for Pydantic Settings configuration."""

from src.recommender.config.settings import Settings


def test_settings_default_values():
    """Settings should provide sane defaults when no env vars are set."""
    settings = Settings(
        _env_file=None,
        mlflow_tracking_uri="sqlite:///mlflow.db",
        aws_default_region="us-east-1",
        aws_region="us-east-1",
        aws_profile="default",
    )
    assert settings.mlflow_tracking_uri == "sqlite:///mlflow.db"
    assert settings.aws_default_region == "us-east-1"
    assert settings.aws_region == "us-east-1"
    assert settings.aws_profile == "default"


def test_settings_from_env(monkeypatch, tmp_path):
    """Settings should load values from environment variables."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.example.com")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_PROFILE", "production")

    settings = Settings(_env_file=None)
    assert settings.mlflow_tracking_uri == "http://mlflow.example.com"
    assert settings.aws_default_region == "eu-west-1"
    assert settings.aws_region == "eu-west-1"
    assert settings.aws_profile == "production"


def test_settings_from_env_file(tmp_path):
    """Settings should load from a .env file."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MLFLOW_TRACKING_URI=http://localhost:5000\n"
        "AWS_DEFAULT_REGION=ap-southeast-1\n"
        "AWS_REGION=ap-southeast-1\n"
        "AWS_PROFILE=dev\n"
    )

    settings = Settings(_env_file=str(env_file))
    assert settings.mlflow_tracking_uri == "http://localhost:5000"
    assert settings.aws_default_region == "ap-southeast-1"
    assert settings.aws_profile == "dev"


def test_settings_extra_fields_ignored(tmp_path):
    """Settings should ignore extra env vars not defined in the model."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MLFLOW_TRACKING_URI=http://localhost:5000\n"
        "AWS_DEFAULT_REGION=us-east-1\n"
        "AWS_REGION=us-east-1\n"
        "AWS_PROFILE=aws\n"
        "SOME_RANDOM_VAR=should_be_ignored\n"
    )

    settings = Settings(_env_file=str(env_file))
    assert not hasattr(settings, "some_random_var")
