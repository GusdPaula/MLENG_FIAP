#!/usr/bin/env python3
"""Script to promote MLflow Model Registry versions using aliases (staging -> production)."""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    # Load .env file from the current directory or parent directory
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Promote MLflow model from staging alias to production alias."
    )
    parser.add_argument(
        "--model-name",
        default="ecommerce_recommender",
        help="Name of the registered model in MLflow registry (default: 'ecommerce_recommender')",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Specific version to promote. If not specified, the version with 'staging' alias will be promoted.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the promotion without making changes to the MLflow registry.",
    )
    args = parser.parse_args()

    # Verify tracking URI is set
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        logger.error(
            "MLFLOW_TRACKING_URI is not set. Please set it as an environment variable or in a .env file."
        )
        sys.exit(1)

    logger.info("Connecting to MLflow tracking server at: %s", tracking_uri)

    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        logger.error("mlflow package is not installed. Please run: pip install mlflow")
        sys.exit(1)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # 1. Resolve version
    version_to_promote = args.version
    if not version_to_promote:
        logger.info(
            "No version specified. Searching for the model version with 'staging' alias..."
        )
        try:
            staging_version_obj = client.get_model_version_by_alias(
                args.model_name, "staging"
            )
            version_to_promote = staging_version_obj.version
            logger.info(
                "Found model version %s with 'staging' alias.", version_to_promote
            )
        except Exception as e:
            logger.error(
                "Failed to find any model version with 'staging' alias for model '%s': %s",
                args.model_name,
                e,
            )
            sys.exit(1)

    # 2. Get info of the version we are promoting
    try:
        mv_info = client.get_model_version(args.model_name, version_to_promote)
        logger.info(
            "Target model version %s has current aliases: %s (Run ID: %s)",
            version_to_promote,
            mv_info.aliases,
            mv_info.run_id,
        )
    except Exception as e:
        logger.error(
            "Failed to retrieve details for model version %s of '%s': %s",
            version_to_promote,
            args.model_name,
            e,
        )
        sys.exit(1)

    # Check if already in production
    if "production" in (mv_info.aliases or []):
        logger.info(
            "Model version %s already has the 'production' alias. Nothing to do.",
            version_to_promote,
        )
        sys.exit(0)

    # 3. Perform Promotion
    if args.dry_run:
        logger.info(
            "[DRY-RUN] Model version %s of '%s' would be assigned the 'production' alias.",
            version_to_promote,
            args.model_name,
        )
    else:
        logger.info(
            "Assigning 'production' alias to model version %s of '%s'...",
            version_to_promote,
            args.model_name,
        )
        try:
            client.set_registered_model_alias(
                name=args.model_name, alias="production", version=version_to_promote
            )
            logger.info(
                "Successfully assigned 'production' alias to model version %s of '%s'.",
                version_to_promote,
                args.model_name,
            )
        except Exception as e:
            logger.error(
                "Failed to assign 'production' alias to model version %s: %s",
                version_to_promote,
                e,
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
