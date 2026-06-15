#!/usr/bin/env python3
"""Script to promote MLflow Model Registry versions between stages (Staging -> Production)."""

import argparse
import logging
import os
import sys
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    # Load .env file from the current directory or parent directory
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Promote MLflow model from Staging to Production.")
    parser.add_argument(
        "--model-name",
        default="ecommerce_recommender",
        help="Name of the registered model in MLflow registry (default: 'ecommerce_recommender')",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Specific version to promote. If not specified, the latest model in Staging will be promoted.",
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
        logger.error("MLFLOW_TRACKING_URI is not set. Please set it as an environment variable or in a .env file.")
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
        logger.info("No version specified. Searching for the latest model version in 'Staging' stage...")
        try:
            staging_versions = client.get_latest_versions(args.model_name, stages=["Staging"])
            if not staging_versions:
                logger.error("No model versions found in 'Staging' stage for model '%s'. Cannot proceed.", args.model_name)
                sys.exit(1)
            latest_staging = staging_versions[0]
            version_to_promote = latest_staging.version
            logger.info("Found model version %s in 'Staging' stage.", version_to_promote)
        except Exception as e:
            logger.error("Failed to fetch latest versions for model '%s': %s", args.model_name, e)
            sys.exit(1)
            
    # 2. Get info of the version we are promoting
    try:
        mv_info = client.get_model_version(args.model_name, version_to_promote)
        logger.info("Target model version %s is currently in stage: %s (Run ID: %s)", 
                    version_to_promote, mv_info.current_stage, mv_info.run_id)
    except Exception as e:
        logger.error("Failed to retrieve details for model version %s of '%s': %s", version_to_promote, args.model_name, e)
        sys.exit(1)
        
    # Check if already in production
    if mv_info.current_stage == "Production":
        logger.info("Model version %s is already in 'Production' stage. Nothing to do.", version_to_promote)
        sys.exit(0)
        
    # 3. Perform Promotion
    if args.dry_run:
        logger.info("[DRY-RUN] Model version %s of '%s' would be promoted to 'Production' stage (archiving existing versions).",
                    version_to_promote, args.model_name)
    else:
        logger.info("Transitioning model version %s of '%s' to 'Production' stage...", version_to_promote, args.model_name)
        try:
            client.transition_model_version_stage(
                name=args.model_name,
                version=version_to_promote,
                stage="Production",
                archive_existing_versions=True
            )
            logger.info("Successfully promoted model version %s of '%s' to 'Production'. Previous production versions archived.",
                        version_to_promote, args.model_name)
        except Exception as e:
            logger.error("Failed to promote model version %s to Production: %s", version_to_promote, e)
            sys.exit(1)

if __name__ == "__main__":
    main()
