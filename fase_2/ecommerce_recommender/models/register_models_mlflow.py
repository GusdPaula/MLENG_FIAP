"""Script to register local models in MLflow model registry.

This script loops through all model files in models/mlflow_experiments/
and registers them in the MLflow model registry.
"""

import mlflow
from pathlib import Path

# Configure MLflow tracking URI
MLFLOW_TRACKING_URI = "http://localhost:5001"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Path to local models
MODELS_DIR = Path("models/mlflow_experiments")

# List of model files to register
MODEL_FILES = [
    "gmf_binary.pt",
    "gmf_implicit.pt",
    "gmf_weighted.pt",
    "matrix_factorization_binary.pt",
    "matrix_factorization_implicit.pt",
    "matrix_factorization_weighted.pt",
    "ncf_binary.pt",
    "ncf_implicit.pt",
    "ncf_weighted.pt",
]

def register_model(model_file: str) -> None:
    """Register a single model in MLflow model registry as an artifact.

    Args:
        model_file: Name of the model file.
    """
    model_path = MODELS_DIR / model_file
    model_name = model_file.replace(".pt", "")

    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return

    print(f"📦 Registering model: {model_name}")

    try:
        # Start a new MLflow run
        with mlflow.start_run(run_name=f"register_{model_name}"):
            # Log the model as an artifact
            mlflow.log_artifact(str(model_path), artifact_path=model_name)

            # Log model metadata
            import torch
            checkpoint = torch.load(model_path, map_location="cpu")
            model_type = checkpoint.get("model_type", "unknown")
            mlflow.log_param("model_type", model_type)

            # Get the run info
            run_id = mlflow.active_run().info.run_id
            artifact_uri = f"{mlflow.get_artifact_uri()}/{model_name}/{model_file}"

            # Register the model using the client API
            client = mlflow.tracking.MlflowClient()

            # Create registered model if it doesn't exist
            try:
                client.create_registered_model(model_name)
                print(f"✅ Created registered model: {model_name}")
            except Exception:
                print(f"ℹ️  Registered model already exists: {model_name}")

            # Create a new model version
            try:
                model_version = client.create_model_version(
                    name=model_name,
                    source=artifact_uri,
                    run_id=run_id
                )
                print(f"✅ Created model version {model_version.version} for: {model_name}")
            except Exception as e:
                print(f"⚠️  Failed to create model version: {e}")

    except Exception as e:
        print(f"❌ Failed to register {model_name}: {e}")


def main():
    """Main function to register all models."""
    print(f"🚀 Connecting to MLflow at {MLFLOW_TRACKING_URI}")

    for model_file in MODEL_FILES:
        register_model(model_file)

    print("\n✨ Model registration complete!")
    print(f"📊 View models at: {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
