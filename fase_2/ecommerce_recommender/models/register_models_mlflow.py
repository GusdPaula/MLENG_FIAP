"""Script to register local models in MLflow model registry.

This script loops through all model files in models/mlflow_experiments/
and registers them in the MLflow model registry using proper MLflow PyTorch model format.
"""

import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch

# Configure MLflow tracking URI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Path to local models (works both locally and in Docker)
MODELS_DIR = Path(__file__).parent.parent / "models" / "mlflow_experiments"

# List of model files to register
MODEL_FILES = [
    "gmf_binary.pt",
]


class CheckpointWrapper(torch.nn.Module):
    """Wrapper class to make checkpoint compatible with mlflow.pytorch.log_model."""

    def __init__(self, checkpoint):
        super().__init__()
        self.checkpoint = checkpoint

    def forward(self, x):
        raise NotImplementedError("This is a wrapper for checkpoint storage only")


def register_model(model_file: str) -> None:
    """Register a single model in MLflow model registry.

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
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")
        model_type = checkpoint.get("model_type", "unknown")

        # Wrap checkpoint in a PyTorch module for MLflow logging
        wrapped_model = CheckpointWrapper(checkpoint)

        # Start a new MLflow run
        with mlflow.start_run(run_name=f"register_{model_name}"):
            # Log the model using MLflow's PyTorch model logging
            mlflow.pytorch.log_model(
                wrapped_model,
                artifact_path=model_name,
                registered_model_name=model_name,
            )

            # Log model metadata
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("model_file", model_file)

            # Set the champion alias
            client = mlflow.tracking.MlflowClient()
            try:
                latest_version = client.get_latest_versions(model_name)[0]
                client.set_registered_model_alias(
                    name=model_name, alias="champion", version=latest_version.version
                )
                print(
                    f"✅ Set champion alias for {model_name} version {latest_version.version}"
                )
            except Exception as e:
                print(f"⚠️  Failed to set champion alias: {e}")

            print(f"✅ Successfully registered {model_name} as MLflow PyTorch model")

    except Exception as e:
        print(f"❌ Failed to register {model_name}: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Main function to register all models."""
    print(f"🚀 Connecting to MLflow at {MLFLOW_TRACKING_URI}")

    for model_file in MODEL_FILES:
        register_model(model_file)

    print("\n✨ Model registration complete!")
    print(f"📊 View models at: {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
