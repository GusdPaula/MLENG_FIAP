"""Script to assign champion alias to a model in MLflow.

This script assigns the "champion" alias to a specified model version.
"""

import mlflow

# Configure MLflow tracking URI
MLFLOW_TRACKING_URI = "http://localhost:5001"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Model configuration
MODEL_NAME = "gmf_binary"  # Change this to your desired model
MODEL_VERSION = "4"  # Updated to version 4
ALIAS = "champion"


def assign_alias():
    """Assign alias to model version."""
    client = mlflow.tracking.MlflowClient()

    print(f"🏷️  Assigning alias '{ALIAS}' to {MODEL_NAME} version {MODEL_VERSION}")

    try:
        # Assign the alias using the correct method name
        client.set_registered_model_alias(
            name=MODEL_NAME, alias=ALIAS, version=MODEL_VERSION
        )
        print(
            f"✅ Successfully assigned alias '{ALIAS}' to {MODEL_NAME} version {MODEL_VERSION}"
        )

        # Verify the assignment
        model_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        print(
            f"📊 Verified: {MODEL_NAME}@{ALIAS} points to version {model_version.version}"
        )

    except Exception as e:
        print(f"❌ Failed to assign alias: {e}")


if __name__ == "__main__":
    assign_alias()
