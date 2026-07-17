#!/usr/bin/env python3
"""
Deploy cold start solution to ECS service.

This script:
1. Gets the current ECS task definition
2. Updates it with cold start environment variables
3. Registers the new task definition
4. Updates the ECS service to use the new task definition
"""


import boto3

REGION = "us-east-1"
ECS_CLUSTER = "mlflow-fiap-api-cluster"
ECS_SERVICE = "mlflow-fiap-api-service"
TASK_DEFINITION_FAMILY = "mlflow-fiap-api-task"

# Cold start artifact paths in the container
FEATURE_EXTRACTOR_PATH = "ecommerce_recommender/models/item_feature_extractor.pkl"
CONTENT_RECOMMENDER_PATH = "ecommerce_recommender/models/content_recommender.pkl"


def get_current_task_definition():
    """Get the current ECS task definition."""
    ecs = boto3.client("ecs", region_name=REGION)
    try:
        response = ecs.describe_task_definition(taskDefinition=TASK_DEFINITION_FAMILY)
        print(f"✅ Retrieved current task definition: {TASK_DEFINITION_FAMILY}")
        return response["taskDefinition"]
    except Exception as e:
        print(f"❌ Failed to get task definition: {e}")
        return None


def update_task_definition_with_cold_start(task_def):
    """Update task definition with cold start environment variables.

    Args:
        task_def: Current task definition dictionary.

    Returns:
        Updated task definition dictionary.
    """
    # Update container definitions with cold start environment variables
    for container in task_def["containerDefinitions"]:
        if "environment" not in container:
            container["environment"] = []

        # Remove existing cold start variables if they exist
        container["environment"] = [
            env for env in container["environment"]
            if env["name"] not in ["FEATURE_EXTRACTOR_PATH", "CONTENT_RECOMMENDER_PATH"]
        ]

        # Add cold start environment variables
        container["environment"].extend([
            {
                "name": "FEATURE_EXTRACTOR_PATH",
                "value": FEATURE_EXTRACTOR_PATH
            },
            {
                "name": "CONTENT_RECOMMENDER_PATH",
                "value": CONTENT_RECOMMENDER_PATH
            }
        ])

    print("✅ Updated task definition with cold start environment variables")
    return task_def


def register_new_task_definition(task_def):
    """Register the new task definition.

    Args:
        task_def: Task definition dictionary to register.

    Returns:
        Registered task definition ARN.
    """
    ecs = boto3.client("ecs", region_name=REGION)

    # Remove fields that can't be included in registration
    clean_task_def = {
        "family": task_def["family"],
        "taskRoleArn": task_def.get("taskRoleArn"),
        "executionRoleArn": task_def.get("executionRoleArn"),
        "networkMode": task_def.get("networkMode"),
        "containerDefinitions": task_def["containerDefinitions"],
        "requiresCompatibilities": task_def.get("requiresCompatibilities"),
        "cpu": task_def.get("cpu"),
        "memory": task_def.get("memory"),
        "volumes": task_def.get("volumes", []),
        "placementConstraints": task_def.get("placementConstraints", []),
        "requiresCompatibilities": task_def.get("requiresCompatibilities", []),
    }

    try:
        response = ecs.register_task_definition(**clean_task_def)
        new_revision = response["taskDefinition"]["revision"]
        print(f"✅ Registered new task definition: {TASK_DEFINITION_FAMILY}:{new_revision}")
        return response["taskDefinition"]["taskDefinitionArn"]
    except Exception as e:
        print(f"❌ Failed to register task definition: {e}")
        return None


def update_ecs_service(task_def_arn):
    """Update the ECS service to use the new task definition.

    Args:
        task_def_arn: ARN of the new task definition.
    """
    ecs = boto3.client("ecs", region_name=REGION)

    try:
        response = ecs.update_service(
            cluster=ECS_CLUSTER,
            service=ECS_SERVICE,
            taskDefinition=task_def_arn
        )
        print(f"✅ Updated ECS service: {ECS_SERVICE}")

        # Wait for service to stabilize
        print("⏳ Waiting for service to stabilize...")
        waiter = ecs.get_waiter("services_stable")
        waiter.wait(cluster=ECS_CLUSTER, services=[ECS_SERVICE])
        print("✅ ECS service is now stable")

        return True
    except Exception as e:
        print(f"❌ Failed to update ECS service: {e}")
        return False


def main():
    """Deploy cold start solution to ECS."""
    print("🚀 Deploying cold start solution to ECS...\n")

    # Get current task definition
    task_def = get_current_task_definition()
    if not task_def:
        print("❌ Deployment failed: Could not get current task definition")
        return

    # Update with cold start configuration
    updated_task_def = update_task_definition_with_cold_start(task_def)

    # Register new task definition
    task_def_arn = register_new_task_definition(updated_task_def)
    if not task_def_arn:
        print("❌ Deployment failed: Could not register new task definition")
        return

    # Update ECS service
    success = update_ecs_service(task_def_arn)

    if success:
        print("\n✨ Cold start deployment complete!")
        print("📋 Environment variables added:")
        print(f"   - FEATURE_EXTRACTOR_PATH={FEATURE_EXTRACTOR_PATH}")
        print(f"   - CONTENT_RECOMMENDER_PATH={CONTENT_RECOMMENDER_PATH}")
        print("\n⚠️  Make sure the cold start artifacts are copied to the ECS container:")
        print(f"   - {FEATURE_EXTRACTOR_PATH}")
        print(f"   - {CONTENT_RECOMMENDER_PATH}")
    else:
        print("\n❌ Deployment failed: Could not update ECS service")


if __name__ == "__main__":
    main()
