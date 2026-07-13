#!/usr/bin/env python3
"""
Script to start AWS resources.

Starts:
- MLflow EC2 instance
- Grafana EC2 instance
- ECS API service (sets desired count to 1)
- Grafana workspace status check
- Optionally starts RDS database (uncomment if needed)
"""

import time

import boto3

REGION = "us-east-1"
ECS_CLUSTER = "mlflow-fiap-api-cluster"
ECS_SERVICE = "mlflow-fiap-api-service"
RDS_INSTANCE_ID = "mlflow-fiap-db"  # Uncomment if you want to start RDS
GRAFANA_INSTANCE_TAG = "ml-recommender-grafana-server"  # Grafana EC2 instance tag


def get_mlflow_instance_id():
    """Get the MLflow EC2 instance ID by tag name."""
    ec2 = boto3.client("ec2", region_name=REGION)
    try:
        response = ec2.describe_instances(
            Filters=[{"Name": "tag:Name", "Values": ["mlflow-fiap-server"]}]
        )
        if response["Reservations"]:
            return response["Reservations"][0]["Instances"][0]["InstanceId"]
        else:
            print("❌ No MLflow EC2 instance found with tag 'mlflow-fiap-server'")
            return None
    except Exception as e:
        print(f"❌ Failed to get MLflow instance ID: {e}")
        return None


def start_mlflow_ec2():
    """Start the MLflow EC2 instance."""
    mlflow_instance_id = get_mlflow_instance_id()
    if not mlflow_instance_id:
        return

    ec2 = boto3.client("ec2", region_name=REGION)
    try:
        ec2.start_instances(InstanceIds=[mlflow_instance_id])
        print(f"✅ Started MLflow EC2 instance: {mlflow_instance_id}")
        print("⏳ Waiting for instance to be running...")
        ec2.get_waiter("instance_running").wait(InstanceIds=[mlflow_instance_id])
        print("✅ MLflow EC2 instance is now running")

        # Restart MLflow server on the instance
        restart_mlflow_server(mlflow_instance_id)
    except Exception as e:
        print(f"❌ Failed to start MLflow EC2: {e}")


def get_grafana_instance_id():
    """Get the Grafana EC2 instance ID by tag name."""
    ec2 = boto3.client("ec2", region_name=REGION)
    try:
        response = ec2.describe_instances(
            Filters=[{"Name": "tag:Name", "Values": [GRAFANA_INSTANCE_TAG]}]
        )
        if response["Reservations"]:
            return response["Reservations"][0]["Instances"][0]["InstanceId"]
        else:
            print(f"❌ No Grafana EC2 instance found with tag '{GRAFANA_INSTANCE_TAG}'")
            return None
    except Exception as e:
        print(f"❌ Failed to get Grafana instance ID: {e}")
        return None


def start_grafana_ec2():
    """Start the Grafana EC2 instance."""
    grafana_instance_id = get_grafana_instance_id()
    if not grafana_instance_id:
        return

    ec2 = boto3.client("ec2", region_name=REGION)
    try:
        ec2.start_instances(InstanceIds=[grafana_instance_id])
        print(f"✅ Started Grafana EC2 instance: {grafana_instance_id}")
        print("⏳ Waiting for instance to be running...")
        ec2.get_waiter("instance_running").wait(InstanceIds=[grafana_instance_id])
        print("✅ Grafana EC2 instance is now running")
    except Exception as e:
        print(f"❌ Failed to start Grafana EC2: {e}")


def restart_mlflow_server(instance_id):
    """Restart the MLflow server on the EC2 instance using SSM."""
    ssm = boto3.client("ssm", region_name=REGION)

    print("⏳ Waiting for SSM agent to be ready...")
    time.sleep(30)  # Wait for SSM agent to initialize

    try:
        # Use a single bash script to ensure environment variables persist
        script = """
#!/bin/bash
set -e
echo "=== Restarting MLflow Server ==="
cd /home/ubuntu/mlflow
export AWS_DEFAULT_REGION=us-east-1
DB_PASSWORD=$(aws secretsmanager get-secret-value --secret-id arn:aws:secretsmanager:us-east-1:043929685977:secret:mlflow-fiap-db-password-new-7jy4yo6d-lvGVUa --query SecretString --output text)
pkill -f "mlflow server" || true
sleep 2
nohup /opt/mlflow-venv/bin/mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri postgresql://mlflow_user:$DB_PASSWORD@mlflow-fiap-db.cyl4su8oo8x9.us-east-1.rds.amazonaws.com/mlflow --default-artifact-root "s3://mlflow-artifacts-fiap-7jy4yo6d" --serve-artifacts --allowed-hosts "*" --cors-allowed-origins "*" > /var/log/mlflow-server.log 2>&1 &
echo "MLflow server started"
sleep 5
curl -f http://localhost:5000/health || echo "Health check failed"
"""

        response = ssm.send_command(
            InstanceIds=[instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": [script]},
            TimeoutSeconds=60,
        )

        command_id = response["Command"]["CommandId"]
        print(f"⏳ Restarting MLflow server (command: {command_id})...")

        # Wait for command to complete
        time.sleep(15)

        output = ssm.get_command_invocation(
            CommandId=command_id, InstanceId=instance_id
        )

        if output["Status"] == "Success":
            print("✅ MLflow server restart command completed")
            print(
                f"📋 Output: {output.get('StandardOutputContent', 'No output')[-500:]}"
            )
        else:
            print(f"⚠️  MLflow server restart command status: {output['Status']}")
            if output.get("StandardErrorContent"):
                print(f"❌ Error: {output['StandardErrorContent']}")

    except Exception as e:
        print(f"❌ Failed to restart MLflow server: {e}")
        print("ℹ️  You may need to manually restart MLflow on the instance")


def start_ecs_service():
    """Start the ECS API service by setting desired count to 1."""
    ecs = boto3.client("ecs", region_name=REGION)
    try:
        ecs.update_service(cluster=ECS_CLUSTER, service=ECS_SERVICE, desiredCount=1)
        print(f"✅ Started ECS service: {ECS_SERVICE} (desired count set to 1)")
        print("⏳ Waiting for service to be stable...")
        ecs.get_waiter("services_stable").wait(
            cluster=ECS_CLUSTER, services=[ECS_SERVICE]
        )
        print("✅ ECS service is now stable")
    except Exception as e:
        print(f"❌ Failed to start ECS service: {e}")


def start_rds():
    """Start the RDS database (optional)."""
    rds = boto3.client("rds", region_name=REGION)
    try:
        rds.start_db_instance(DBInstanceIdentifier=RDS_INSTANCE_ID)
        print(f"✅ Started RDS instance: {RDS_INSTANCE_ID}")
        print("⏳ Waiting for instance to be available...")
        rds.get_waiter("db_instance_available").wait(
            DBInstanceIdentifier=RDS_INSTANCE_ID
        )
        print("✅ RDS instance is now available")
    except Exception as e:
        print(f"❌ Failed to start RDS: {e}")


def check_grafana_workspace():
    """Check Grafana workspace status and display URL."""
    grafana = boto3.client("grafana", region_name=REGION)
    try:
        response = grafana.list_workspaces()
        workspace_name = "ml-recommender-grafana"
        workspace = None
        for ws in response["workspaces"]:
            if ws["name"] == workspace_name:
                workspace = ws
                break

        if workspace:
            print(f"✅ Grafana workspace '{workspace_name}' is {workspace['status']}")
            print(f"🔗 Grafana URL: {workspace['endpoint']}")
            return workspace["endpoint"]
        else:
            print(f"❌ No Grafana workspace found with name '{workspace_name}'")
            return None
    except Exception as e:
        print(f"❌ Failed to check Grafana workspace: {e}")
        return None


if __name__ == "__main__":
    print("🚀 Starting AWS resources...")
    start_mlflow_ec2()
    start_grafana_ec2()
    start_ecs_service()
    check_grafana_workspace()
    # start_rds()  # Uncomment to start RDS
    print("\n✨ Resource starting complete!")
