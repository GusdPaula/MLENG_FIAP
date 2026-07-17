#!/usr/bin/env python3
"""
Script to stop AWS resources to reduce costs.

Stops:
- MLflow EC2 instance
- Grafana EC2 instance
- Prometheus EC2 instance
- ECS API service (sets desired count to 0)
- Optionally stops RDS database (uncomment if needed)
"""

import time

import boto3

REGION = "us-east-1"
ECS_CLUSTER = "mlflow-fiap-api-cluster"
ECS_SERVICE = "mlflow-fiap-api-service"
RDS_INSTANCE_ID = "mlflow-fiap-db"  # Uncomment if you want to stop RDS
GRAFANA_INSTANCE_TAG = "mlflow-fiap-grafana-server"  # Grafana EC2 instance tag
PROMETHEUS_INSTANCE_TAG = "mlflow-fiap-prometheus"  # Prometheus EC2 instance tag


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


def stop_grafana_service(instance_id):
    """Stop Grafana services gracefully using SSM."""
    ssm = boto3.client("ssm", region_name=REGION)
    try:
        script = """#!/bin/bash
set -e
echo "=== Stopping Grafana Services ==="
systemctl stop grafana-server
systemctl stop nginx
echo "Grafana services stopped"
"""
        response = ssm.send_command(
            InstanceIds=[instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": [script]},
            TimeoutSeconds=30,
        )

        command_id = response["Command"]["CommandId"]
        print(f"⏳ Stopping Grafana services (command: {command_id})...")
        time.sleep(10)

        output = ssm.get_command_invocation(
            CommandId=command_id, InstanceId=instance_id
        )

        if output["Status"] == "Success":
            print("✅ Grafana services stopped gracefully")
        else:
            print(f"⚠️  Grafana services stop status: {output['Status']}")
    except Exception as e:
        print(f"❌ Failed to stop Grafana services: {e}")


def stop_prometheus_service(instance_id):
    """Stop Prometheus Docker container gracefully using SSM."""
    ssm = boto3.client("ssm", region_name=REGION)
    try:
        script = """#!/bin/bash
set -e
echo "=== Stopping Prometheus Service ==="
docker stop prometheus || true
echo "Prometheus container stopped"
"""
        response = ssm.send_command(
            InstanceIds=[instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": [script]},
            TimeoutSeconds=30,
        )

        command_id = response["Command"]["CommandId"]
        print(f"⏳ Stopping Prometheus service (command: {command_id})...")
        time.sleep(10)

        output = ssm.get_command_invocation(
            CommandId=command_id, InstanceId=instance_id
        )

        if output["Status"] == "Success":
            print("✅ Prometheus service stopped gracefully")
        else:
            print(f"⚠️  Prometheus service stop status: {output['Status']}")
    except Exception as e:
        print(f"❌ Failed to stop Prometheus service: {e}")


def stop_mlflow_ec2():
    """Stop the MLflow EC2 instance."""
    mlflow_instance_id = get_mlflow_instance_id()
    if not mlflow_instance_id:
        return

    ec2 = boto3.client("ec2", region_name=REGION)
    try:
        ec2.stop_instances(InstanceIds=[mlflow_instance_id])
        print(f"✅ Stopped MLflow EC2 instance: {mlflow_instance_id}")
    except Exception as e:
        print(f"❌ Failed to stop MLflow EC2: {e}")


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


def stop_grafana_ec2():
    """Stop the Grafana EC2 instance."""
    grafana_instance_id = get_grafana_instance_id()
    if not grafana_instance_id:
        return

    # Stop Grafana services gracefully before stopping instance
    stop_grafana_service(grafana_instance_id)

    ec2 = boto3.client("ec2", region_name=REGION)
    try:
        ec2.stop_instances(InstanceIds=[grafana_instance_id])
        print(f"✅ Stopped Grafana EC2 instance: {grafana_instance_id}")
    except Exception as e:
        print(f"❌ Failed to stop Grafana EC2: {e}")


def get_prometheus_instance_id():
    """Get the Prometheus EC2 instance ID by tag name."""
    ec2 = boto3.client("ec2", region_name=REGION)
    try:
        response = ec2.describe_instances(
            Filters=[{"Name": "tag:Name", "Values": [PROMETHEUS_INSTANCE_TAG]}]
        )
        if response["Reservations"]:
            return response["Reservations"][0]["Instances"][0]["InstanceId"]
        else:
            print(f"❌ No Prometheus EC2 instance found with tag '{PROMETHEUS_INSTANCE_TAG}'")
            return None
    except Exception as e:
        print(f"❌ Failed to get Prometheus instance ID: {e}")
        return None


def stop_prometheus_ec2():
    """Stop the Prometheus EC2 instance."""
    prometheus_instance_id = get_prometheus_instance_id()
    if not prometheus_instance_id:
        return

    # Stop Prometheus Docker container gracefully before stopping instance
    stop_prometheus_service(prometheus_instance_id)

    ec2 = boto3.client("ec2", region_name=REGION)
    try:
        ec2.stop_instances(InstanceIds=[prometheus_instance_id])
        print(f"✅ Stopped Prometheus EC2 instance: {prometheus_instance_id}")
    except Exception as e:
        print(f"❌ Failed to stop Prometheus EC2: {e}")


def stop_ecs_service():
    """Stop the ECS API service by setting desired count to 0."""
    ecs = boto3.client("ecs", region_name=REGION)
    try:
        ecs.update_service(cluster=ECS_CLUSTER, service=ECS_SERVICE, desiredCount=0)
        print(f"✅ Stopped ECS service: {ECS_SERVICE} (desired count set to 0)")
    except Exception as e:
        print(f"❌ Failed to stop ECS service: {e}")


def stop_rds():
    """Stop the RDS database (optional)."""
    rds = boto3.client("rds", region_name=REGION)
    try:
        rds.stop_db_instance(DBInstanceIdentifier=RDS_INSTANCE_ID)
        print(f"✅ Stopped RDS instance: {RDS_INSTANCE_ID}")
    except Exception as e:
        print(f"❌ Failed to stop RDS: {e}")


if __name__ == "__main__":
    print("🛑 Stopping AWS resources to reduce costs...")
    stop_mlflow_ec2()
    stop_grafana_ec2()
    stop_prometheus_ec2()
    stop_ecs_service()
    # stop_rds()  # Uncomment to stop RDS (data will be preserved but connection will be lost)
    print("\n✨ Resource stopping complete!")
