#!/usr/bin/env python3
"""
Script to configure Prometheus datasource and import API dashboard to Grafana.
"""

import requests
import json
import sys
import time
import boto3
from pathlib import Path

# Configuration
GRAFANA_URL = "https://d3naqrkpy0vqtm.cloudfront.net"
PROMETHEUS_URL = "http://172.31.84.126:9090"
AWS_REGION = "us-east-1"
DASHBOARD_PATH = Path(__file__).parent / "dashboards" / "api-overview.json"


def get_grafana_admin_password(instance_id):
    """Get Grafana admin password from EC2 instance via SSM."""
    print(f"Getting Grafana admin password from instance {instance_id}...")

    ssm = boto3.client('ssm', region_name=AWS_REGION)

    # Command to read the admin password from grafana.ini
    command = "sudo grep admin_password /etc/grafana/grafana.ini | cut -d '=' -f2 | xargs"

    try:
        response = ssm.send_command(
            InstanceIds=[instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={'commands': [command]},
            TimeoutSeconds=30
        )

        command_id = response['Command']['CommandId']
        print(f"SSM command sent: {command_id}")

        # Wait for command to complete
        time.sleep(5)

        # Get command output
        output = ssm.get_command_invocation(
            CommandId=command_id,
            InstanceId=instance_id
        )

        if output['Status'] == 'Success':
            password = output['StandardOutputContent'].strip()
            if password:
                print("Admin password retrieved successfully")
                return password
            else:
                print("Password is empty")
                return None
        else:
            print(f"Command failed: {output['Status']}")
            print(f"Error: {output.get('StandardErrorContent', 'N/A')}")
            return None

    except Exception as e:
        print(f"Failed to get admin password via SSM: {e}")
        return None


def wait_for_grafana(url, timeout=60):
    """Wait for Grafana to be ready."""
    print(f"Waiting for Grafana at {url}...")
    for i in range(timeout):
        try:
            response = requests.get(f"{url}/api/health", timeout=5)
            if response.status_code == 200:
                print("Grafana is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        print(f"Waiting... ({i+1}/{timeout})")
        time.sleep(1)
    print("Grafana did not become ready in time")
    return False


def configure_prometheus_datasource(url, username, password, prometheus_url):
    """Configure Prometheus datasource."""
    print(f"Configuring Prometheus datasource at {prometheus_url}...")

    payload = {
        "name": "Prometheus",
        "type": "prometheus",
        "access": "proxy",
        "url": prometheus_url,
        "isDefault": True,
        "jsonData": {
            "timeInterval": "15s",
            "httpMethod": "POST"
        }
    }

    response = requests.post(
        f"{url}/api/datasources",
        auth=(username, password),
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=10
    )

    if response.status_code in [200, 409]:  # 409 means datasource already exists
        print("Prometheus datasource configured successfully")
        return True

    print(f"Failed to configure Prometheus: {response.status_code} - {response.text}")
    return False


def get_datasource_uid(url, username, password, datasource_name="Prometheus"):
    """Get the actual datasource UID from Grafana."""
    print(f"Getting datasource UID for {datasource_name}...")

    response = requests.get(
        f"{url}/api/datasources",
        auth=(username, password),
        timeout=10
    )

    if response.status_code == 200:
        datasources = response.json()
        for ds in datasources:
            if ds.get("name") == datasource_name:
                uid = ds.get("uid")
                print(f"Found datasource UID: {uid}")
                return uid

    print(f"Datasource {datasource_name} not found")
    return None


def import_dashboard(url, username, password, dashboard_path, datasource_uid):
    """Import the API dashboard."""
    print(f"Importing dashboard from {dashboard_path}...")

    if not dashboard_path.exists():
        print(f"Dashboard file not found: {dashboard_path}")
        return False

    with open(dashboard_path, 'r') as f:
        data = json.load(f)

    # Extract the dashboard object
    dashboard = data.get("dashboard", data)

    # Remove the dashboard ID to allow importing as a new dashboard
    dashboard.pop("id", None)
    dashboard.pop("uid", None)

    # Update datasource references
    if datasource_uid:
        for panel in dashboard.get("panels", []):
            for target in panel.get("targets", []):
                if "datasource" in target:
                    if isinstance(target["datasource"], dict):
                        target["datasource"]["uid"] = datasource_uid
                        target["datasource"]["type"] = "prometheus"
                    elif isinstance(target["datasource"], str):
                        target["datasource"] = {"uid": datasource_uid, "type": "prometheus"}
        print(f"Updated datasource references to use UID: {datasource_uid}")

    payload = {
        "dashboard": dashboard,
        "overwrite": True,
        "message": "Imported via setup script"
    }

    response = requests.post(
        f"{url}/api/dashboards/db",
        auth=(username, password),
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=10
    )

    if response.status_code == 200:
        data = response.json()
        print(f"Dashboard imported successfully: {data.get('url', 'N/A')}")
        return True

    print(f"Failed to import dashboard: {response.status_code} - {response.text}")
    return False


def main():
    """Main function to configure Grafana datasources."""
    print("=" * 60)
    print("Grafana Prometheus Datasource Configuration")
    print("=" * 60)
    print(f"Grafana URL: {GRAFANA_URL}")
    print(f"Prometheus URL: {PROMETHEUS_URL}")
    print()

    # Get instance ID from Terraform output
    import subprocess
    try:
        instance_id = subprocess.check_output(
            ["terraform", "output", "-raw", "grafana_instance_id"],
            cwd=Path(__file__).parent
        ).decode().strip()
        print(f"Grafana EC2 Instance ID: {instance_id}")
    except Exception as e:
        print(f"Failed to get instance ID: {e}")
        sys.exit(1)

    # Get admin password from instance
    admin_password = get_grafana_admin_password(instance_id)
    if not admin_password:
        print("Failed to get admin password.")
        sys.exit(1)

    # Wait for Grafana to be ready
    if not wait_for_grafana(GRAFANA_URL):
        sys.exit(1)

    # Configure Prometheus datasource
    if not configure_prometheus_datasource(GRAFANA_URL, "admin", admin_password, PROMETHEUS_URL):
        sys.exit(1)

    # Get datasource UID
    datasource_uid = get_datasource_uid(GRAFANA_URL, "admin", admin_password, "Prometheus")
    if not datasource_uid:
        print("Failed to get Prometheus datasource UID")
        sys.exit(1)

    # Import dashboard
    if not import_dashboard(GRAFANA_URL, "admin", admin_password, DASHBOARD_PATH, datasource_uid):
        sys.exit(1)

    print()
    print("✅ Prometheus datasource and dashboard configured successfully!")
    print("You can now view the API dashboard in Grafana.")


if __name__ == "__main__":
    main()
