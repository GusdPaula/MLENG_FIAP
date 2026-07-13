#!/usr/bin/env python3
"""
Script to configure Grafana CloudWatch datasources using AWS SSM to get admin credentials.
This script connects to the Grafana EC2 instance via SSM to get the admin password,
then configures CloudWatch Metrics and Logs datasources.
"""

import requests
import json
import sys
import time
import base64
import boto3
from pathlib import Path

# Configuration
GRAFANA_URL = "https://d3naqrkpy0vqtm.cloudfront.net"
AWS_REGION = "us-east-1"
DASHBOARD_PATH = Path(__file__).parent / "dashboards" / "model_monitoring.json"


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


def configure_cloudwatch_metrics_datasource(url, username, password):
    """Configure CloudWatch Metrics datasource."""
    print("Configuring CloudWatch Metrics datasource...")

    payload = {
        "name": "CloudWatch",
        "type": "cloudwatch",
        "access": "proxy",
        "jsonData": {
            "authType": "default",
            "defaultRegion": AWS_REGION
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
        print("CloudWatch Metrics datasource configured successfully")
        return True

    print(f"Failed to configure CloudWatch Metrics: {response.status_code} - {response.text}")
    return False


def configure_cloudwatch_logs_datasource(url, username, password):
    """Configure CloudWatch Logs datasource."""
    print("Configuring CloudWatch Logs datasource...")

    payload = {
        "name": "CloudWatch Logs",
        "type": "cloudwatch-logs",
        "access": "proxy",
        "jsonData": {
            "authType": "default",
            "defaultRegion": AWS_REGION
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
        print("CloudWatch Logs datasource configured successfully")
        return True

    print(f"Failed to configure CloudWatch Logs: {response.status_code} - {response.text}")
    return False


def get_datasource_uid(url, username, password, datasource_name="CloudWatch"):
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


def import_dashboard(url, username, password, dashboard_path):
    """Import the model monitoring dashboard."""
    print(f"Importing dashboard from {dashboard_path}...")

    if not dashboard_path.exists():
        print(f"Dashboard file not found: {dashboard_path}")
        return False

    with open(dashboard_path, 'r') as f:
        data = json.load(f)

    # Extract the dashboard object (the file might have it nested under "dashboard" key)
    dashboard = data.get("dashboard", data)

    # Remove the dashboard ID to allow importing as a new dashboard
    dashboard.pop("id", None)
    dashboard.pop("uid", None)

    # Get the actual datasource UID and replace the variable
    datasource_uid = get_datasource_uid(url, username, password, "CloudWatch")
    if datasource_uid:
        # Replace the variable with the actual UID in all targets
        for panel in dashboard.get("panels", []):
            for target in panel.get("targets", []):
                if "datasource" in target and target["datasource"].get("uid") == "${__datasource.uid}":
                    target["datasource"]["uid"] = datasource_uid
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


def install_cloudwatch_plugin(instance_id):
    """Install the CloudWatch plugin via SSM."""
    print(f"Installing CloudWatch plugin on instance {instance_id}...")

    ssm = boto3.client('ssm', region_name=AWS_REGION)

    # Install the CloudWatch plugin using the new grafana cli syntax
    command = "sudo grafana cli plugins install grafana-cloudwatch-datasource"

    try:
        response = ssm.send_command(
            InstanceIds=[instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={'commands': [command]},
            TimeoutSeconds=120
        )

        command_id = response['Command']['CommandId']
        print(f"SSM command sent: {command_id}")

        # Wait for command to complete
        time.sleep(15)

        # Get command output
        output = ssm.get_command_invocation(
            CommandId=command_id,
            InstanceId=instance_id
        )

        if output['Status'] == 'Success':
            print("CloudWatch plugin installed successfully")
            print(f"Output: {output.get('StandardOutputContent', 'N/A')}")
            return True
        else:
            print(f"Command failed: {output['Status']}")
            print(f"Error: {output.get('StandardErrorContent', 'N/A')}")
            return False

    except Exception as e:
        print(f"Failed to install CloudWatch plugin: {e}")
        return False


def main():
    """Main function to configure Grafana datasources."""
    print("=" * 60)
    print("Grafana Datasource Configuration Script (with SSM)")
    print("=" * 60)
    print(f"Grafana URL: {GRAFANA_URL}")
    print(f"AWS Region: {AWS_REGION}")
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
        print("Please ensure you're in the grafana-ec2 directory with terraform initialized")
        sys.exit(1)

    # Get admin password from instance
    admin_password = get_grafana_admin_password(instance_id)
    if not admin_password:
        print("Failed to get admin password. Please configure datasources manually.")
        print_manual_instructions()
        sys.exit(1)

    # Wait for Grafana to be ready
    if not wait_for_grafana(GRAFANA_URL):
        sys.exit(1)

    # Configure datasources with admin credentials
    success = True
    success &= configure_cloudwatch_metrics_datasource(GRAFANA_URL, "admin", admin_password)
    success &= configure_cloudwatch_logs_datasource(GRAFANA_URL, "admin", admin_password)

    # Get datasource UID and install CloudWatch plugin if needed
    datasource_uid = get_datasource_uid(GRAFANA_URL, "admin", admin_password, "CloudWatch")
    if datasource_uid:
        install_cloudwatch_plugin(instance_id)

    # Import dashboard
    success &= import_dashboard(GRAFANA_URL, "admin", admin_password, DASHBOARD_PATH)

    print()
    if success:
        print("✅ All datasources configured successfully!")
        print("You can now view the model monitoring dashboard in Grafana.")
    else:
        print("❌ Some configurations failed. Please check the errors above.")
        sys.exit(1)


def print_manual_instructions():
    """Print manual configuration instructions."""
    print()
    print("=" * 60)
    print("MANUAL CONFIGURATION INSTRUCTIONS")
    print("=" * 60)
    print()
    print("1. Access Grafana at: https://d3naqrkpy0vqtm.cloudfront.net")
    print("2. Login with admin credentials (get password from EC2 instance)")
    print("3. Go to Configuration → Data Sources")
    print("4. Click 'Add data source' and search for 'CloudWatch'")
    print("5. Configure CloudWatch Metrics:")
    print("   - Name: CloudWatch")
    print("   - Auth Type: Default (uses EC2 instance profile)")
    print("   - Default Region: us-east-1")
    print("   - Click 'Save & Test'")
    print("6. Add CloudWatch Logs datasource similarly:")
    print("   - Name: CloudWatch Logs")
    print("   - Auth Type: Default")
    print("   - Default Region: us-east-1")
    print("   - Click 'Save & Test'")
    print("7. Import the dashboard:")
    print("   - Go to Dashboards → Import")
    print("   - Upload model_monitoring.json")
    print("   - Select CloudWatch datasource for all panels")
    print("   - Click 'Import'")
    print()


if __name__ == "__main__":
    main()
