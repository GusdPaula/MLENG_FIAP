#!/usr/bin/env python3
"""
Script to configure Grafana CloudWatch datasources remotely.
This script connects to Grafana via API and configures CloudWatch Metrics and Logs datasources.
"""

import json
import sys
import time
from pathlib import Path

import requests

# Configuration
GRAFANA_URL = "https://d3naqrkpy0vqtm.cloudfront.net"  # Update if different
AWS_REGION = "us-east-1"
DASHBOARD_PATH = Path(__file__).parent / "dashboards" / "model_monitoring.json"


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


def create_api_key(url):
    """Create a Grafana API key."""
    print("Creating Grafana API key...")

    # Try to create API key without auth (might work with anonymous access)
    response = requests.post(
        f"{url}/api/auth/keys",
        headers={"Content-Type": "application/json"},
        json={"name": "setup-key", "role": "Admin"},
        timeout=10
    )

    if response.status_code == 200:
        data = response.json()
        api_key = data.get("key")
        if api_key and api_key != "null":
            print("API key created successfully")
            return api_key

    # If that fails, try to list existing datasources without auth
    print("Trying to use anonymous access...")
    return "anonymous"


def configure_cloudwatch_metrics_datasource(url, api_key):
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

    headers = {"Content-Type": "application/json"}
    if api_key != "anonymous":
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.post(
        f"{url}/api/datasources",
        headers=headers,
        json=payload,
        timeout=10
    )

    if response.status_code in [200, 409]:  # 409 means datasource already exists
        print("CloudWatch Metrics datasource configured successfully")
        return True

    print(f"Failed to configure CloudWatch Metrics: {response.status_code} - {response.text}")
    return False


def configure_cloudwatch_logs_datasource(url, api_key):
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

    headers = {"Content-Type": "application/json"}
    if api_key != "anonymous":
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.post(
        f"{url}/api/datasources",
        headers=headers,
        json=payload,
        timeout=10
    )

    if response.status_code in [200, 409]:  # 409 means datasource already exists
        print("CloudWatch Logs datasource configured successfully")
        return True

    print(f"Failed to configure CloudWatch Logs: {response.status_code} - {response.text}")
    return False


def import_dashboard(url, api_key, dashboard_path):
    """Import the model monitoring dashboard."""
    print(f"Importing dashboard from {dashboard_path}...")

    if not dashboard_path.exists():
        print(f"Dashboard file not found: {dashboard_path}")
        return False

    with open(dashboard_path, 'r') as f:
        dashboard = json.load(f)

    # Remove the dashboard ID to allow importing as a new dashboard
    dashboard.pop("id", None)
    dashboard.pop("uid", None)

    payload = {
        "dashboard": dashboard,
        "overwrite": True,
        "message": "Imported via setup script"
    }

    headers = {"Content-Type": "application/json"}
    if api_key != "anonymous":
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.post(
        f"{url}/api/dashboards/db",
        headers=headers,
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
    print("Grafana Datasource Configuration Script")
    print("=" * 60)
    print(f"Grafana URL: {GRAFANA_URL}")
    print(f"AWS Region: {AWS_REGION}")
    print()

    # Wait for Grafana to be ready
    if not wait_for_grafana(GRAFANA_URL):
        sys.exit(1)

    # Create API key
    api_key = create_api_key(GRAFANA_URL)
    if not api_key:
        sys.exit(1)

    # Configure datasources
    success = True
    success &= configure_cloudwatch_metrics_datasource(GRAFANA_URL, api_key)
    success &= configure_cloudwatch_logs_datasource(GRAFANA_URL, api_key)

    # Import dashboard
    success &= import_dashboard(GRAFANA_URL, api_key, DASHBOARD_PATH)

    print()
    if success:
        print("✅ All datasources configured successfully!")
        print("You can now view the model monitoring dashboard in Grafana.")
    else:
        print("❌ Some configurations failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
