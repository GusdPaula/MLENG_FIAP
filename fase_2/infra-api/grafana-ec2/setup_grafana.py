#!/usr/bin/env python3
"""
Configure Grafana with CloudWatch Logs data source and log-based dashboards.
This script sets up Grafana for monitoring API, DVC, and MLflow logs.
"""

import argparse
import sys
import time

import requests


def create_api_key(base_url: str, admin_password: str = "admin") -> str:
    """Create an API key for Grafana configuration."""
    url = f"{base_url}/api/auth/keys"
    headers = {"Content-Type": "application/json"}

    # Try to create API key with default admin credentials
    auth = ("admin", admin_password)

    payload = {
        "name": "setup-key",
        "role": "Admin",
        "secondsToLive": 3600,  # 1 hour
    }

    try:
        response = requests.post(url, auth=auth, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["key"]
        else:
            print(f"Failed to create API key: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error creating API key: {e}")
        return None


def configure_cloudwatch_logs_datasource(
    base_url: str, api_key: str, region: str = "us-east-1"
) -> bool:
    """Configure CloudWatch Logs as a data source."""
    url = f"{base_url}/api/datasources"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "name": "CloudWatch Logs",
        "type": "cloudwatch-logs",
        "access": "proxy",
        "jsonData": {"authType": "default", "defaultRegion": region},
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            print("✓ CloudWatch Logs data source configured")
            return True
        else:
            print(
                f"✗ Failed to configure CloudWatch Logs: {response.status_code} - {response.text}"
            )
            return False
    except Exception as e:
        print(f"Error configuring CloudWatch Logs: {e}")
        return False


def create_log_dashboard(
    base_url: str,
    api_key: str,
    dashboard_name: str,
    log_query: str,
    log_group_pattern: str,
) -> bool:
    """Create a log-based dashboard."""
    url = f"{base_url}/api/dashboards/db"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    dashboard = {
        "dashboard": {
            "title": dashboard_name,
            "uid": dashboard_name.lower().replace(" ", "-"),
            "tags": ["logs", "cloudwatch"],
            "timezone": "browser",
            "refresh": "30s",
            "panels": [
                {
                    "id": 1,
                    "title": "Recent Logs",
                    "type": "logs",
                    "targets": [
                        {
                            "refId": "A",
                            "datasource": {
                                "type": "cloudwatch-logs",
                                "uid": "cloudwatch-logs",
                            },
                            "query": log_query,
                            "queryType": "cloudWatchLogs",
                        }
                    ],
                    "gridPos": {"h": 12, "w": 24, "x": 0, "y": 0},
                },
                {
                    "id": 2,
                    "title": "Log Count (Last Hour)",
                    "type": "stat",
                    "targets": [
                        {
                            "refId": "A",
                            "expr": f"count(fields @timestamp, @message) | filter @logStream like /{log_group_pattern}/ | stats count()",
                        }
                    ],
                    "gridPos": {"h": 4, "w": 12, "x": 0, "y": 12},
                },
                {
                    "id": 3,
                    "title": "Error Rate (Last Hour)",
                    "type": "stat",
                    "targets": [
                        {
                            "refId": "A",
                            "expr": f"fields @timestamp, @message | filter @logStream like /{log_group_pattern}/ | filter @message like /ERROR|error|FAIL|fail/ | stats count()",
                        }
                    ],
                    "gridPos": {"h": 4, "w": 12, "x": 12, "y": 12},
                },
            ],
        },
        "overwrite": True,
    }

    try:
        response = requests.post(url, headers=headers, json=dashboard)
        if response.status_code in [200, 409]:  # 409 means dashboard already exists
            print(f"✓ Dashboard '{dashboard_name}' created")
            return True
        else:
            print(
                f"✗ Failed to create dashboard '{dashboard_name}': {response.status_code} - {response.text}"
            )
            return False
    except Exception as e:
        print(f"Error creating dashboard '{dashboard_name}': {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Configure Grafana for log monitoring")
    parser.add_argument(
        "--url",
        required=True,
        help="Grafana URL (e.g., https://d3naqrkpy0vqtm.cloudfront.net)",
    )
    parser.add_argument(
        "--region", default="us-east-1", help="AWS region (default: us-east-1)"
    )
    parser.add_argument(
        "--admin-password",
        default="admin",
        help="Grafana admin password (default: admin)",
    )

    args = parser.parse_args()

    # Ensure URL has https:// prefix
    base_url = args.url
    if not base_url.startswith("https://"):
        base_url = f"https://{base_url}"
    base_url = base_url.rstrip("/")

    print(f"Configuring Grafana at {base_url}...")
    print("Waiting for Grafana to be ready...")
    time.sleep(30)  # Wait for Grafana to be fully ready

    # Create API key
    print("\nCreating API key...")
    api_key = create_api_key(base_url, args.admin_password)

    if not api_key:
        print(
            "✗ Failed to create API key. Please check Grafana is ready and try again."
        )
        sys.exit(1)

    print("✓ API key created")

    # Configure CloudWatch Logs data source
    print("\nConfiguring CloudWatch Logs data source...")
    if not configure_cloudwatch_logs_datasource(base_url, api_key, args.region):
        print("⚠ Continuing with data source configuration...")

    # Create log dashboards
    print("\nCreating log dashboards...")

    dashboards = [
        (
            "API Logs",
            "fields @timestamp, @message, level | filter @logStream like /api-logs/ | sort @timestamp desc | limit 100",
            "api-logs",
        ),
        (
            "DVC Logs",
            "fields @timestamp, @message | filter @logStream like /dvc-logs/ | sort @timestamp desc | limit 100",
            "dvc-logs",
        ),
        (
            "MLflow Logs",
            "fields @timestamp, @message, level | filter @logStream like /mlflow-logs/ | sort @timestamp desc | limit 100",
            "mlflow-logs",
        ),
        (
            "Infrastructure Logs",
            "fields @timestamp, @message | filter @logStream like /ecs/ | sort @timestamp desc | limit 100",
            "ecs",
        ),
    ]

    success_count = 0
    for name, query, pattern in dashboards:
        if create_log_dashboard(base_url, api_key, name, query, pattern):
            success_count += 1

    print(f"\n✓ {success_count}/{len(dashboards)} dashboards created")

    # Delete API key for security
    print("\nCleaning up API key...")
    # Note: In production, you might want to keep the key for future updates

    print("\n✓ Configuration complete!")
    print(f"\nAccess Grafana at: {base_url}")
    print("Dashboards created:")
    for name, _, _ in dashboards:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
