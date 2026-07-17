#!/usr/bin/env python3
"""Configure Grafana with Prometheus data source and API dashboard."""

import time

import requests
from requests.auth import HTTPBasicAuth

GRAFANA_URL = "http://localhost:3000"
GRAFANA_USER = "admin"
GRAFANA_PASSWORD = "admin"
PROMETHEUS_URL = "http://172.31.84.126:9090"

def wait_for_grafana():
    """Wait for Grafana to be ready."""
    print("Waiting for Grafana to be ready...")
    for i in range(30):
        try:
            response = requests.get(f"{GRAFANA_URL}/api/health")
            if response.status_code == 200:
                print("Grafana is ready")
                return True
        except requests.exceptions.ConnectionError:
            pass
        print(f"Waiting... ({i+1}/30)")
        time.sleep(2)
    print("Grafana did not become ready")
    return False

def create_api_key():
    """Create an API key for Grafana."""
    print("Creating API key...")
    response = requests.post(
        f"{GRAFANA_URL}/api/auth/keys",
        auth=HTTPBasicAuth(GRAFANA_USER, GRAFANA_PASSWORD),
        json={"name": "setup-key", "role": "Admin", "secondsToLive": 3600}
    )
    if response.status_code == 200:
        api_key = response.json()["key"]
        print("API key created successfully")
        return api_key
    else:
        print(f"Failed to create API key: {response.text}")
        print("Using basic auth instead...")
        return None

def add_prometheus_datasource(api_key):
    """Add Prometheus as a data source."""
    print("Adding Prometheus data source...")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        auth = None
    else:
        auth = HTTPBasicAuth(GRAFANA_USER, GRAFANA_PASSWORD)

    response = requests.post(
        f"{GRAFANA_URL}/api/datasources",
        headers=headers,
        auth=auth,
        json={
            "name": "Prometheus",
            "type": "prometheus",
            "access": "proxy",
            "url": PROMETHEUS_URL,
            "isDefault": True,
            "jsonData": {
                "timeInterval": "15s"
            }
        }
    )
    if response.status_code == 200:
        print("Prometheus data source added successfully")
        return True
    else:
        print(f"Failed to add Prometheus data source: {response.text}")
        return False

def create_dashboard(api_key):
    """Create API metrics dashboard."""
    print("Creating API dashboard...")

    dashboard = {
        "dashboard": {
            "title": "API Metrics Dashboard",
            "panels": [
                {
                    "title": "Prediction Rate",
                    "targets": [
                        {
                            "expr": "rate(predictions_total[5m])",
                            "legendFormat": "{{predictor_type}} - {{model_version}}"
                        }
                    ],
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                },
                {
                    "title": "API Request Rate",
                    "targets": [
                        {
                            "expr": "rate(api_requests_total[5m])",
                            "legendFormat": "{{endpoint}} - {{method}} - {{status}}"
                        },
                        {
                            "expr": "api_requests_total",
                            "legendFormat": "{{endpoint}} - {{method}} - {{status}} (total)"
                        }
                    ],
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                },
                {
                    "title": "Prediction Duration",
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.95, rate(prediction_duration_seconds_bucket[5m]))",
                            "legendFormat": "p95"
                        },
                        {
                            "expr": "histogram_quantile(0.50, rate(prediction_duration_seconds_bucket[5m]))",
                            "legendFormat": "p50"
                        }
                    ],
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
                },
                {
                    "title": "API Request Duration",
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m]))",
                            "legendFormat": "p95 - {{endpoint}}"
                        }
                    ],
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
                },
                {
                    "title": "Total Predictions",
                    "targets": [
                        {
                            "expr": "predictions_total",
                            "legendFormat": "{{predictor_type}}"
                        }
                    ],
                    "type": "stat",
                    "gridPos": {"h": 4, "w": 6, "x": 0, "y": 16}
                },
                {
                    "title": "Total API Requests",
                    "targets": [
                        {
                            "expr": "sum(api_requests_total)",
                            "legendFormat": "Total"
                        },
                        {
                            "expr": "sum(api_requests_total{status=\"400\"})",
                            "legendFormat": "Errors"
                        }
                    ],
                    "type": "stat",
                    "gridPos": {"h": 4, "w": 6, "x": 6, "y": 16}
                },
                {
                    "title": "Error Rate",
                    "targets": [
                        {
                            "expr": "rate(errors_total[5m])",
                            "legendFormat": "{{error_type}}"
                        },
                        {
                            "expr": "errors_total",
                            "legendFormat": "{{error_type}} (total)"
                        }
                    ],
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16}
                },
                {
                    "title": "Active Users",
                    "targets": [
                        {
                            "expr": "active_users",
                            "legendFormat": "Active Users"
                        }
                    ],
                    "type": "stat",
                    "gridPos": {"h": 4, "w": 6, "x": 0, "y": 20}
                },
                {
                    "title": "Active Items",
                    "targets": [
                        {
                            "expr": "active_items",
                            "legendFormat": "Active Items"
                        }
                    ],
                    "type": "stat",
                    "gridPos": {"h": 4, "w": 6, "x": 6, "y": 20}
                }
            ],
            "refresh": "5s",
            "timezone": "browser"
        },
        "overwrite": True,
        "message": "API Metrics Dashboard created via script"
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        auth = None
    else:
        auth = HTTPBasicAuth(GRAFANA_USER, GRAFANA_PASSWORD)

    response = requests.post(
        f"{GRAFANA_URL}/api/dashboards/db",
        headers=headers,
        auth=auth,
        json=dashboard
    )

    if response.status_code == 200:
        print("Dashboard created successfully")
        return True
    else:
        print(f"Failed to create dashboard: {response.text}")
        return False

def main():
    """Main configuration function."""
    if not wait_for_grafana():
        return

    api_key = create_api_key()

    # Try to add datasource, but continue if it already exists
    add_prometheus_datasource(api_key)

    # Always try to create/update the dashboard
    if create_dashboard(api_key):
        print("\n✅ Grafana dashboard updated successfully!")
        print(f"Access Grafana at: {GRAFANA_URL}")
        print(f"Login: {GRAFANA_USER}/{GRAFANA_PASSWORD}")
    else:
        print("❌ Failed to update Grafana dashboard")

if __name__ == "__main__":
    main()
