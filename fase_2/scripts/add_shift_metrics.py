#!/usr/bin/env python3
"""Add shift/drift metrics to existing Grafana dashboard."""

import json
import requests
from requests.auth import HTTPBasicAuth

GRAFANA_URL = "http://localhost:3000"
GRAFANA_USER = "admin"
GRAFANA_PASSWORD = "admin"

def get_dashboard_by_uid(uid):
    """Get dashboard by UID."""
    response = requests.get(
        f"{GRAFANA_URL}/api/dashboards/uid/{uid}",
        auth=HTTPBasicAuth(GRAFANA_USER, GRAFANA_PASSWORD)
    )
    if response.status_code == 200:
        return response.json()
    return None

def get_all_dashboards():
    """Get all dashboards."""
    response = requests.get(
        f"{GRAFANA_URL}/api/search",
        auth=HTTPBasicAuth(GRAFANA_USER, GRAFANA_PASSWORD)
    )
    return response.json()

def update_dashboard_with_shift_metrics():
    """Add shift metrics panels to the API dashboard."""
    # Find the API Metrics Dashboard
    dashboards = get_all_dashboards()
    api_dashboard = None
    dashboard_uid = None

    for db in dashboards:
        if 'API Metrics Dashboard' in db.get('title', ''):
            api_dashboard = db
            dashboard_uid = db.get('uid')
            break

    if not api_dashboard:
        print("API Metrics Dashboard not found")
        return False

    print(f"Found dashboard: {api_dashboard['title']} (UID: {dashboard_uid})")

    # Get the full dashboard
    dashboard_data = get_dashboard_by_uid(dashboard_uid)
    if not dashboard_data:
        print("Failed to get dashboard data")
        return False

    dashboard = dashboard_data['dashboard']

    # Add shift metrics panels
    shift_panels = [
        {
            "title": "Drift Alert Rate",
            "targets": [
                {
                    "expr": "rate(drift_alerts_total[5m])",
                    "legendFormat": "{{drift_type}} - {{severity}}"
                },
                {
                    "expr": "drift_alerts_total",
                    "legendFormat": "{{drift_type}} - {{severity}} (total)"
                }
            ],
            "type": "graph",
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 24}
        },
        {
            "title": "Drift Score",
            "targets": [
                {
                    "expr": "drift_score",
                    "legendFormat": "{{drift_type}}"
                }
            ],
            "type": "graph",
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 24}
        },
        {
            "title": "Total Drift Alerts",
            "targets": [
                {
                    "expr": "sum(drift_alerts_total)",
                    "legendFormat": "Total Alerts"
                },
                {
                    "expr": "sum(drift_alerts_total{severity=\"high\"})",
                    "legendFormat": "High Severity"
                },
                {
                    "expr": "sum(drift_alerts_total{severity=\"medium\"})",
                    "legendFormat": "Medium Severity"
                }
            ],
            "type": "stat",
            "gridPos": {"h": 4, "w": 12, "x": 0, "y": 32}
        },
        {
            "title": "Drift by Type",
            "targets": [
                {
                    "expr": "drift_alerts_total",
                    "legendFormat": "{{drift_type}}"
                }
            ],
            "type": "piechart",
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 32}
        }
    ]

    # Add panels to dashboard
    for panel in shift_panels:
        dashboard['panels'].append(panel)

    # Update dashboard
    payload = {
        "dashboard": dashboard,
        "overwrite": True,
        "message": "Added shift/drift metrics panels"
    }

    response = requests.post(
        f"{GRAFANA_URL}/api/dashboards/db",
        headers={"Content-Type": "application/json"},
        auth=HTTPBasicAuth(GRAFANA_USER, GRAFANA_PASSWORD),
        json=payload
    )

    if response.status_code == 200:
        print("✅ Shift metrics panels added successfully")
        return True
    else:
        print(f"❌ Failed to update dashboard: {response.text}")
        return False

if __name__ == "__main__":
    update_dashboard_with_shift_metrics()
