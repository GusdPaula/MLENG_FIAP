#!/usr/bin/env python3
"""Update existing Grafana Prometheus data source configuration."""

import requests
from requests.auth import HTTPBasicAuth

GRAFANA_URL = "http://localhost:3000"
GRAFANA_USER = "admin"
GRAFANA_PASSWORD = "admin"

def get_datasources():
    """Get all data sources."""
    response = requests.get(
        f"{GRAFANA_URL}/api/datasources",
        auth=HTTPBasicAuth(GRAFANA_USER, GRAFANA_PASSWORD)
    )
    return response.json()

def update_prometheus_datasource():
    """Update Prometheus data source with correct URL."""
    datasources = get_datasources()

    for ds in datasources:
        if ds['name'] == 'Prometheus' and ds['type'] == 'prometheus':
            print(f"Found Prometheus datasource: {ds['name']}")
            print(f"Current URL: {ds['url']}")

            # Update the datasource
            ds_id = ds['id']
            response = requests.put(
                f"{GRAFANA_URL}/api/datasources/{ds_id}",
                auth=HTTPBasicAuth(GRAFANA_USER, GRAFANA_PASSWORD),
                json={
                    "name": "Prometheus",
                    "type": "prometheus",
                    "access": "proxy",
                    "url": "http://prometheus:9090",
                    "isDefault": True,
                    "jsonData": {
                        "timeInterval": "15s"
                    }
                }
            )

            if response.status_code == 200:
                print("✅ Prometheus datasource updated successfully")
                print("New URL: http://prometheus:9090")
                return True
            else:
                print(f"❌ Failed to update datasource: {response.text}")
                return False

    print("❌ Prometheus datasource not found")
    return False

if __name__ == "__main__":
    update_prometheus_datasource()
