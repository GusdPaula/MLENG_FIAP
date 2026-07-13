#!/usr/bin/env python3
"""Upload Grafana dashboards to Grafana instance."""

import argparse
import json
import sys

import requests


def upload_dashboard(
    base_url: str, dashboard_file: str, api_key: str | None = None
) -> bool:
    """Upload a dashboard to Grafana.

    Args:
        base_url: Grafana base URL.
        dashboard_file: Path to dashboard JSON file.
        api_key: Grafana API key (optional, will use basic auth if not provided).

    Returns:
        True if successful, False otherwise.
    """
    with open(dashboard_file, "r") as f:
        dashboard_data = json.load(f)

    url = f"{base_url}/api/dashboards/db"
    headers = {"Content-Type": "application/json"}

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        # Use basic auth (will prompt for password if needed)
        import getpass

        username = "admin"
        password = getpass.getpass("Enter Grafana admin password: ")
        auth = (username, password)

    try:
        if api_key:
            response = requests.post(url, headers=headers, json=dashboard_data)
        else:
            response = requests.post(url, headers=headers, json=dashboard_data, auth=auth)

        if response.status_code in [200, 409]:  # 409 = dashboard already exists
            print(f"✓ Dashboard uploaded successfully: {dashboard_file}")
            return True
        else:
            print(f"✗ Failed to upload dashboard: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error uploading dashboard: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload Grafana dashboards")
    parser.add_argument(
        "--url",
        required=True,
        help="Grafana URL (e.g., https://d3naqrkpy0vqtm.cloudfront.net)",
    )
    parser.add_argument(
        "--dashboard",
        required=True,
        help="Path to dashboard JSON file",
    )
    parser.add_argument(
        "--api-key",
        help="Grafana API key (optional, will use basic auth if not provided)",
    )
    parser.add_argument(
        "--password",
        default="GrafanaAdmin123!",
        help="Grafana admin password (default: GrafanaAdmin123!)",
    )

    args = parser.parse_args()

    # Ensure URL has https:// prefix
    base_url = args.url
    if not base_url.startswith("https://"):
        base_url = f"https://{base_url}"
    base_url = base_url.rstrip("/")

    # Try with basic auth first
    auth = ("admin", args.password)
    url = f"{base_url}/api/dashboards/db"
    headers = {"Content-Type": "application/json"}

    with open(args.dashboard, "r") as f:
        dashboard_data = json.load(f)

    try:
        response = requests.post(url, headers=headers, json=dashboard_data, auth=auth)
        if response.status_code in [200, 409]:
            print(f"✓ Dashboard uploaded successfully: {args.dashboard}")
            sys.exit(0)
        else:
            print(f"✗ Failed to upload dashboard: {response.status_code} - {response.text}")
            sys.exit(1)
    except Exception as e:
        print(f"✗ Error uploading dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
