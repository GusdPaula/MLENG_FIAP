#!/usr/bin/env python3
"""
Export model shift metrics from API to CloudWatch.

This script periodically polls the API monitoring endpoints and exports
the metrics to AWS CloudWatch for visualization in Grafana dashboards.
"""

import os
import sys
import time
from typing import Any

import requests
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ecommerce_recommender", "api", "services"))

from cloudwatch_exporter import CloudWatchMetricsExporter

# Load environment variables
load_dotenv()

# Configuration
API_URL = os.getenv("API_URL", "https://d1b386spzciemm.cloudfront.net")
API_KEY = os.getenv("API_KEY")
REGION = os.getenv("AWS_REGION", "us-east-1")
EXPORT_INTERVAL_SECONDS = int(os.getenv("METRIC_EXPORT_INTERVAL", "60"))


def get_monitoring_summary(api_url: str, api_key: str) -> dict[str, Any]:
    """Get monitoring summary from API.

    Args:
        api_url: Base URL of the API.
        api_key: API key for authentication.

    Returns:
        Dictionary with monitoring summary.
    """
    headers = {"X-API-Key": api_key}
    try:
        response = requests.get(f"{api_url}/monitoring/summary", headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Failed to get monitoring summary: {e}")
        return {}


def get_shift_check(api_url: str, api_key: str) -> dict[str, Any]:
    """Get shift check results from API.

    Args:
        api_url: Base URL of the API.
        api_key: API key for authentication.

    Returns:
        Dictionary with shift check results.
    """
    headers = {"X-API-Key": api_key}
    try:
        response = requests.get(f"{api_url}/monitoring/check", headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Failed to get shift check: {e}")
        return {}


def export_metrics():
    """Export metrics from API to CloudWatch."""
    if not API_KEY:
        print("ERROR: API_KEY not set in environment variables")
        sys.exit(1)

    exporter = CloudWatchMetricsExporter(region=REGION)

    print(f"Starting metrics export from {API_URL}")
    print(f"Export interval: {EXPORT_INTERVAL_SECONDS} seconds")
    print("Press Ctrl+C to stop\n")

    while True:
        try:
            # Get monitoring summary
            summary = get_monitoring_summary(API_URL, API_KEY)
            if summary and "performance_stats" in summary:
                stats = summary["performance_stats"]
                print(f"Exporting performance stats: {stats}")
                exporter.export_performance_stats(stats)

            # Get shift check results
            shift_results = get_shift_check(API_URL, API_KEY)
            if shift_results:
                if "data_shift" in shift_results:
                    print(f"Exporting data shift metrics")
                    exporter.export_shift_detection_result(
                        shift_results["data_shift"], "DataShift"
                    )
                if "performance_drift" in shift_results:
                    print(f"Exporting performance drift metrics")
                    exporter.export_shift_detection_result(
                        shift_results["performance_drift"], "ModelDrift"
                    )

            print(f"Metrics exported successfully at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        except Exception as e:
            print(f"Error during metrics export: {e}\n")

        time.sleep(EXPORT_INTERVAL_SECONDS)


if __name__ == "__main__":
    export_metrics()
