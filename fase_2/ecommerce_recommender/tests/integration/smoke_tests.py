#!/usr/bin/env python3
"""Smoke tests for the deployed API.

This script runs smoke tests against the deployed AWS API using the API key
from the .env file.
"""

import os
import sys
from typing import Dict

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL", "https://d1b386spzciemm.cloudfront.net")

# Test data for gmf_binary model
TEST_USER_ID = 138131
TEST_ITEM_IDS = [430292, 277119, 183411, 457231, 259078]


def print_result(test_name: str, success: bool, message: str = "") -> None:
    """Print test result with formatting.

    Args:
        test_name: Name of the test
        success: Whether the test passed
        message: Additional message
    """
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} - {test_name}")
    if message:
        print(f"     {message}")


def test_health_check() -> bool:
    """Test the health check endpoint.

    Returns:
        True if test passed, False otherwise
    """
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        success = response.status_code == 200
        message = (
            f"Status: {response.status_code}"
            if success
            else f"Status: {response.status_code}"
        )
        print_result("Health Check", success, message)
        return success
    except Exception as e:
        print_result("Health Check", False, str(e))
        return False


def test_model_info() -> bool:
    """Test the model info endpoint.

    Returns:
        True if test passed, False otherwise
    """
    try:
        headers = {"X-API-Key": API_KEY}
        response = requests.get(f"{API_URL}/model/info", headers=headers, timeout=10)
        success = response.status_code == 200
        if success:
            data = response.json()
            message = f"Model loaded: {data.get('model_path', 'N/A')}"
        else:
            message = f"Status: {response.status_code}"
        print_result("Model Info", success, message)
        return success
    except Exception as e:
        print_result("Model Info", False, str(e))
        return False


def test_single_prediction() -> bool:
    """Test the single prediction endpoint.

    Returns:
        True if test passed, False otherwise
    """
    try:
        headers = {"X-API-Key": API_KEY}
        payload = {"user_id": TEST_USER_ID, "item_ids": TEST_ITEM_IDS}
        response = requests.post(
            f"{API_URL}/predict", headers=headers, json=payload, timeout=30
        )
        success = response.status_code == 200
        if success:
            data = response.json()
            num_scores = len(data.get("item_scores", {}))
            message = f"Received {num_scores} item scores"
        else:
            message = f"Status: {response.status_code}"
        print_result("Single Prediction", success, message)
        return success
    except Exception as e:
        print_result("Single Prediction", False, str(e))
        return False


def test_recommendations() -> bool:
    """Test the recommendations endpoint.

    Returns:
        True if test passed, False otherwise
    """
    try:
        headers = {"X-API-Key": API_KEY}
        response = requests.get(
            f"{API_URL}/recommend/{TEST_USER_ID}?k=10", headers=headers, timeout=30
        )
        success = response.status_code == 200
        if success:
            data = response.json()
            num_recs = len(data.get("recommendations", []))
            message = f"Received {num_recs} recommendations"
        else:
            message = f"Status: {response.status_code}"
        print_result("Recommendations", success, message)
        return success
    except Exception as e:
        print_result("Recommendations", False, str(e))
        return False


def test_batch_prediction() -> bool:
    """Test the batch prediction endpoint.

    Returns:
        True if test passed, False otherwise
    """
    try:
        headers = {"X-API-Key": API_KEY}
        payload = {
            "user_item_pairs": [
                [TEST_USER_ID, [430292, 277119, 183411]],
                [911093, [457231, 259078, 183087]],
            ],
            "k": None,
        }
        response = requests.post(
            f"{API_URL}/predict/batch", headers=headers, json=payload, timeout=30
        )
        success = response.status_code == 200
        if success:
            data = response.json()
            num_predictions = len(data.get("predictions", []))
            message = f"Received {num_predictions} predictions"
        else:
            message = f"Status: {response.status_code}"
        print_result("Batch Prediction", success, message)
        return success
    except Exception as e:
        print_result("Batch Prediction", False, str(e))
        return False


def test_monitoring_check() -> bool:
    """Test the monitoring check endpoint.

    Returns:
        True if test passed, False otherwise
    """
    try:
        headers = {"X-API-Key": API_KEY}
        response = requests.get(
            f"{API_URL}/monitoring/check", headers=headers, timeout=10
        )
        success = response.status_code == 200
        if success:
            data = response.json()
            data_shift = data.get("data_shift", {}).get("has_shift", False)
            perf_drift = data.get("performance_drift", {}).get("has_shift", False)
            message = f"Data shift: {data_shift}, Performance drift: {perf_drift}"
        else:
            message = f"Status: {response.status_code}"
        print_result("Monitoring Check", success, message)
        return success
    except Exception as e:
        print_result("Monitoring Check", False, str(e))
        return False


def test_authentication() -> bool:
    """Test API authentication.

    Returns:
        True if test passed, False otherwise
    """
    try:
        # Test without API key on model/info (requires auth)
        response = requests.get(f"{API_URL}/model/info", timeout=10)
        success_without_key = response.status_code == 401
        print_result(
            "Auth - No API Key", success_without_key, f"Status: {response.status_code}"
        )

        # Test with invalid API key on model/info
        response = requests.get(
            f"{API_URL}/model/info", headers={"X-API-Key": "invalid-key"}, timeout=10
        )
        success_with_invalid_key = response.status_code == 403
        print_result(
            "Auth - Invalid API Key",
            success_with_invalid_key,
            f"Status: {response.status_code}",
        )

        # Test with valid API key on model/info
        response = requests.get(
            f"{API_URL}/model/info", headers={"X-API-Key": API_KEY}, timeout=10
        )
        success_with_valid_key = response.status_code == 200
        print_result(
            "Auth - Valid API Key",
            success_with_valid_key,
            f"Status: {response.status_code}",
        )

        return (
            success_without_key and success_with_invalid_key and success_with_valid_key
        )
    except Exception as e:
        print_result("Authentication", False, str(e))
        return False


def run_smoke_tests() -> Dict[str, bool]:
    """Run all smoke tests.

    Returns:
        Dictionary mapping test names to results
    """
    print("=" * 60)
    print("SMOKE TESTS FOR DEPLOYED API")
    print("=" * 60)
    print(f"API URL: {API_URL}")
    print(
        f"API Key: {'*' * (len(API_KEY) - 4) + API_KEY[-4:] if API_KEY else 'NOT SET'}"
    )
    print("=" * 60)
    print()

    if not API_KEY:
        print("❌ ERROR: API_KEY not set in .env file")
        print("Please set API_KEY in your .env file")
        sys.exit(1)

    results: Dict[str, bool] = {}

    # Run tests
    results["health_check"] = test_health_check()
    results["authentication"] = test_authentication()
    results["model_info"] = test_model_info()
    results["single_prediction"] = test_single_prediction()
    results["recommendations"] = test_recommendations()
    results["batch_prediction"] = test_batch_prediction()
    results["monitoring_check"] = test_monitoring_check()

    # Summary
    print()
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")

    if passed == total:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_smoke_tests())
