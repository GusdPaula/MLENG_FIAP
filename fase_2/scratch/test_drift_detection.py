#!/usr/bin/env python3
"""Script to test drift detection in the recommendation API."""

import requests
import time
import json

API_BASE_URL = "http://localhost:8000"
API_KEY = "default-api-key-change-in-production"
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}


def health_check():
    """Check if API is healthy."""
    response = requests.get(f"{API_BASE_URL}/health", headers=HEADERS)
    print(f"Health check: {response.status_code} - {response.json()}")
    return response.status_code == 200


def make_prediction(user_id, item_ids):
    """Make a prediction request."""
    payload = {"user_id": user_id, "item_ids": item_ids}
    response = requests.post(f"{API_BASE_URL}/predict", headers=HEADERS, json=payload)
    return response


def make_recommendation(user_id, k=10):
    """Make a recommendation request."""
    response = requests.get(f"{API_BASE_URL}/recommend/{user_id}?k={k}", headers=HEADERS)
    return response


def set_baselines():
    """Set monitoring baselines."""
    response = requests.post(f"{API_BASE_URL}/monitoring/baselines", headers=HEADERS)
    print(f"Set baselines: {response.status_code} - {response.json()}")
    return response.status_code == 200


def get_monitoring_summary():
    """Get monitoring summary."""
    response = requests.get(f"{API_BASE_URL}/monitoring/summary", headers=HEADERS)
    print(f"Monitoring summary: {response.status_code}")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    return response.json()


def check_shifts():
    """Check for drift and shifts."""
    response = requests.get(f"{API_BASE_URL}/monitoring/check")
    print(f"Check shifts: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(json.dumps(result, indent=2))
        return result
    return None


def test_drift_detection():
    """Test drift detection workflow."""
    print("=" * 60)
    print("DRIFT DETECTION TEST")
    print("=" * 60)

    # 1. Health check
    print("\n1. Health Check")
    if not health_check():
        print("❌ API is not healthy. Exiting.")
        return
    print("✅ API is healthy")

    # 2. Make normal predictions to establish baseline
    print("\n2. Making 20 normal predictions (user_id=138131, items=[430292,277119,183411,457231,259078])")
    for i in range(20):
        response = make_prediction(user_id=138131, item_ids=[430292, 277119, 183411, 457231, 259078])
        if response.status_code == 200:
            print(f"  Prediction {i+1}/20: ✅")
        else:
            print(f"  Prediction {i+1}/20: ❌ {response.status_code}")
            if response.status_code == 400:
                print(f"    Error: {response.text}")
        time.sleep(0.1)  # Small delay between requests

    # 3. Set baselines
    print("\n3. Setting monitoring baselines")
    if not set_baselines():
        print("❌ Failed to set baselines")
        return
    print("✅ Baselines set successfully")

    # 4. Check monitoring summary
    print("\n4. Monitoring summary after setting baseline")
    get_monitoring_summary()

    # 5. Check for shifts (should be none)
    print("\n5. Checking for shifts (should be none)")
    check_shifts()

    # 6. Make predictions with different patterns to create drift
    print("\n6. Making 100 recommendations (scoring ALL items) to force drift")
    # Use different valid user IDs from processed data - repeat to create more volume
    drift_users = [911093, 1161163, 457926, 404403, 286616, 1235292, 434418, 535937, 1093035, 1236753,
                  469194, 1020169, 138131, 911093, 1161163, 457926, 404403, 286616, 1235292, 434418,
                  535937, 1093035, 1236753, 469194, 1020169, 138131, 911093, 1161163, 457926, 404403] * 4

    for i in range(100):
        user_id = drift_users[i]
        # Use recommendations instead of predictions - this scores ALL items, creating different distribution
        response = make_recommendation(user_id=user_id, k=50)
        if response.status_code == 200:
            if (i+1) % 10 == 0:
                print(f"  Drift prediction {i+1}/100: ✅ (user_id={user_id})")
        else:
            print(f"  Drift prediction {i+1}/100: ❌ {response.status_code}")
            if response.status_code == 400:
                print(f"    Error: {response.text}")
        time.sleep(0.05)

    # 7. Check for shifts (should detect drift now)
    print("\n7. Checking for shifts (should detect drift)")
    result = check_shifts()

    # 8. Final monitoring summary
    print("\n8. Final monitoring summary")
    get_monitoring_summary()

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    # Summary
    if result:
        print("\n📊 DRIFT DETECTION RESULTS:")
        for shift_type, shift_result in result.items():
            status = "🔴 DETECTED" if shift_result.get("has_shift") else "🟢 NOT DETECTED"
            print(f"  {shift_type}: {status}")
            print(f"    Message: {shift_result.get('message')}")


if __name__ == "__main__":
    try:
        test_drift_detection()
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the API is running on http://localhost:8000")
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
