#!/usr/bin/env python3
"""Generate real drift by using extreme data concentration patterns."""

import random
import time
import requests

API_KEY = "hm8K1JkR2BY4-zn1VGsFO-1MP_xp39GjdoUUacfyEvk"
BASE_URL = "https://d1b386spzciemm.cloudfront.net"

# Valid user IDs from the gmf_binary model
VALID_USERS = [138131, 911093, 150877, 103776, 231482, 138427, 431417, 165348, 399592, 390648]
# Valid item IDs from the gmf_binary model
VALID_ITEMS = [430292, 277119, 183411, 457231, 259078, 394974, 150877, 103776, 231482, 138427]

def make_prediction(user_id: int, item_ids: list[int]) -> dict:
    """Make a single prediction request."""
    response = requests.post(
        f"{BASE_URL}/predict",
        headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
        json={"user_id": user_id, "item_ids": item_ids}
    )
    return response.json()

def set_baselines() -> bool:
    """Set monitoring baselines."""
    response = requests.post(
        f"{BASE_URL}/monitoring/baselines",
        headers={"X-API-Key": API_KEY}
    )
    if response.status_code == 200:
        print("✅ Baselines set successfully")
        return True
    else:
        print(f"❌ Failed to set baselines: {response.text}")
        return False

def check_drift() -> dict:
    """Check for drift."""
    response = requests.get(
        f"{BASE_URL}/monitoring/check",
        headers={"X-API-Key": API_KEY}
    )
    return response.json()

def generate_concentrated_baseline(num_requests: int = 200):
    """Generate baseline predictions with extreme concentration on single user/item."""
    print(f"Generating {num_requests} baseline predictions with extreme concentration...")

    # Use single user and single item repeatedly
    user_id = VALID_USERS[0]
    item_ids = [VALID_ITEMS[0]]

    for i in range(num_requests):
        make_prediction(user_id, item_ids)

        if (i + 1) % 40 == 0:
            print(f"  Completed {i + 1}/{num_requests} requests")
        time.sleep(0.02)

    print(f"✅ Generated {num_requests} concentrated baseline predictions")

def generate_diverse_drift(num_requests: int = 200):
    """Generate drift predictions with maximum diversity."""
    print(f"Generating {num_requests} drift predictions with maximum diversity...")

    # Use all different users and items
    for i in range(num_requests):
        user_id = random.choice(VALID_USERS[5:])  # Different users than baseline
        item_ids = random.sample(VALID_ITEMS[5:], k=random.randint(1, 5))  # Different items

        make_prediction(user_id, item_ids)

        if (i + 1) % 40 == 0:
            print(f"  Completed {i + 1}/{num_requests} requests")
        time.sleep(0.02)

    print(f"✅ Generated {num_requests} diverse drift predictions")

def main():
    """Main function to generate real drift through extreme concentration/diversity."""
    print("=== Extreme Drift Generation ===\n")

    # Phase 1: Generate baseline with extreme concentration
    print("Phase 1: Generating baseline with extreme concentration on single user/item")
    generate_concentrated_baseline(num_requests=200)

    # Set baselines
    print("\nSetting monitoring baselines...")
    if not set_baselines():
        print("Failed to set baselines, exiting...")
        return

    time.sleep(2)

    # Phase 2: Generate drift with maximum diversity
    print("\nPhase 2: Generating drift with maximum diversity")
    generate_diverse_drift(num_requests=200)

    # Phase 3: Check for drift
    print("\nPhase 3: Checking for drift")
    drift_results = check_drift()

    print("\nDrift Detection Results:")
    for shift_type, result in drift_results.items():
        status = "🚨 DETECTED" if result.get('has_shift') else "✅ NOT DETECTED"
        print(f"  {shift_type}: {status}")
        print(f"    Message: {result.get('message', 'N/A')}")
        if result.get('has_shift'):
            print(f"    Test Statistic: {result.get('test_statistic', 'N/A')}")
            print(f"    Threshold: {result.get('threshold', 'N/A')}")

    print("\n=== Experiment Complete ===")
    print("Check Grafana dashboard to see drift metrics:")
    print("http://localhost:3000 (admin/admin)")
    print("Dashboard: API Metrics Dashboard")

if __name__ == "__main__":
    main()
