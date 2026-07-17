#!/usr/bin/env python3
"""Generate drift by running experiments with different data patterns."""

import random
import time
import requests
from typing import List

API_KEY = "hm8K1JkR2BY4-zn1VGsFO-1MP_xp39GjdoUUacfyEvk"
BASE_URL = "https://d1b386spzciemm.cloudfront.net"
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

# Valid user IDs from the gmf_binary model
VALID_USERS = [138131, 911093, 150877, 103776, 231482, 138427, 431417, 165348, 399592, 390648]
# Valid item IDs from the gmf_binary model
VALID_ITEMS = [430292, 277119, 183411, 457231, 259078, 394974, 150877, 103776, 231482, 138427]

def make_prediction(user_id: int, item_ids: List[int]) -> dict:
    """Make a single prediction request."""
    response = requests.post(
        f"{BASE_URL}/predict",
        headers=HEADERS,
        json={"user_id": user_id, "item_ids": item_ids}
    )
    return response.json()

def make_recommendation(user_id: int, k: int = 10) -> dict:
    """Make a recommendation request."""
    response = requests.get(
        f"{BASE_URL}/recommend/{user_id}?k={k}",
        headers={"X-API-Key": API_KEY}
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

def get_monitoring_summary() -> dict:
    """Get monitoring summary."""
    response = requests.get(
        f"{BASE_URL}/monitoring/summary",
        headers={"X-API-Key": API_KEY}
    )
    return response.json()

def generate_baseline_data(num_requests: int = 100):
    """Generate baseline prediction data."""
    print(f"Generating {num_requests} baseline requests...")

    for i in range(num_requests):
        user_id = random.choice(VALID_USERS)
        item_ids = random.sample(VALID_ITEMS, k=random.randint(3, 5))

        if random.random() < 0.5:
            make_prediction(user_id, item_ids)
        else:
            make_recommendation(user_id, k=random.randint(5, 10))

        if (i + 1) % 20 == 0:
            print(f"  Completed {i + 1}/{num_requests} requests")
            time.sleep(0.1)  # Small delay to avoid overwhelming the API

    print(f"✅ Generated {num_requests} baseline requests")

def generate_drift_scenario_1(num_requests: int = 50):
    """Generate drift by focusing on specific users (user distribution shift)."""
    print(f"Generating drift scenario 1: User distribution shift ({num_requests} requests)...")

    # Focus heavily on a single user to create distribution shift
    focused_user = VALID_USERS[0]

    for i in range(num_requests):
        # 80% of requests use the focused user, 20% use others
        if random.random() < 0.8:
            user_id = focused_user
        else:
            user_id = random.choice(VALID_USERS[1:])

        item_ids = random.sample(VALID_ITEMS, k=random.randint(3, 5))

        make_prediction(user_id, item_ids)

        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{num_requests} drift requests")
            time.sleep(0.05)

    print(f"✅ Generated {num_requests} drift requests (user distribution shift)")

def generate_drift_scenario_2(num_requests: int = 50):
    """Generate drift by focusing on specific items (item distribution shift)."""
    print(f"Generating drift scenario 2: Item distribution shift ({num_requests} requests)...")

    # Focus heavily on specific items to create distribution shift
    focused_items = VALID_ITEMS[:3]

    for i in range(num_requests):
        user_id = random.choice(VALID_USERS)

        # 80% of requests use focused items, 20% use others
        if random.random() < 0.8:
            item_ids = random.sample(focused_items, k=len(focused_items))
        else:
            item_ids = random.sample(VALID_ITEMS[3:], k=random.randint(3, 5))

        make_prediction(user_id, item_ids)

        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{num_requests} drift requests")
            time.sleep(0.05)

    print(f"✅ Generated {num_requests} drift requests (item distribution shift)")

def generate_extreme_drift(num_requests: int = 200):
    """Generate extreme drift by using completely different patterns."""
    print(f"Generating extreme drift scenario ({num_requests} requests)...")

    # Use completely different users and items than baseline
    extreme_users = VALID_USERS[5:]  # Use different users than baseline
    extreme_items = VALID_ITEMS[5:]  # Use different items than baseline

    for i in range(num_requests):
        user_id = random.choice(extreme_users)

        # Always use the same item set to create extreme concentration
        item_ids = extreme_items[:3]  # Always use same 3 items

        make_prediction(user_id, item_ids)

        if (i + 1) % 25 == 0:
            print(f"  Completed {i + 1}/{num_requests} extreme drift requests")
            time.sleep(0.02)

    print(f"✅ Generated {num_requests} extreme drift requests")

def main():
    """Main function to generate drift."""
    print("=== Drift Generation Experiment ===\n")

    # Phase 1: Generate baseline data
    print("Phase 1: Generating baseline data")
    generate_baseline_data(num_requests=200)

    # Set baselines
    print("\nSetting monitoring baselines...")
    if not set_baselines():
        print("Failed to set baselines, exiting...")
        return

    time.sleep(2)

    # Phase 2: Generate extreme drift scenarios
    print("\nPhase 2: Generating extreme drift scenarios")

    # Extreme drift scenario
    generate_extreme_drift(num_requests=300)
    time.sleep(1)

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

    # Phase 4: Get monitoring summary
    print("\nPhase 4: Monitoring Summary")
    summary = get_monitoring_summary()
    print(f"  Total predictions: {summary.get('performance_stats', {}).get('count', 'N/A')}")
    print(f"  Mean score: {summary.get('performance_stats', {}).get('mean', 'N/A'):.4f}")
    print(f"  Std score: {summary.get('performance_stats', {}).get('std', 'N/A'):.4f}")
    print(f"  Has baseline: {summary.get('has_baseline', 'N/A')}")

    print("\n=== Experiment Complete ===")
    print("Check Grafana dashboard to see drift metrics:")
    print("http://localhost:3000 (admin/admin)")
    print("Dashboard: API Metrics Dashboard")

if __name__ == "__main__":
    main()
