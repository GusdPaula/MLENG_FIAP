#!/usr/bin/env python3
"""Test script for cold start solution."""

import requests
import json

API_KEY = "hm8K1JkR2BY4-zn1VGsFO-1MP_xp39GjdoUUacfyEvk"
BASE_URL = "https://d1b386spzciemm.cloudfront.net"

def test_unknown_user():
    """Test recommendation for unknown user (user cold start)."""
    print("=== Testing Unknown User (User Cold Start) ===")

    # Use a user ID that doesn't exist in the training data
    unknown_user_id = 999999

    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
            json={"user_id": unknown_user_id, "item_ids": [430292, 277119, 183411]}
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Unknown user prediction successful")
            print(f"User ID: {result['user_id']}")
            print(f"Item scores: {result['item_scores']}")
            print(f"Metadata: {result.get('metadata', {})}")

            if result.get('metadata', {}).get('cold_start'):
                print("✅ Cold start fallback was triggered")
            else:
                print("❌ Cold start fallback was NOT triggered")
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Error: {response.text}")

    except Exception as e:
        print(f"❌ Exception occurred: {e}")

def test_unknown_items():
    """Test recommendation with unknown items (item cold start)."""
    print("\n=== Testing Unknown Items (Item Cold Start) ===")

    # Use a known user with unknown items
    known_user_id = 138131
    unknown_item_ids = [999998, 999999, 999997]

    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
            json={"user_id": known_user_id, "item_ids": unknown_item_ids}
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Unknown items prediction successful")
            print(f"User ID: {result['user_id']}")
            print(f"Item scores: {result['item_scores']}")
            print(f"Metadata: {result.get('metadata', {})}")

            if result.get('metadata', {}).get('cold_start'):
                print("✅ Cold start fallback was triggered")
            else:
                print("❌ Cold start fallback was NOT triggered")
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Error: {response.text}")

    except Exception as e:
        print(f"❌ Exception occurred: {e}")

def test_mixed_items():
    """Test recommendation with mix of known and unknown items."""
    print("\n=== Testing Mixed Known/Unknown Items ===")

    known_user_id = 138131
    mixed_items = [430292, 999998, 277119, 999999]  # Mix of known and unknown

    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
            json={"user_id": known_user_id, "item_ids": mixed_items}
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Mixed items prediction successful")
            print(f"User ID: {result['user_id']}")
            print(f"Item scores: {result['item_scores']}")
            print(f"Metadata: {result.get('metadata', {})}")

            # Check if we got scores for all items
            if len(result['item_scores']) == len(mixed_items):
                print("✅ Scores returned for all items")
            else:
                print(f"❌ Only got scores for {len(result['item_scores'])} out of {len(mixed_items)} items")

            if result.get('metadata', {}).get('cold_start'):
                print("✅ Cold start fallback was triggered")
            else:
                print("ℹ️  Cold start fallback was NOT triggered (normal CF for known items)")
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Error: {response.text}")

    except Exception as e:
        print(f"❌ Exception occurred: {e}")

def test_recommend_unknown_user():
    """Test top-k recommendations for unknown user."""
    print("\n=== Testing Top-K Recommendations for Unknown User ===")

    unknown_user_id = 999999

    try:
        response = requests.get(
            f"{BASE_URL}/recommend/{unknown_user_id}?k=10",
            headers={"X-API-Key": API_KEY}
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Unknown user recommendations successful")
            print(f"User ID: {result['user_id']}")
            print(f"Number of recommendations: {len(result.get('recommendations', []))}")
            print(f"Top 3 recommendations: {result.get('recommendations', [])[:3]}")
            print(f"Metadata: {result.get('metadata', {})}")

            if result.get('metadata', {}).get('cold_start'):
                print("✅ Cold start fallback was triggered")
            else:
                print("❌ Cold start fallback was NOT triggered")
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Error: {response.text}")

    except Exception as e:
        print(f"❌ Exception occurred: {e}")

def test_health():
    """Test API health endpoint."""
    print("=== Testing API Health ===")

    try:
        response = requests.get(
            f"{BASE_URL}/health",
            headers={"X-API-Key": API_KEY}
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ API is healthy: {result}")
        else:
            print(f"❌ Health check failed with status {response.status_code}")

    except Exception as e:
        print(f"❌ Exception occurred: {e}")

def main():
    """Run all cold start tests."""
    print("🧪 Cold Start Solution Testing\n")

    # Test API health first
    test_health()

    # Test cold start scenarios
    test_unknown_user()
    test_unknown_items()
    test_mixed_items()
    test_recommend_unknown_user()

    print("\n=== Testing Complete ===")

if __name__ == "__main__":
    main()
