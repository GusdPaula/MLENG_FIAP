#!/usr/bin/env python3
"""Train item feature extractor from interaction data for cold start solution."""

import os
import pickle
import sys

import pandas as pd

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ecommerce_recommender', 'src'))

from recommender.features.item_features import (
    ContentBasedRecommender,
    ItemFeatureExtractor,
)


def main():
    """Train and save item feature extractor."""
    print("=== Training Item Feature Extractor ===\n")

    # Load interaction data
    # Try processed first, fall back to raw events
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)

    interactions_path = os.path.join(base_dir, 'ecommerce_recommender/data/processed/interactions.csv')
    raw_events_path = os.path.join(base_dir, 'ecommerce_recommender/data/raw/events.csv')

    if os.path.exists(interactions_path):
        print(f"Loading interactions from {interactions_path}")
        interactions = pd.read_csv(interactions_path)
    elif os.path.exists(raw_events_path):
        print(f"Loading raw events from {raw_events_path}")
        interactions = pd.read_csv(raw_events_path)
        # Add weight column if not present
        if 'weight' not in interactions.columns:
            event_weights = {"view": 1, "addtocart": 2, "transaction": 3}
            interactions["weight"] = interactions["event"].map(event_weights).fillna(1.0)
    else:
        print("❌ No interaction data found")
        print(f"Tried: {interactions_path} and {raw_events_path}")
        return

    print(f"✅ Loaded {len(interactions)} interactions")

    # Check required columns
    required_columns = ['visitorid', 'itemid']
    missing_columns = [col for col in required_columns if col not in interactions.columns]
    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        print(f"Available columns: {interactions.columns.tolist()}")
        return

    # Initialize feature extractor
    print("\nInitializing feature extractor...")
    feature_extractor = ItemFeatureExtractor()

    # Fit on interaction data
    print("Training feature extractor on interaction data...")
    feature_extractor.fit_from_interactions(interactions)
    print(f"✅ Feature extractor trained on {len(feature_extractor.item_features)} items")

    # Create content-based recommender
    print("\nCreating content-based recommender...")
    content_recommender = ContentBasedRecommender(feature_extractor)
    print("✅ Content-based recommender created")

    # Save the artifacts
    output_dir = os.path.join(base_dir, 'ecommerce_recommender/models')
    os.makedirs(output_dir, exist_ok=True)

    feature_extractor_path = os.path.join(output_dir, 'item_feature_extractor.pkl')
    content_recommender_path = os.path.join(output_dir, 'content_recommender.pkl')

    print(f"\nSaving feature extractor to {feature_extractor_path}")
    with open(feature_extractor_path, 'wb') as f:
        pickle.dump(feature_extractor, f)
    print("✅ Feature extractor saved")

    print(f"Saving content recommender to {content_recommender_path}")
    with open(content_recommender_path, 'wb') as f:
        pickle.dump(content_recommender, f)
    print("✅ Content recommender saved")

    # Print feature statistics
    print("\n=== Feature Statistics ===")
    print(f"Total items: {len(feature_extractor.item_features)}")
    if feature_extractor.item_features:
        sample_item = list(feature_extractor.item_features.keys())[0]
        sample_features = feature_extractor.item_features[sample_item]
        print(f"Sample item: {sample_item}")
        print(f"Feature dimensions: {len(sample_features)}")
        print(f"Sample features: {sample_features}")

    print("\n=== Training Complete ===")
    print("Artifacts saved successfully. Ready for deployment.")

if __name__ == "__main__":
    main()
