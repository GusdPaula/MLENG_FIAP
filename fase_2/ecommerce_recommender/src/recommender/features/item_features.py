"""Content-based recommendation features for cold start problem solving."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ItemFeatureExtractor:
    """Extract and manage item features for content-based recommendations.

    This class handles feature extraction from item metadata to enable
    recommendations for new items (item cold start problem).
    """

    def __init__(self) -> None:
        """Initialize the item feature extractor."""
        self.item_features: Dict[int, np.ndarray] = {}
        self.feature_matrix: Optional[np.ndarray] = None
        self.item_ids: List[int] = []
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None

    def fit_from_interactions(
        self,
        interactions: pd.DataFrame,
        item_metadata: Optional[pd.DataFrame] = None,
    ) -> None:
        """Extract features from interaction data and optional item metadata.

        Args:
            interactions: DataFrame with user-item interactions.
            item_metadata: Optional DataFrame with item attributes (category, price, etc).
        """
        # Extract basic statistical features from interactions
        item_stats = self._extract_interaction_features(interactions)

        # If metadata is available, enhance features
        if item_metadata is not None:
            self._enhance_with_metadata(item_stats, item_metadata)
        else:
            self.item_features = {
                item_id: np.array([stats['popularity'], stats['avg_weight']])
                for item_id, stats in item_stats.items()
            }

        # Create feature matrix for similarity computation
        self._create_feature_matrix()

    def _extract_interaction_features(self, interactions: pd.DataFrame) -> Dict[int, Dict]:
        """Extract statistical features from interaction data.

        Args:
            interactions: DataFrame with user-item interactions.

        Returns:
            Dictionary mapping item IDs to feature dictionaries.
        """
        item_stats = {}

        # Group by item and compute statistics
        for item_id in interactions['itemid'].unique():
            item_data = interactions[interactions['itemid'] == item_id]

            stats = {
                'popularity': len(item_data),
                'avg_weight': item_data['weight'].mean() if 'weight' in item_data.columns else 1.0,
                'unique_users': item_data['visitorid'].nunique(),
                'transaction_ratio': len(item_data[item_data['event'] == 'transaction']) / len(item_data)
            }

            item_stats[item_id] = stats

        return item_stats

    def _enhance_with_metadata(
        self,
        item_stats: Dict[int, Dict],
        item_metadata: pd.DataFrame
    ) -> None:
        """Enhance interaction features with item metadata.

        Args:
            item_stats: Dictionary of item statistics from interactions.
            item_metadata: DataFrame with item attributes.
        """
        for item_id, stats in item_stats.items():
            if item_id in item_metadata.index:
                # Combine interaction stats with metadata
                features = np.array([
                    stats['popularity'],
                    stats['avg_weight'],
                    stats['unique_users'],
                    stats['transaction_ratio']
                ])

                self.item_features[item_id] = features
            else:
                # Fallback to interaction-only features
                self.item_features[item_id] = np.array([
                    stats['popularity'],
                    stats['avg_weight'],
                    stats['unique_users'],
                    stats['transaction_ratio']
                ])

    def _create_feature_matrix(self) -> None:
        """Create feature matrix for similarity computation."""
        if not self.item_features:
            return

        self.item_ids = list(self.item_features.keys())
        self.feature_matrix = np.array([self.item_features[iid] for iid in self.item_ids])

        # Normalize features
        if self.feature_matrix is not None:
            feature_norms = np.linalg.norm(self.feature_matrix, axis=1, keepdims=True)
            feature_norms[feature_norms == 0] = 1  # Avoid division by zero
            self.feature_matrix = self.feature_matrix / feature_norms

    def get_item_similarity(self, item_id_1: int, item_id_2: int) -> float:
        """Compute cosine similarity between two items.

        Args:
            item_id_1: First item ID.
            item_id_2: Second item ID.

        Returns:
            Cosine similarity score between 0 and 1.
        """
        if (item_id_1 not in self.item_features or
            item_id_2 not in self.item_features or
            self.feature_matrix is None):
            return 0.0

        idx1 = self.item_ids.index(item_id_1)
        idx2 = self.item_ids.index(item_id_2)

        similarity = cosine_similarity(
            self.feature_matrix[idx1:idx2+1],
            self.feature_matrix[idx2:idx2+1]
        )[0, 0]

        return float(similarity)

    def get_similar_items(self, item_id: int, top_k: int = 10) -> List[tuple[int, float]]:
        """Find most similar items to a given item.

        Args:
            item_id: Target item ID.
            top_k: Number of similar items to return.

        Returns:
            List of (item_id, similarity_score) tuples.
        """
        if item_id not in self.item_features or self.feature_matrix is None:
            return []

        idx = self.item_ids.index(item_id)
        similarities = cosine_similarity(
            self.feature_matrix[idx:idx+1],
            self.feature_matrix
        )[0]

        # Get top-k similar items (excluding the item itself)
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]

        return [
            (self.item_ids[i], float(similarities[i]))
            for i in similar_indices
        ]

    def add_new_item(self, item_id: int, features: np.ndarray) -> None:
        """Add a new item with features (for item cold start).

        Args:
            item_id: New item ID.
            features: Feature vector for the new item.
        """
        self.item_features[item_id] = features
        self.item_ids.append(item_id)

        # Recreate feature matrix
        self._create_feature_matrix()


class ContentBasedRecommender:
    """Content-based recommender for handling item cold start.

    This recommender uses item features to make recommendations for
    new items or when collaborative filtering is not available.
    """

    def __init__(self, feature_extractor: ItemFeatureExtractor) -> None:
        """Initialize the content-based recommender.

        Args:
            feature_extractor: Fitted item feature extractor.
        """
        self.feature_extractor = feature_extractor

    def recommend_for_user(
        self,
        user_id: int,
        user_interactions: pd.DataFrame,
        k: int = 10,
        candidate_items: Optional[List[int]] = None
    ) -> List[tuple[int, float]]:
        """Generate content-based recommendations for a user.

        Args:
            user_id: User ID.
            user_interactions: DataFrame with user's past interactions.
            k: Number of recommendations to return.
            candidate_items: Optional list of candidate item IDs.

        Returns:
            List of (item_id, score) tuples.
        """
        # Get user's interacted items
        user_items = user_interactions[user_interactions['visitorid'] == user_id]['itemid'].unique()

        if len(user_items) == 0:
            # Fallback to popular items
            return self._get_popular_items(k, candidate_items)

        # Find similar items to user's interacted items
        item_scores = {}
        for interacted_item in user_items:
            similar_items = self.feature_extractor.get_similar_items(interacted_item, top_k=k*2)
            for item_id, similarity in similar_items:
                if candidate_items is None or item_id in candidate_items:
                    item_scores[item_id] = item_scores.get(item_id, 0) + similarity

        # Sort by score and return top-k
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:k]

    def recommend_similar_items(
        self,
        item_id: int,
        k: int = 10
    ) -> List[tuple[int, float]]:
        """Recommend items similar to a given item.

        Args:
            item_id: Target item ID.
            k: Number of recommendations to return.

        Returns:
            List of (item_id, similarity_score) tuples.
        """
        return self.feature_extractor.get_similar_items(item_id, top_k=k)

    def _get_popular_items(
        self,
        k: int,
        candidate_items: Optional[List[int]] = None
    ) -> List[tuple[int, float]]:
        """Fallback to popular items when no user history available.

        Args:
            k: Number of recommendations to return.
            candidate_items: Optional list of candidate item IDs.

        Returns:
            List of (item_id, score) tuples.
        """
        # Sort items by popularity (first feature dimension)
        item_popularity = [
            (item_id, float(features[0]))
            for item_id, features in self.feature_extractor.item_features.items()
            if candidate_items is None or item_id in candidate_items
        ]

        item_popularity.sort(key=lambda x: x[1], reverse=True)
        return item_popularity[:k]
