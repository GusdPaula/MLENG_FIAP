"""Hybrid recommendation model combining collaborative filtering and content-based features."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from ..features.item_features import ContentBasedRecommender, ItemFeatureExtractor
from .base import BaseRecommender


class HybridRecommender(BaseRecommender):
    """Hybrid recommender combining collaborative filtering with content-based features.

    This model dynamically weights collaborative filtering and content-based
    recommendations based on data availability to handle cold start scenarios.
    """

    def __init__(
        self,
        cf_model: BaseRecommender,
        content_recommender: ContentBasedRecommender,
        feature_extractor: ItemFeatureExtractor,
        alpha: float = 0.7,
    ) -> None:
        """Initialize the hybrid recommender.

        Args:
            cf_model: Collaborative filtering model (NCF, GMF, etc).
            content_recommender: Content-based recommender for cold start.
            feature_extractor: Item feature extractor.
            alpha: Weight for collaborative filtering (0-1). Content-based weight = 1-alpha.
        """
        super().__init__()
        self.cf_model = cf_model
        self.content_recommender = content_recommender
        self.feature_extractor = feature_extractor
        self.alpha = alpha

        # Copy mappings from CF model
        self.user2idx = cf_model.user2idx
        self.item2idx = cf_model.item2idx
        self.idx2item = cf_model.idx2item

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Forward pass for hybrid scoring.

        Args:
            users: User indices tensor.
            items: Item indices tensor.

        Returns:
            Combined scores tensor.
        """
        # Get collaborative filtering scores
        cf_scores = self.cf_model(users, items)

        # Get content-based scores
        content_scores = self._get_content_scores(users, items)

        # Dynamic weighting based on data availability
        weights = self._compute_dynamic_weights(users, items)

        # Combine scores
        combined_scores = weights * cf_scores + (1 - weights) * content_scores
        return combined_scores

    def _get_content_scores(
        self, users: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        """Compute content-based scores for user-item pairs.

        Args:
            users: User indices tensor.
            items: Item indices tensor.

        Returns:
            Content-based scores tensor.
        """
        # Convert to item IDs
        item_ids = [self.idx2item[idx.item()] for idx in items]
        user_ids = [self.idx2user[idx.item()] if hasattr(self, 'idx2user') else None for idx in users]

        # Compute content-based similarity scores
        scores = []
        for _user_id, item_id in zip(user_ids, item_ids, strict=False):
            # For content-based, use item popularity as baseline
            if item_id in self.feature_extractor.item_features:
                features = self.feature_extractor.item_features[item_id]
                # Use normalized popularity as score
                score = features[0] / (features[0].max() if len(features) > 0 else 1)
            else:
                score = 0.0
            scores.append(score)

        return torch.tensor(scores, dtype=torch.float32, device=items.device)

    def _compute_dynamic_weights(
        self, users: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        """Compute dynamic weights based on data availability.

        Args:
            users: User indices tensor.
            items: Item indices tensor.

        Returns:
            Weight tensor for collaborative filtering (0-1).
        """
        weights = []
        for user_idx, item_idx in zip(users, items, strict=False):
            # Check if user and item are known
            user_known = user_idx < len(self.user2idx)
            item_known = item_idx < len(self.item2idx)

            if user_known and item_known:
                # Both known: use pure collaborative filtering
                weight = 1.0
            elif user_known and not item_known:
                # Item cold start: use content-based
                weight = 0.3
            elif not user_known and item_known:
                # User cold start: mix of CF and content
                weight = 0.5
            else:
                # Both cold start: use content-based
                weight = 0.2

            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float32, device=users.device).unsqueeze(1)

    def predict(
        self,
        user_id: int,
        item_ids: List[int],
        user_interactions: Optional[object] = None,
    ) -> Dict[int, float]:
        """Generate predictions for a user-item pair.

        Args:
            user_id: User ID.
            item_ids: List of item IDs.
            user_interactions: Optional user interaction data for content-based.

        Returns:
            Dictionary mapping item IDs to scores.
        """
        user_idx = self.user2idx.get(user_id)
        item_indices = []
        unknown_items = []

        for item_id in item_ids:
            if item_id in self.item2idx:
                item_indices.append(self.item2idx[item_id])
            else:
                item_indices.append(-1)  # Unknown item marker
                unknown_items.append(item_id)

        if user_idx is not None:
            # Known user: use collaborative filtering for known items
            known_item_indices = [i for i in item_indices if i >= 0]
            known_item_ids = [item_ids[i] for i, idx in enumerate(item_indices) if idx >= 0]

            if known_item_indices:
                with torch.no_grad():
                    user_tensor = torch.tensor([user_idx] * len(known_item_indices), dtype=torch.long)
                    item_tensor = torch.tensor(known_item_indices, dtype=torch.long)
                    cf_scores = self.cf_model(user_tensor, item_tensor)

                # Combine with content-based for unknown items
                scores = {}
                for item_id, score in zip(known_item_ids, cf_scores.squeeze().tolist(), strict=False):
                    scores[item_id] = score * self.alpha

                # Handle unknown items with content-based
                if unknown_items and user_interactions is not None:
                    content_recs = self.content_recommender.recommend_for_user(
                        user_id, user_interactions, k=len(unknown_items)
                    )
                    for item_id, content_score in content_recs:
                        if item_id in unknown_items:
                            scores[item_id] = content_score * (1 - self.alpha)

                return scores
            else:
                # All items unknown: use content-based
                if user_interactions is not None:
                    content_recs = self.content_recommender.recommend_for_user(
                        user_id, user_interactions, k=len(item_ids)
                    )
                    return dict(content_recs)
                else:
                    # Fallback to popularity
                    return dict.fromkeys(item_ids, 0.5)
        else:
            # Unknown user: use content-based recommendations
            if user_interactions is not None:
                content_recs = self.content_recommender.recommend_for_user(
                    user_id, user_interactions, k=len(item_ids)
                )
                return dict(content_recs)
            else:
                # Fallback to popularity
                return dict.fromkeys(item_ids, 0.5)

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        user_interactions: Optional[object] = None,
    ) -> List[Tuple[int, float]]:
        """Generate top-k recommendations for a user.

        Args:
            user_id: User ID.
            k: Number of recommendations.
            user_interactions: Optional user interaction data.

        Returns:
            List of (item_id, score) tuples.
        """
        user_idx = self.user2idx.get(user_id)

        if user_idx is not None:
            # Known user: use collaborative filtering
            num_items = len(self.item2idx)
            with torch.no_grad():
                user_tensor = torch.tensor([user_idx] * num_items, dtype=torch.long)
                item_tensor = torch.tensor(range(num_items), dtype=torch.long)
                cf_scores = self.cf_model(user_tensor, item_tensor)

            # Get top-k from CF
            item_scores = list(zip(range(num_items), cf_scores.squeeze().tolist(), strict=False))
            item_scores.sort(key=lambda x: x[1], reverse=True)
            top_k_cf = [(self.idx2item[idx], score) for idx, score in item_scores[:k]]

            # If user has interactions, enhance with content-based
            if user_interactions is not None:
                content_recs = self.content_recommender.recommend_for_user(
                    user_id, user_interactions, k=k
                )
                # Combine and re-rank
                combined_scores = {}
                for item_id, score in top_k_cf:
                    combined_scores[item_id] = score * self.alpha
                for item_id, score in content_recs:
                    combined_scores[item_id] = combined_scores.get(item_id, 0) + score * (1 - self.alpha)

                sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
                return sorted_recs[:k]
            else:
                return top_k_cf
        else:
            # Unknown user: use content-based
            if user_interactions is not None:
                return self.content_recommender.recommend_for_user(
                    user_id, user_interactions, k=k
                )
            else:
                # Fallback to popular items
                popular_items = self.feature_extractor._get_popular_items(k)
                return popular_items

    @property
    def model_name(self) -> str:
        """Return model name."""
        return f"hybrid_{self.cf_model.model_name}"
