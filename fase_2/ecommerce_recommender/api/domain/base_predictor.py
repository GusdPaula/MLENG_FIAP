"""Base predictor interface following SOLID principles.

This abstract class defines the contract that all concrete predictors must implement.
It follows the Interface Segregation Principle by providing a focused interface
and the Liskov Substitution Principle by ensuring all implementations can be
used interchangeably.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch

from ..models.schemas import PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)


class BasePredictor(ABC):
    """Abstract base class for all predictors.

    This class defines the interface that all concrete predictors must implement.
    It ensures that different prediction strategies can be used interchangeably
    (Liskov Substitution Principle) and that the system is open for extension
    but closed for modification (Open/Closed Principle).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        user2idx: dict[int, int],
        item2idx: dict[int, int],
        popular_items: dict[int, float] | None = None,
        feature_extractor=None,
        content_recommender=None,
    ):
        """Initialize the base predictor.

        Args:
            model: The trained recommender model.
            user2idx: Mapping from user IDs to internal indices.
            item2idx: Mapping from item IDs to internal indices.
            popular_items: Optional mapping of item IDs to popularity scores for cold start fallback.
            feature_extractor: Optional item feature extractor for content-based recommendations.
            content_recommender: Optional content-based recommender for hybrid approach.
        """
        self.model = model
        self.user2idx = user2idx
        self.item2idx = item2idx
        self.idx2user = {idx: user for user, idx in user2idx.items()}
        self.idx2item = {idx: item for item, idx in item2idx.items()}
        self.popular_items = popular_items or {}
        self.enable_cold_start_fallback = len(self.popular_items) > 0
        self.feature_extractor = feature_extractor
        self.content_recommender = content_recommender
        self.enable_hybrid = (feature_extractor is not None and content_recommender is not None)
        self.model.eval()
        logger.info(
            "Initialized %s with %d users and %d items",
            self.__class__.__name__,
            len(user2idx),
            len(item2idx),
        )
        if self.enable_cold_start_fallback:
            logger.info(
                "Cold start fallback enabled with %d popular items",
                len(self.popular_items),
            )
        if self.enable_hybrid:
            logger.info(
                "Hybrid cold start enabled with content-based recommendations"
            )

    @abstractmethod
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Generate predictions for a single user.

        Args:
            request: The prediction request containing user and item information.

        Returns:
            A prediction response with item scores.

        Raises:
            InvalidInputError: If the request contains invalid data.
        """

    @abstractmethod
    def predict_batch(
        self, requests: list[PredictionRequest]
    ) -> list[PredictionResponse]:
        """Generate predictions for multiple users.

        Args:
            requests: List of prediction requests.

        Returns:
            List of prediction responses.

        Raises:
            InvalidInputError: If any request contains invalid data.
        """

    def _get_user_idx(self, user_id: int) -> int | None:
        """Get internal index for a user ID.

        Args:
            user_id: The external user ID.

        Returns:
            The internal user index, or None if user not found and cold start fallback is enabled.

        Raises:
            InvalidInputError: If the user ID is not found and cold start fallback is disabled.
        """
        if user_id not in self.user2idx:
            if self.enable_cold_start_fallback or self.enable_hybrid:
                logger.warning(
                    "User ID %d not found in training data, using cold start fallback",
                    user_id,
                )
                return None
            else:
                from ..exceptions import InvalidInputError

                logger.error("User ID %d not found in training data", user_id)
                raise InvalidInputError(
                    f"User ID {user_id} not found in training data."
                )
        return self.user2idx[user_id]

    def _get_item_idx(self, item_id: int) -> int | None:
        """Get internal index for an item ID.

        Args:
            item_id: The external item ID.

        Returns:
            The internal item index, or None if item not found and hybrid cold start is enabled.

        Raises:
            InvalidInputError: If the item ID is not found and cold start fallback is disabled.
        """
        if item_id not in self.item2idx:
            if self.enable_hybrid:
                logger.warning(
                    "Item ID %d not found in training data, using hybrid cold start",
                    item_id,
                )
                return None
            elif self.enable_cold_start_fallback:
                logger.warning(
                    "Item ID %d not found in training data, using popularity fallback",
                    item_id,
                )
                return None
            else:
                from ..exceptions import InvalidInputError

                logger.error("Item ID %d not found in training data", item_id)
                raise InvalidInputError(f"Item ID {item_id} not found in training data.")
        return self.item2idx[item_id]

    def _get_item_indices(self, item_ids: list[int]) -> list[int | None]:
        """Get internal indices for multiple item IDs.

        Args:
            item_ids: List of external item IDs.

        Returns:
            List of internal item indices, with None for unknown items when hybrid is enabled.

        Raises:
            InvalidInputError: If any item ID is not found and cold start fallback is disabled.
        """
        from ..exceptions import InvalidInputError

        indices = []
        for item_id in item_ids:
            if item_id not in self.item2idx:
                if self.enable_hybrid or self.enable_cold_start_fallback:
                    logger.warning("Item ID %d not found in training data, will use cold start", item_id)
                    indices.append(None)
                else:
                    logger.error("Item ID %d not found in training data", item_id)
                    raise InvalidInputError(
                        f"Item ID {item_id} not found in training data."
                    )
            else:
                indices.append(self.item2idx[item_id])
        return indices

    def _get_popular_items(self, k: int = 10) -> list[tuple[int, float]]:
        """Get top-k popular items for cold start fallback.

        Args:
            k: Number of popular items to return.

        Returns:
            List of (item_id, popularity_score) tuples sorted by popularity.
        """
        if not self.enable_cold_start_fallback:
            return []

        # Sort items by popularity score and return top-k
        sorted_items = sorted(
            self.popular_items.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_items[:k]

    def _get_popular_item_scores(self, item_ids: list[int]) -> dict[int, float]:
        """Get popularity scores for specific items.

        Args:
            item_ids: List of item IDs to get popularity scores for.

        Returns:
            Dictionary mapping item IDs to their popularity scores.
            Items not in popular_items get a score of 0.0.
        """
        if not self.enable_cold_start_fallback:
            return dict.fromkeys(item_ids, 0.0)

        return {item_id: self.popular_items.get(item_id, 0.0) for item_id in item_ids}

    def _get_content_based_scores(self, item_ids: list[int]) -> dict[int, float]:
        """Get content-based scores for specific items using hybrid approach.

        Args:
            item_ids: List of item IDs to get content-based scores for.

        Returns:
            Dictionary mapping item IDs to their content-based scores.
            Items not in feature extractor get a score of 0.0.
        """
        if not self.enable_hybrid or self.feature_extractor is None:
            return self._get_popular_item_scores(item_ids)

        scores = {}
        for item_id in item_ids:
            if item_id in self.feature_extractor.item_features:
                # Use normalized popularity from features
                features = self.feature_extractor.item_features[item_id]
                # Normalize to 0-1 range
                max_pop = max(f[0] for f in self.feature_extractor.item_features.values()) if self.feature_extractor.item_features else 1.0
                score = features[0] / max_pop if max_pop > 0 else 0.0
                scores[item_id] = score
            else:
                # Fallback to popularity for items not in feature extractor
                scores[item_id] = self.popular_items.get(item_id, 0.0)
        return scores

    def _get_content_based_items(self, k: int = 10) -> list[tuple[int, float]]:
        """Get top-k items using content-based features for cold start.

        Args:
            k: Number of items to return.

        Returns:
            List of (item_id, score) tuples sorted by content-based score.
        """
        if not self.enable_hybrid or self.feature_extractor is None:
            return self._get_popular_items(k)

        # Sort items by their feature-based popularity
        item_scores = [
            (item_id, features[0])
            for item_id, features in self.feature_extractor.item_features.items()
        ]

        # Normalize scores
        if item_scores:
            max_score = max(score for _, score in item_scores)
            if max_score > 0:
                item_scores = [(item_id, score / max_score) for item_id, score in item_scores]

        item_scores.sort(key=lambda x: x[1], reverse=True)
        return item_scores[:k]
