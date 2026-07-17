"""Concrete predictor implementations.

This module contains specific predictor strategies that implement the BasePredictor interface.
Each predictor follows the Single Responsibility Principle by handling a specific
prediction use case.
"""

from __future__ import annotations

import logging

import torch

from ..exceptions import InvalidInputError
from ..models.schemas import (
    PredictionRequest,
    PredictionResponse,
    RecommendationResponse,
)
from .base_predictor import BasePredictor

logger = logging.getLogger(__name__)


class SingleUserPredictor(BasePredictor):
    """Predictor for single user predictions.

    This predictor handles predictions for a single user against specified items.
    It follows the Single Responsibility Principle by focusing only on single-user scenarios.
    """

    name = "single_user"

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Generate predictions for a single user.

        Args:
            request: The prediction request containing user and item information.

        Returns:
            A prediction response with item scores.

        Raises:
            InvalidInputError: If the request contains invalid data.
        """
        logger.info(
            "Generating predictions for user %d with %d items",
            request.user_id,
            len(request.item_ids) if request.item_ids else 0,
        )

        if not request.item_ids:
            logger.error("No item_ids provided for single user prediction")
            raise InvalidInputError(
                "item_ids must be provided for single user prediction."
            )

        user_idx = self._get_user_idx(request.user_id)

        # Cold start fallback: use popularity scores for unknown users
        if user_idx is None:
            logger.info("Using cold start fallback for user %d", request.user_id)
            if self.enable_hybrid:
                item_scores = self._get_content_based_scores(request.item_ids)
            else:
                item_scores = self._get_popular_item_scores(request.item_ids)
            return PredictionResponse(
                user_id=request.user_id,
                item_scores=item_scores,
                metadata={
                    "predictor": self.name,
                    "model_type": self.model.model_name,
                    "cold_start": True,
                },
            )

        item_indices = self._get_item_indices(request.item_ids)

        # Handle unknown items with hybrid approach
        known_items = []
        known_indices = []
        unknown_items = []

        for item_id, item_idx in zip(request.item_ids, item_indices, strict=False):
            if item_idx is not None:
                known_items.append(item_id)
                known_indices.append(item_idx)
            else:
                unknown_items.append(item_id)

        item_scores = {}

        # Get CF scores for known items
        if known_indices:
            with torch.no_grad():
                user_tensor = torch.tensor([user_idx] * len(known_indices), dtype=torch.long)
                item_tensor = torch.tensor(known_indices, dtype=torch.long)
                scores = self.model(user_tensor, item_tensor)

            item_scores.update({
                item_id: float(score)
                for item_id, score in zip(known_items, scores.squeeze().tolist(), strict=True)
            })

        # Get content-based scores for unknown items
        if unknown_items and self.enable_hybrid:
            content_scores = self._get_content_based_scores(unknown_items)
            item_scores.update(content_scores)
        elif unknown_items:
            # Fallback to popularity for unknown items
            popularity_scores = self._get_popular_item_scores(unknown_items)
            item_scores.update(popularity_scores)

        logger.debug(
            "Generated %d item scores for user %d", len(item_scores), request.user_id
        )

        cold_start_used = user_idx is None or len(unknown_items) > 0
        return PredictionResponse(
            user_id=request.user_id,
            item_scores=item_scores,
            metadata={
                "predictor": self.name,
                "model_type": self.model.model_name,
                "cold_start": cold_start_used,
            },
        )

    def predict_batch(
        self, requests: list[PredictionRequest]
    ) -> list[PredictionResponse]:
        """Generate predictions for multiple users sequentially.

        Args:
            requests: List of prediction requests.

        Returns:
            List of prediction responses.
        """
        logger.info("Generating batch predictions for %d users", len(requests))
        return [self.predict(request) for request in requests]


class TopKRecommendationPredictor(BasePredictor):
    """Predictor for top-k recommendations.

    This predictor generates top-k item recommendations for a user by scoring
    all items and returning the highest-scoring ones.
    """

    name = "top_k"

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Generate predictions for a single user.

        For TopKRecommendationPredictor, this method scores all items if k is specified,
        otherwise it requires item_ids to be provided.

        Args:
            request: The prediction request containing user and item information.

        Returns:
            A prediction response with item scores.

        Raises:
            InvalidInputError: If the request contains invalid data.
        """
        logger.info(
            "Generating predictions for user %d (k=%s, items=%d)",
            request.user_id,
            request.k,
            len(request.item_ids) if request.item_ids else 0,
        )

        if request.k is not None:
            return self._predict_top_k(request)
        elif request.item_ids:
            user_idx = self._get_user_idx(request.user_id)

            # Cold start fallback: use popularity scores for unknown users
            if user_idx is None:
                logger.info("Using cold start fallback for user %d", request.user_id)
                item_scores = self._get_popular_item_scores(request.item_ids)
                return PredictionResponse(
                    user_id=request.user_id,
                    item_scores=item_scores,
                    metadata={
                        "predictor": self.name,
                        "model_type": self.model.model_name,
                        "cold_start": True,
                    },
                )

            item_indices = self._get_item_indices(request.item_ids)

            # Check if any items are unknown (None indices)
            if None in item_indices:
                logger.info("Using hybrid cold start for unknown items")
                # Use content-based scores for unknown items
                content_scores = self._get_content_based_scores(request.item_ids)

                # Get known items for model prediction
                known_items = [(idx, item_id) for idx, item_id in zip(item_indices, request.item_ids, strict=False) if idx is not None]

                if known_items:
                    # Get model scores for known items
                    known_indices = [idx for idx, _ in known_items]
                    known_item_ids = [item_id for _, item_id in known_items]

                    with torch.no_grad():
                        user_tensor = torch.tensor([user_idx] * len(known_indices), dtype=torch.long)
                        item_tensor = torch.tensor(known_indices, dtype=torch.long)
                        scores = self.model(user_tensor, item_tensor)

                    model_scores = {
                        item_id: float(score)
                        for item_id, score in zip(known_item_ids, scores.squeeze().tolist(), strict=True)
                    }
                    # Combine with content scores
                    content_scores.update(model_scores)

                item_scores = content_scores
            else:
                # All items are known, use model prediction
                with torch.no_grad():
                    user_tensor = torch.tensor([user_idx] * len(item_indices), dtype=torch.long)
                    item_tensor = torch.tensor(item_indices, dtype=torch.long)
                    scores = self.model(user_tensor, item_tensor)

                item_scores = {
                    item_id: float(score)
                    for item_id, score in zip(
                        request.item_ids, scores.squeeze().tolist(), strict=True
                    )
                }

            logger.debug(
                "Generated %d item scores for user %d",
                len(item_scores),
                request.user_id,
            )

            return PredictionResponse(
                user_id=request.user_id,
                item_scores=item_scores,
                metadata={"predictor": self.name, "model_type": self.model.model_name},
            )
        else:
            logger.error("Neither item_ids nor k provided for prediction")
            raise InvalidInputError("Either item_ids or k must be provided.")

    def predict_batch(
        self, requests: list[PredictionRequest]
    ) -> list[PredictionResponse]:
        """Generate predictions for multiple users sequentially.

        Args:
            requests: List of prediction requests.

        Returns:
            List of prediction responses.
        """
        return [self.predict(request) for request in requests]

    def _predict_top_k(self, request: PredictionRequest) -> PredictionResponse:
        """Generate top-k recommendations for a single user.

        Args:
            request: The prediction request containing user and k value.

        Returns:
            A prediction response with top-k item scores.

        Raises:
            InvalidInputError: If k is not specified or is invalid.
        """
        if request.k is None or request.k <= 0:
            logger.error("Invalid k value: %s", request.k)
            raise InvalidInputError("k must be a positive integer.")

        logger.info(
            "Generating top-%d recommendations for user %d", request.k, request.user_id
        )

        user_idx = self._get_user_idx(request.user_id)

        # Cold start fallback: use popular items for unknown users
        if user_idx is None:
            logger.info("Using cold start fallback for user %d", request.user_id)
            if self.enable_hybrid:
                popular_items = self._get_content_based_items(request.k)
            else:
                popular_items = self._get_popular_items(request.k)
            top_k_scores = {
                item_id: float(score) for item_id, score in popular_items
            }
            return PredictionResponse(
                user_id=request.user_id,
                item_scores=top_k_scores,
                metadata={
                    "predictor": self.name,
                    "model_type": self.model.model_name,
                    "k": request.k,
                    "cold_start": True,
                },
            )

        num_items = len(self.item2idx)

        with torch.no_grad():
            user_tensor = torch.tensor([user_idx] * num_items, dtype=torch.long)
            item_tensor = torch.tensor(range(num_items), dtype=torch.long)
            scores = self.model(user_tensor, item_tensor)

        item_scores = list(
            zip(range(num_items), scores.squeeze().tolist(), strict=True)
        )
        item_scores.sort(key=lambda x: x[1], reverse=True)

        top_k_scores = {
            self.idx2item[idx]: score for idx, score in item_scores[: request.k]
        }

        logger.debug(
            "Generated top-%d recommendations for user %d", request.k, request.user_id
        )

        return PredictionResponse(
            user_id=request.user_id,
            item_scores=top_k_scores,
            metadata={
                "predictor": self.name,
                "model_type": self.model.model_name,
                "k": request.k,
            },
        )

    def recommend(self, user_id: int, k: int = 10) -> RecommendationResponse:
        """Generate top-k recommendations for a user.

        Args:
            user_id: The user ID to generate recommendations for.
            k: Number of recommendations to return.

        Returns:
            A recommendation response with top-k items and scores.

        Raises:
            InvalidInputError: If k is invalid or user not found.
        """
        if k <= 0:
            logger.error("Invalid k value: %d", k)
            raise InvalidInputError("k must be a positive integer.")

        logger.info("Generating top-%d recommendations for user %d", k, user_id)

        user_idx = self._get_user_idx(user_id)

        # Cold start fallback: return popular items for unknown users
        if user_idx is None:
            logger.info("Using cold start fallback for user %d", user_id)
            if self.enable_hybrid:
                popular_items = self._get_content_based_items(k)
            else:
                popular_items = self._get_popular_items(k)
            recommendations = [
                (item_id, float(score)) for item_id, score in popular_items
            ]
            return RecommendationResponse(
                user_id=user_id,
                recommendations=recommendations,
                metadata={
                    "predictor": self.name,
                    "model_type": self.model.model_name,
                    "k": k,
                    "cold_start": True,
                },
            )

        num_items = len(self.item2idx)

        with torch.no_grad():
            user_tensor = torch.tensor([user_idx] * num_items, dtype=torch.long)
            item_tensor = torch.tensor(range(num_items), dtype=torch.long)
            scores = self.model(user_tensor, item_tensor)

        item_scores = list(
            zip(range(num_items), scores.squeeze().tolist(), strict=True)
        )
        item_scores.sort(key=lambda x: x[1], reverse=True)

        recommendations = [
            (self.idx2item[idx], float(score)) for idx, score in item_scores[:k]
        ]

        logger.debug(
            "Generated %d recommendations for user %d", len(recommendations), user_id
        )

        return RecommendationResponse(
            user_id=user_id,
            recommendations=recommendations,
            metadata={
                "predictor": self.name,
                "model_type": self.model.model_name,
                "k": k,
            },
        )


class BatchPredictor(BasePredictor):
    """Optimized predictor for batch predictions.

    This predictor handles multiple user-item pairs in a single forward pass
    for better performance on large batches.
    """

    name = "batch"

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Generate predictions for a single user.

        Args:
            request: The prediction request containing user and item information.

        Returns:
            A prediction response with item scores.

        Raises:
            InvalidInputError: If the request contains invalid data.
        """
        logger.info(
            "Generating predictions for user %d with %d items",
            request.user_id,
            len(request.item_ids) if request.item_ids else 0,
        )

        if not request.item_ids:
            logger.error("No item_ids provided for prediction")
            raise InvalidInputError("item_ids must be provided for prediction.")

        user_idx = self._get_user_idx(request.user_id)

        # Cold start fallback: use popularity scores for unknown users
        if user_idx is None:
            logger.info("Using cold start fallback for user %d", request.user_id)
            if self.enable_hybrid:
                item_scores = self._get_content_based_scores(request.item_ids)
            else:
                item_scores = self._get_popular_item_scores(request.item_ids)
            return PredictionResponse(
                user_id=request.user_id,
                item_scores=item_scores,
                metadata={
                    "predictor": self.name,
                    "model_type": self.model.model_name,
                    "cold_start": True,
                },
            )

        item_indices = self._get_item_indices(request.item_ids)

        # Handle unknown items with hybrid approach
        known_items = []
        known_indices = []
        unknown_items = []

        for item_id, item_idx in zip(request.item_ids, item_indices, strict=False):
            if item_idx is not None:
                known_items.append(item_id)
                known_indices.append(item_idx)
            else:
                unknown_items.append(item_id)

        item_scores = {}

        # Get CF scores for known items
        if known_indices:
            with torch.no_grad():
                user_tensor = torch.tensor([user_idx] * len(known_indices), dtype=torch.long)
                item_tensor = torch.tensor(known_indices, dtype=torch.long)
                scores = self.model(user_tensor, item_tensor)

            item_scores.update({
                item_id: float(score)
                for item_id, score in zip(known_items, scores.squeeze().tolist(), strict=True)
            })

        # Get content-based scores for unknown items
        if unknown_items and self.enable_hybrid:
            content_scores = self._get_content_based_scores(unknown_items)
            item_scores.update(content_scores)
        elif unknown_items:
            # Fallback to popularity for unknown items
            popularity_scores = self._get_popular_item_scores(unknown_items)
            item_scores.update(popularity_scores)

        logger.debug(
            "Generated %d item scores for user %d", len(item_scores), request.user_id
        )

        cold_start_used = user_idx is None or len(unknown_items) > 0
        return PredictionResponse(
            user_id=request.user_id,
            item_scores=item_scores,
            metadata={
                "predictor": self.name,
                "model_type": self.model.model_name,
                "cold_start": cold_start_used,
            },
        )

    def predict_batch(
        self, requests: list[PredictionRequest]
    ) -> list[PredictionResponse]:
        """Generate predictions for multiple users in a single batch.

        This method optimizes performance by batching all user-item pairs together.

        Args:
            requests: List of prediction requests.

        Returns:
            List of prediction responses.

        Raises:
            InvalidInputError: If any request contains invalid data.
        """
        if not requests:
            logger.info("Empty batch prediction request")
            return []

        logger.info("Generating batch predictions for %d users", len(requests))

        # Collect all user-item pairs
        user_indices = []
        item_indices = []
        request_indices = []

        for req_idx, request in enumerate(requests):
            if not request.item_ids:
                logger.error("No item_ids provided for request %d", req_idx)
                raise InvalidInputError(
                    f"item_ids must be provided for request {req_idx}."
                )

            user_idx = self._get_user_idx(request.user_id)
            item_idxs = self._get_item_indices(request.item_ids)

            user_indices.extend([user_idx] * len(item_idxs))
            item_indices.extend(item_idxs)
            request_indices.extend([req_idx] * len(item_idxs))

        # Batch prediction
        logger.debug("Batching %d user-item pairs", len(user_indices))
        with torch.no_grad():
            user_tensor = torch.tensor(user_indices, dtype=torch.long)
            item_tensor = torch.tensor(item_indices, dtype=torch.long)
            scores = self.model(user_tensor, item_tensor)

        # Distribute scores back to requests
        scores_list = scores.squeeze().tolist()
        responses: list[PredictionResponse] = [None] * len(requests)

        for req_idx, request in enumerate(requests):
            start_idx = request_indices.index(req_idx)
            end_idx = len(request_indices) - request_indices[::-1].index(req_idx)
            req_scores = scores_list[start_idx:end_idx]

            item_scores = {
                item_id: float(score)
                for item_id, score in zip(request.item_ids, req_scores, strict=True)
            }

            responses[req_idx] = PredictionResponse(
                user_id=request.user_id,
                item_scores=item_scores,
                metadata={"predictor": self.name, "model_type": self.model.model_name},
            )

        logger.info("Completed batch predictions for %d users", len(responses))
        return responses
