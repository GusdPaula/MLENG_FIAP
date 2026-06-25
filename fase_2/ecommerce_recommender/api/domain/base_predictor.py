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
        self, model: torch.nn.Module, user2idx: dict[int, int], item2idx: dict[int, int]
    ):
        """Initialize the base predictor.

        Args:
            model: The trained recommender model.
            user2idx: Mapping from user IDs to internal indices.
            item2idx: Mapping from item IDs to internal indices.
        """
        self.model = model
        self.user2idx = user2idx
        self.item2idx = item2idx
        self.idx2user = {idx: user for user, idx in user2idx.items()}
        self.idx2item = {idx: item for item, idx in item2idx.items()}
        self.model.eval()
        logger.info(
            "Initialized %s with %d users and %d items",
            self.__class__.__name__,
            len(user2idx),
            len(item2idx),
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

    def _get_user_idx(self, user_id: int) -> int:
        """Get internal index for a user ID.

        Args:
            user_id: The external user ID.

        Returns:
            The internal user index.

        Raises:
            InvalidInputError: If the user ID is not found.
        """
        if user_id not in self.user2idx:
            from ..exceptions import InvalidInputError

            logger.error("User ID %d not found in training data", user_id)
            raise InvalidInputError(f"User ID {user_id} not found in training data.")
        return self.user2idx[user_id]

    def _get_item_idx(self, item_id: int) -> int:
        """Get internal index for an item ID.

        Args:
            item_id: The external item ID.

        Returns:
            The internal item index.

        Raises:
            InvalidInputError: If the item ID is not found.
        """
        if item_id not in self.item2idx:
            from .exceptions import InvalidInputError

            logger.error("Item ID %d not found in training data", item_id)
            raise InvalidInputError(f"Item ID {item_id} not found in training data.")
        return self.item2idx[item_id]

    def _get_item_indices(self, item_ids: list[int]) -> list[int]:
        """Get internal indices for multiple item IDs.

        Args:
            item_ids: List of external item IDs.

        Returns:
            List of internal item indices.

        Raises:
            InvalidInputError: If any item ID is not found.
        """
        from ..exceptions import InvalidInputError

        indices = []
        for item_id in item_ids:
            if item_id not in self.item2idx:
                logger.error("Item ID %d not found in training data", item_id)
                raise InvalidInputError(
                    f"Item ID {item_id} not found in training data."
                )
            indices.append(self.item2idx[item_id])
        return indices
