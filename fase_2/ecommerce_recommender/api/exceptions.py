"""Custom exceptions for the prediction API module."""


class PredictionError(Exception):
    """Base exception for prediction-related errors."""

    pass


class ModelLoadError(PredictionError):
    """Raised when a model fails to load."""

    pass


class InvalidInputError(PredictionError):
    """Raised when input data is invalid."""

    pass


class PredictorNotFoundError(PredictionError):
    """Raised when a requested predictor is not found."""

    pass
