"""Unit tests for API exceptions.

Tests custom exception classes used in the API module.
"""

import pytest
from api.exceptions import (
    InvalidInputError,
    ModelLoadError,
    PredictionError,
    PredictorNotFoundError,
)


class TestInvalidInputError:
    """Tests for InvalidInputError exception."""

    def test_invalid_input_error_creation(self):
        """Test creating an InvalidInputError."""
        error = InvalidInputError("Invalid user ID provided")
        assert str(error) == "Invalid user ID provided"
        assert isinstance(error, PredictionError)

    def test_invalid_input_error_inheritance(self):
        """Test that InvalidInputError inherits from PredictionError."""
        error = InvalidInputError("Test error")
        assert isinstance(error, PredictionError)


class TestModelLoadError:
    """Tests for ModelLoadError exception."""

    def test_model_load_error_creation(self):
        """Test creating a ModelLoadError."""
        error = ModelLoadError("Failed to load model from path")
        assert str(error) == "Failed to load model from path"
        assert isinstance(error, PredictionError)

    def test_model_load_error_inheritance(self):
        """Test that ModelLoadError inherits from PredictionError."""
        error = ModelLoadError("Test error")
        assert isinstance(error, PredictionError)


class TestPredictorNotFoundError:
    """Tests for PredictorNotFoundError exception."""

    def test_predictor_not_found_error_creation(self):
        """Test creating a PredictorNotFoundError."""
        error = PredictorNotFoundError("Predictor type 'invalid_type' not found")
        assert str(error) == "Predictor type 'invalid_type' not found"
        assert isinstance(error, PredictionError)

    def test_predictor_not_found_error_inheritance(self):
        """Test that PredictorNotFoundError inherits from PredictionError."""
        error = PredictorNotFoundError("Test error")
        assert isinstance(error, PredictionError)


class TestPredictionError:
    """Tests for base PredictionError exception."""

    def test_prediction_error_creation(self):
        """Test creating a generic PredictionError."""
        error = PredictionError("Generic prediction error")
        assert str(error) == "Generic prediction error"

    def test_prediction_error_is_exception(self):
        """Test that PredictionError is an Exception."""
        error = PredictionError("Test error")
        assert isinstance(error, Exception)


class TestExceptionRaising:
    """Tests for raising and catching API exceptions."""

    def test_raise_invalid_input_error(self):
        """Test raising and catching InvalidInputError."""
        with pytest.raises(InvalidInputError) as exc_info:
            raise InvalidInputError("Invalid input")
        assert str(exc_info.value) == "Invalid input"

    def test_raise_model_load_error(self):
        """Test raising and catching ModelLoadError."""
        with pytest.raises(ModelLoadError) as exc_info:
            raise ModelLoadError("Model load failed")
        assert str(exc_info.value) == "Model load failed"

    def test_raise_predictor_not_found_error(self):
        """Test raising and catching PredictorNotFoundError."""
        with pytest.raises(PredictorNotFoundError) as exc_info:
            raise PredictorNotFoundError("Predictor not found")
        assert str(exc_info.value) == "Predictor not found"

    def test_catch_base_prediction_error(self):
        """Test catching specific exceptions via base class."""
        with pytest.raises(PredictionError):
            raise InvalidInputError("Invalid input")

        with pytest.raises(PredictionError):
            raise ModelLoadError("Model load failed")

        with pytest.raises(PredictionError):
            raise PredictorNotFoundError("Predictor not found")
