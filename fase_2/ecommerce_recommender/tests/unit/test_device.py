"""Tests for device utilities."""

from unittest.mock import patch

from src.recommender.utils.device import resolve_device


def test_resolve_device_cuda_available():
    """Test device resolution when CUDA is available."""
    with patch("torch.cuda.is_available", return_value=True):
        device = resolve_device()
        assert device == "cuda"


def test_resolve_device_cuda_unavailable():
    """Test device resolution when CUDA is unavailable."""
    with patch("torch.cuda.is_available", return_value=False):
        device = resolve_device()
        assert device == "cpu"
