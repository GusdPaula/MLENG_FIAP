"""Device resolution utilities."""

import warnings

import torch


def resolve_device() -> str:
    """Resolve the appropriate device for PyTorch operations.

    Returns:
        "cuda" if CUDA is available, otherwise "cpu".
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cuda_available = torch.cuda.is_available()
    return "cuda" if cuda_available else "cpu"
