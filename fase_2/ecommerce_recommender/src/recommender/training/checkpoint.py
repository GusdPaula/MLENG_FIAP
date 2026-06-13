"""Model checkpoint saving and loading utilities."""

import logging
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    model_type: str,
    processor_name: str,
    user2idx: Dict[Any, int],
    item2idx: Dict[Any, int],
    config: Dict[str, Any],
    metrics: Dict[str, float],
    early_stopping_info: Dict[str, Any],
    artifact_dir: Path,
) -> Path:
    """Save model checkpoint with all necessary metadata.

    Args:
        model: Trained PyTorch model.
        model_type: Type of model (e.g., "ncf", "gmf", "matrix_factorization").
        processor_name: Name of the data processor used.
        user2idx: User ID to index mapping.
        item2idx: Item ID to index mapping.
        config: Training configuration dictionary.
        metrics: Dictionary of evaluation metrics.
        early_stopping_info: Dictionary with early stopping information.
        artifact_dir: Directory to save the checkpoint.

    Returns:
        Path to the saved checkpoint file.
    """
    checkpoint_path = artifact_dir / f"{model_type}_{processor_name}.pt"

    checkpoint = {
        "model_type": model_type,
        "processor": processor_name,
        "model_state_dict": model.state_dict(),
        "user2idx": user2idx,
        "item2idx": item2idx,
        "config": config,
        "metrics": metrics,
        "early_stopping": early_stopping_info,
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved model checkpoint: {checkpoint_path}")

    return checkpoint_path


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Load model checkpoint from file.

    Args:
        checkpoint_path: Path to the checkpoint file.

    Returns:
        Dictionary containing checkpoint data.
    """
    logger.info(f"Loading model checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    return checkpoint
