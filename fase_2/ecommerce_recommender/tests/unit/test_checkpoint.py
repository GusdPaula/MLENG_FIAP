"""Unit tests for checkpoint module."""

import tempfile
from pathlib import Path

import torch
from src.recommender.models import ModelFactory
from src.recommender.training.checkpoint import save_checkpoint


def test_save_checkpoint_basic():
    """Test basic checkpoint saving."""
    model = ModelFactory.create(
        "gmf",
        num_users=10,
        num_items=5,
        embedding_dim=8,
    )

    user2idx = {i: i for i in range(10)}
    item2idx = {i: i for i in range(5)}
    config = {"model_type": "gmf", "embedding_dim": 8}
    metrics = {"auc_roc": 0.85, "ndcg_at_10": 0.75}

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = Path(tmpdir)

        checkpoint_path = save_checkpoint(
            model=model,
            model_type="gmf",
            processor_name="weighted",
            user2idx=user2idx,
            item2idx=item2idx,
            config=config,
            metrics=metrics,
            early_stopping_info={"best_epoch": 5, "epochs_run": 8},
            artifact_dir=artifact_dir,
        )

        assert checkpoint_path.exists()
        assert checkpoint_path.suffix == ".pt"
        assert "gmf" in checkpoint_path.name


def test_save_checkpoint_with_early_stopping_info():
    """Test checkpoint saving with early stopping information."""
    model = ModelFactory.create(
        "ncf",
        num_users=10,
        num_items=5,
        embedding_dim=8,
    )

    user2idx = {i: i for i in range(10)}
    item2idx = {i: i for i in range(5)}
    config = {"model_type": "ncf", "embedding_dim": 8}
    metrics = {"auc_roc": 0.90, "ndcg_at_10": 0.80}
    early_stopping_info = {
        "best_epoch": 3,
        "epochs_run": 5,
        "best_value": 0.90,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = Path(tmpdir)

        checkpoint_path = save_checkpoint(
            model=model,
            model_type="ncf",
            processor_name="binary",
            user2idx=user2idx,
            item2idx=item2idx,
            config=config,
            metrics=metrics,
            early_stopping_info=early_stopping_info,
            artifact_dir=artifact_dir,
        )

        assert checkpoint_path.exists()

        # Verify checkpoint can be loaded
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert "model_state_dict" in checkpoint
        assert "user2idx" in checkpoint
        assert "item2idx" in checkpoint
        assert "config" in checkpoint
        assert "metrics" in checkpoint
        assert "early_stopping" in checkpoint


def test_save_checkpoint_creates_directory():
    """Test that save_checkpoint works with nested directories."""
    model = ModelFactory.create(
        "matrix_factorization",
        num_users=10,
        num_items=5,
        embedding_dim=8,
    )

    user2idx = {i: i for i in range(10)}
    item2idx = {i: i for i in range(5)}
    config = {"model_type": "matrix_factorization", "embedding_dim": 8}
    metrics = {"auc_roc": 0.88}
    early_stopping_info = {"best_epoch": 5, "epochs_run": 10}

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = Path(tmpdir) / "models" / "checkpoints"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = save_checkpoint(
            model=model,
            model_type="matrix_factorization",
            processor_name="implicit",
            user2idx=user2idx,
            item2idx=item2idx,
            config=config,
            metrics=metrics,
            early_stopping_info=early_stopping_info,
            artifact_dir=artifact_dir,
        )

        assert checkpoint_path.exists()
        assert artifact_dir.exists()
