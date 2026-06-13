"""Experiment orchestration for training recommender models."""

import logging
from pathlib import Path
from typing import Any, Dict, TypedDict

import torch
from torch.utils.data import DataLoader, random_split

from ..data import RecommenderDataset
from ..mlflow_toolkit import MLflowToolkit, create_mlflow_logger
from ..models import ModelFactory
from ..utils import resolve_device
from .checkpoint import save_checkpoint
from .early_stopping import EarlyStopping
from .evaluator import compute_ranking_metrics
from .trainer import Trainer

logger = logging.getLogger(__name__)


class ExperimentConfig(TypedDict, total=False):
    """Configuration for training experiments.

    Attributes:
        batch_size: Batch size for training.
        epochs: Maximum number of training epochs.
        learning_rate: Learning rate for optimizer.
        num_negatives: Number of negative samples per positive.
        show_progress: Whether to show training progress.
        hyperparams: Model-specific hyperparameters.
        early_stopping_patience: Patience for early stopping.
        early_stopping_min_delta: Minimum delta for early stopping.
        early_stopping_mode: Mode for early stopping ('min' or 'max').
        early_stopping_monitor: Metric to monitor for early stopping (e.g., 'auc_roc', 'ndcg_at_10').
        train_split_ratio: Ratio of training data (0.0-1.0).
        ranking_k: K value for ranking metrics.
        ranking_sample_limit: Max samples for ranking evaluation.
        ranking_positive_limit: Max positive samples for ranking.
    """

    batch_size: int
    epochs: int
    learning_rate: float
    num_negatives: int
    show_progress: bool
    hyperparams: Dict[str, Any]
    early_stopping_patience: int
    early_stopping_min_delta: float
    early_stopping_mode: str
    early_stopping_monitor: str
    train_split_ratio: float
    ranking_k: int
    ranking_sample_limit: int
    ranking_positive_limit: int


class ExperimentResult(TypedDict):
    """Result of a training experiment.

    Attributes:
        model_type: Type of model trained.
        processor: Name of data processor used.
        artifact: Path to saved model checkpoint.
        processed_data: Path to processed dataset.
        train_loss: Final training loss.
        auc_roc: AUC-ROC score.
        avg_precision: Average precision score.
        hit_rate_at_k: Hit rate at K.
        ndcg_at_k: NDCG at K.
        best_epoch: Best epoch number.
        epochs_run: Total epochs run.
    """

    model_type: str
    processor: str
    artifact: str
    processed_data: str
    train_loss: float
    auc_roc: float
    avg_precision: float
    hit_rate_at_k: float
    ndcg_at_k: float
    best_epoch: int
    epochs_run: int


def train_one_experiment(
    processor_data: Dict[str, Any],
    model_type: str,
    processor_name: str,
    config: ExperimentConfig,
    mlflow_toolkit: MLflowToolkit,
    artifact_dir: Path,
    seed: int,
) -> ExperimentResult:
    """Train a single model experiment with MLflow logging.

    Args:
        processor_data: Dictionary containing interactions, user2idx, item2idx, and path.
        model_type: Type of model to train (e.g., "ncf", "gmf", "matrix_factorization").
        processor_name: Name of the data processor used.
        config: Training configuration dictionary.
        mlflow_toolkit: MLflowToolkit instance for logging.
        artifact_dir: Directory to save model artifacts.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing experiment results and metrics.
    """
    interactions = processor_data["interactions"]
    user2idx = processor_data["user2idx"]
    item2idx = processor_data["item2idx"]
    processed_path = processor_data["path"]

    logger.info(f"Training model={model_type}, processor={processor_name}")

    # Log dataset to MLflow
    mlflow_toolkit.log_dataset(
        interactions,
        name=f"{processor_name}_interactions",
        source=str(processed_path),
        context="training",
    )

    num_users = len(user2idx)
    num_items = len(item2idx)

    # Create dataset
    dataset = RecommenderDataset(
        interactions,
        num_items,
        num_negatives=config["num_negatives"],
    )

    # Split dataset
    train_size = int(config["train_split_ratio"] * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
    )

    # Resolve device
    device = resolve_device()

    logger.info(
        f"Device: {device} | "
        f"samples={len(dataset):,} | "
        f"train={train_size:,} | "
        f"val={val_size:,}"
    )

    # Create model
    model = ModelFactory.create(
        model_type,
        num_users=num_users,
        num_items=num_items,
        **config.get("hyperparams", {}),
    )

    # Create trainer
    trainer = Trainer(model, config, device=device)

    # Create early stopping
    early_stopping = EarlyStopping(
        patience=config["early_stopping_patience"],
        mode=config["early_stopping_mode"],
        min_delta=config["early_stopping_min_delta"],
    )

    # Create MLflow callback
    mlflow_logger = create_mlflow_logger(mlflow_toolkit)

    # Train with early stopping
    history, best = trainer.fit_with_early_stopping(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["epochs"],
        early_stopping=early_stopping,
        monitor=config.get("early_stopping_monitor", "auc_roc"),
        show_progress=config.get("show_progress", True),
        log_callback=mlflow_logger,
        num_items=num_items,
        ranking_k=config.get("ranking_k", 10),
    )

    # Extract best epoch metrics
    best_result = next(r for r in history if r.epoch == best["epoch"])
    best_loss = float(best_result.train_loss)
    best_metrics = best_result.eval_metrics

    # Log best epoch metrics to MLflow
    mlflow_toolkit.log_metrics(
        {
            "best_epoch": int(best["epoch"]),
            "best_auc_roc": float(best["value"]),
            "epochs_run": len(history),
        }
    )

    # Compute ranking metrics
    hr, ndcg = compute_ranking_metrics(
        model=model,
        val_dataset=val_dataset,
        dataset=dataset,
        num_items=num_items,
        device=device,
        k=config.get("ranking_k", 10),
        sample_limit=config.get("ranking_sample_limit", 10000),
        positive_limit=config.get("ranking_positive_limit", 1000),
    )

    # Save checkpoint
    metrics = {
        "loss": best_loss,
        "auc_roc": float(best_metrics["auc_roc"]),
        "avg_precision": float(best_metrics["avg_precision"]),
        "hit_rate_at_10": float(hr),
        "ndcg_at_10": float(ndcg),
    }

    early_stopping_info = {
        "best_epoch": best["epoch"],
        "best_auc_roc": best["value"],
        "epochs_run": len(history),
    }

    checkpoint_path = save_checkpoint(
        model=model,
        model_type=model_type,
        processor_name=processor_name,
        user2idx=user2idx,
        item2idx=item2idx,
        config=config,
        metrics=metrics,
        early_stopping_info=early_stopping_info,
        artifact_dir=artifact_dir,
    )

    # Log artifact and final metrics to MLflow
    mlflow_toolkit.log_artifact(checkpoint_path)

    mlflow_toolkit.log_metrics(
        {
            "final_train_loss": best_loss,
            "final_auc_roc": float(best_metrics["auc_roc"]),
            "final_avg_precision": float(best_metrics["avg_precision"]),
            "hit_rate_at_10": float(hr),
            "ndcg_at_10": float(ndcg),
        }
    )

    # Build result dictionary
    result: ExperimentResult = {
        "model_type": model_type,
        "processor": processor_name,
        "artifact": str(checkpoint_path),
        "processed_data": str(processed_path),
        "train_loss": best_loss,
        "auc_roc": float(best_metrics["auc_roc"]),
        "avg_precision": float(best_metrics["avg_precision"]),
        "hit_rate_at_k": float(hr),
        "ndcg_at_k": float(ndcg),
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
    }

    logger.info(f"Experiment completed. Artifact saved to: {checkpoint_path}")

    return result
