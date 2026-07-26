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
        precision_at_k: Precision at K.
        recall_at_k: Recall at K.
        mrr_at_k: Mean Reciprocal Rank at K.
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
    precision_at_k: float
    recall_at_k: float
    mrr_at_k: float
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
    num_items = len(item2idx)

    logger.info(f"Training model={model_type}, processor={processor_name}")

    mlflow_toolkit.log_dataset(
        interactions,
        name=f"{processor_name}_interactions",
        source=str(processed_path),
        context="training",
    )

    train_loader, val_loader, dataset, val_dataset = _prepare_data_loaders(
        interactions, num_items, config, seed
    )

    model, history, best = _train_model_with_early_stopping(
        model_type, len(user2idx), num_items, config, train_loader, val_loader, mlflow_toolkit
    )

    best_loss, best_metrics, ranking = _compute_final_metrics(
        model, history, best, val_dataset, dataset, num_items, config, mlflow_toolkit
    )

    ranking_k = config.get("ranking_k", 10)
    checkpoint_path = _save_and_log_artifacts(
        model, model_type, processor_name, user2idx, item2idx, config,
        best_loss, best_metrics, ranking, best, history,
        artifact_dir, mlflow_toolkit, ranking_k,
    )

    return _build_experiment_result(
        model_type, processor_name, checkpoint_path, processed_path,
        best_loss, best_metrics, ranking, best, history,
    )


def _prepare_data_loaders(
    interactions: Any,
    num_items: int,
    config: ExperimentConfig,
    seed: int,
) -> tuple:
    """Create dataset, split it, and return train/val DataLoaders."""
    dataset = RecommenderDataset(
        interactions, num_items, num_negatives=config["num_negatives"],
    )

    train_size = int(config["train_split_ratio"] * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    device = resolve_device()
    logger.info(
        f"Device: {device} | samples={len(dataset):,} | "
        f"train={train_size:,} | val={val_size:,}"
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    return train_loader, val_loader, dataset, val_dataset


def _train_model_with_early_stopping(
    model_type: str,
    num_users: int,
    num_items: int,
    config: ExperimentConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    mlflow_toolkit: MLflowToolkit,
) -> tuple:
    """Create model, trainer, and run training with early stopping."""
    model = ModelFactory.create(
        model_type, num_users=num_users, num_items=num_items,
        **config.get("hyperparams", {}),
    )
    trainer = Trainer(model, config, device=resolve_device())

    early_stopping = EarlyStopping(
        patience=config["early_stopping_patience"],
        mode=config["early_stopping_mode"],
        min_delta=config["early_stopping_min_delta"],
    )

    history, best = trainer.fit_with_early_stopping(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["epochs"],
        early_stopping=early_stopping,
        monitor=config.get("early_stopping_monitor", "auc_roc"),
        show_progress=config.get("show_progress", True),
        log_callback=create_mlflow_logger(mlflow_toolkit),
        num_items=num_items,
        ranking_k=config.get("ranking_k", 10),
    )

    return model, history, best


def _compute_final_metrics(
    model: Any,
    history: list,
    best: dict,
    val_dataset: Any,
    dataset: Any,
    num_items: int,
    config: ExperimentConfig,
    mlflow_toolkit: MLflowToolkit,
) -> tuple:
    """Extract best-epoch metrics and compute ranking metrics."""
    best_result = next(r for r in history if r.epoch == best["epoch"])
    best_loss = float(best_result.train_loss)
    best_metrics = best_result.eval_metrics

    mlflow_toolkit.log_metrics({
        "best_epoch": int(best["epoch"]),
        "best_auc_roc": float(best["value"]),
        "epochs_run": len(history),
    })

    ranking_k = config.get("ranking_k", 10)
    ranking = compute_ranking_metrics(
        model=model,
        val_dataset=val_dataset,
        dataset=dataset,
        num_items=num_items,
        device=resolve_device(),
        k=ranking_k,
        sample_limit=config.get("ranking_sample_limit", 10000),
        positive_limit=config.get("ranking_positive_limit", 1000),
    )

    return best_loss, best_metrics, ranking


def _save_and_log_artifacts(
    model: Any,
    model_type: str,
    processor_name: str,
    user2idx: dict,
    item2idx: dict,
    config: ExperimentConfig,
    best_loss: float,
    best_metrics: dict,
    ranking: Any,
    best: dict,
    history: list,
    artifact_dir: Path,
    mlflow_toolkit: MLflowToolkit,
    ranking_k: int,
) -> Path:
    """Save checkpoint and log final metrics/artifacts to MLflow."""
    metrics = {
        "loss": best_loss,
        "auc_roc": float(best_metrics["auc_roc"]),
        "avg_precision": float(best_metrics["avg_precision"]),
        **ranking.to_dict(ranking_k),
    }

    checkpoint_path = save_checkpoint(
        model=model, model_type=model_type, processor_name=processor_name,
        user2idx=user2idx, item2idx=item2idx, config=config, metrics=metrics,
        early_stopping_info={
            "best_epoch": best["epoch"],
            "best_auc_roc": best["value"],
            "epochs_run": len(history),
        },
        artifact_dir=artifact_dir,
    )

    mlflow_toolkit.log_artifact(checkpoint_path)
    mlflow_toolkit.log_metrics({
        "final_train_loss": best_loss,
        "final_auc_roc": float(best_metrics["auc_roc"]),
        "final_avg_precision": float(best_metrics["avg_precision"]),
        **ranking.to_dict(ranking_k),
    })

    logger.info(f"Experiment completed. Artifact saved to: {checkpoint_path}")
    return checkpoint_path


def _build_experiment_result(
    model_type: str,
    processor_name: str,
    checkpoint_path: Path,
    processed_path: Any,
    best_loss: float,
    best_metrics: dict,
    ranking: Any,
    best: dict,
    history: list,
) -> ExperimentResult:
    """Assemble the ExperimentResult dictionary."""
    return {
        "model_type": model_type,
        "processor": processor_name,
        "artifact": str(checkpoint_path),
        "processed_data": str(processed_path),
        "train_loss": best_loss,
        "auc_roc": float(best_metrics["auc_roc"]),
        "avg_precision": float(best_metrics["avg_precision"]),
        "hit_rate_at_k": ranking.hit_rate,
        "ndcg_at_k": ranking.ndcg,
        "precision_at_k": ranking.precision,
        "recall_at_k": ranking.recall,
        "mrr_at_k": ranking.mrr,
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
    }

