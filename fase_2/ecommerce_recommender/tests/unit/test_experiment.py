"""Tests for experiment orchestration."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from src.recommender.training.experiment import (
    _build_experiment_result,
    _compute_final_metrics,
    _prepare_data_loaders,
    _save_and_log_artifacts,
    _train_model_with_early_stopping,
    train_one_experiment,
)


@pytest.fixture
def mock_processor_data():
    return {
        "interactions": MagicMock(),
        "user2idx": {1: 0, 2: 1},
        "item2idx": {10: 0, 20: 1, 30: 2},
        "path": Path("/dummy/path"),
    }


@pytest.fixture
def mock_config():
    return {
        "batch_size": 16,
        "epochs": 2,
        "learning_rate": 0.01,
        "num_negatives": 4,
        "show_progress": False,
        "hyperparams": {"embedding_dim": 8},
        "early_stopping_patience": 2,
        "early_stopping_min_delta": 0.001,
        "early_stopping_mode": "max",
        "early_stopping_monitor": "auc_roc",
        "train_split_ratio": 0.8,
        "ranking_k": 10,
        "ranking_sample_limit": 100,
        "ranking_positive_limit": 10,
    }


@pytest.fixture
def mock_mlflow_toolkit():
    toolkit = MagicMock()
    return toolkit


@patch("src.recommender.training.experiment.resolve_device")
@patch("src.recommender.training.experiment.random_split")
@patch("src.recommender.training.experiment.RecommenderDataset")
def test_prepare_data_loaders(mock_dataset_class, mock_random_split, mock_resolve_device, mock_config):
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 100
    mock_dataset_class.return_value = mock_dataset

    mock_train_dataset = MagicMock()
    mock_train_dataset.__len__.return_value = 80
    mock_val_dataset = MagicMock()
    mock_val_dataset.__len__.return_value = 20
    mock_random_split.return_value = (mock_train_dataset, mock_val_dataset)

    mock_resolve_device.return_value = "cpu"

    interactions = MagicMock()
    train_loader, val_loader, dataset, val_dataset = _prepare_data_loaders(interactions, num_items=10, config=mock_config, seed=42)

    assert dataset == mock_dataset
    assert val_dataset == mock_val_dataset
    assert train_loader.batch_size == 16
    assert val_loader.batch_size == 16


@patch("src.recommender.training.experiment.create_mlflow_logger")
@patch("src.recommender.training.experiment.Trainer")
@patch("src.recommender.training.experiment.ModelFactory")
def test_train_model_with_early_stopping(
    mock_model_factory,
    mock_trainer_class,
    mock_create_logger,
    mock_config,
    mock_mlflow_toolkit,
):
    mock_model = MagicMock()
    mock_model_factory.create.return_value = mock_model

    mock_trainer = MagicMock()
    mock_trainer_class.return_value = mock_trainer

    mock_history = [MagicMock()]
    mock_best = {"epoch": 1, "value": 0.9}
    mock_trainer.fit_with_early_stopping.return_value = (mock_history, mock_best)

    train_loader = MagicMock()
    val_loader = MagicMock()

    model, history, best = _train_model_with_early_stopping(
        model_type="gmf",
        num_users=20,
        num_items=10,
        config=mock_config,
        train_loader=train_loader,
        val_loader=val_loader,
        mlflow_toolkit=mock_mlflow_toolkit,
    )

    assert model == mock_model
    assert history == mock_history
    assert best == mock_best


@patch("src.recommender.training.experiment.resolve_device")
@patch("src.recommender.training.experiment.compute_ranking_metrics")
def test_compute_final_metrics(mock_compute_ranking, mock_resolve_device, mock_config, mock_mlflow_toolkit):
    mock_model = MagicMock()
    mock_val_dataset = MagicMock()
    mock_dataset = MagicMock()

    mock_history_entry = MagicMock()
    mock_history_entry.epoch = 1
    mock_history_entry.train_loss = 0.5
    mock_history_entry.eval_metrics = {"auc_roc": 0.9, "avg_precision": 0.85}
    history = [mock_history_entry]

    best = {"epoch": 1, "value": 0.9}

    mock_ranking = MagicMock()
    mock_compute_ranking.return_value = mock_ranking

    best_loss, best_metrics, ranking = _compute_final_metrics(
        model=mock_model,
        history=history,
        best=best,
        val_dataset=mock_val_dataset,
        dataset=mock_dataset,
        num_items=10,
        config=mock_config,
        mlflow_toolkit=mock_mlflow_toolkit,
    )

    assert best_loss == 0.5
    assert best_metrics["auc_roc"] == 0.9
    assert ranking == mock_ranking
    mock_mlflow_toolkit.log_metrics.assert_called_once()


@patch("src.recommender.training.experiment.save_checkpoint")
def test_save_and_log_artifacts(mock_save_checkpoint, mock_config, mock_mlflow_toolkit):
    mock_model = MagicMock()
    mock_ranking = MagicMock()
    mock_ranking.to_dict.return_value = {"hit_rate_at_10": 0.5}

    mock_save_checkpoint.return_value = Path("/dummy/model.pt")

    checkpoint_path = _save_and_log_artifacts(
        model=mock_model,
        model_type="gmf",
        processor_name="proc",
        user2idx={},
        item2idx={},
        config=mock_config,
        best_loss=0.5,
        best_metrics={"auc_roc": 0.9, "avg_precision": 0.85},
        ranking=mock_ranking,
        best={"epoch": 1, "value": 0.9},
        history=[MagicMock()],
        artifact_dir=Path("/dummy"),
        mlflow_toolkit=mock_mlflow_toolkit,
        ranking_k=10,
    )

    assert checkpoint_path == Path("/dummy/model.pt")
    mock_mlflow_toolkit.log_artifact.assert_called_once_with(Path("/dummy/model.pt"))
    mock_mlflow_toolkit.log_metrics.assert_called_once()


def test_build_experiment_result():
    mock_ranking = MagicMock()
    mock_ranking.hit_rate = 0.5
    mock_ranking.ndcg = 0.4
    mock_ranking.precision = 0.3
    mock_ranking.recall = 0.2
    mock_ranking.mrr = 0.1

    result = _build_experiment_result(
        model_type="gmf",
        processor_name="proc",
        checkpoint_path=Path("/dummy/model.pt"),
        processed_path=Path("/dummy/data.csv"),
        best_loss=0.5,
        best_metrics={"auc_roc": 0.9, "avg_precision": 0.85},
        ranking=mock_ranking,
        best={"epoch": 1},
        history=[MagicMock()],
    )

    assert result["model_type"] == "gmf"
    assert result["train_loss"] == 0.5
    assert result["auc_roc"] == 0.9


@patch("src.recommender.training.experiment._prepare_data_loaders")
@patch("src.recommender.training.experiment._train_model_with_early_stopping")
@patch("src.recommender.training.experiment._compute_final_metrics")
@patch("src.recommender.training.experiment._save_and_log_artifacts")
@patch("src.recommender.training.experiment._build_experiment_result")
def test_train_one_experiment(
    mock_build,
    mock_save,
    mock_compute,
    mock_train,
    mock_prepare,
    mock_processor_data,
    mock_config,
    mock_mlflow_toolkit,
):
    mock_prepare.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
    mock_train.return_value = (MagicMock(), MagicMock(), MagicMock())
    mock_compute.return_value = (0.5, {}, MagicMock())
    mock_save.return_value = Path("/dummy/model.pt")
    mock_build.return_value = {"model_type": "gmf"}

    result = train_one_experiment(
        processor_data=mock_processor_data,
        model_type="gmf",
        processor_name="proc",
        config=mock_config,
        mlflow_toolkit=mock_mlflow_toolkit,
        artifact_dir=Path("/dummy"),
        seed=42,
    )

    assert result == {"model_type": "gmf"}
    mock_prepare.assert_called_once()
    mock_train.assert_called_once()
    mock_compute.assert_called_once()
    mock_save.assert_called_once()
    mock_build.assert_called_once()
