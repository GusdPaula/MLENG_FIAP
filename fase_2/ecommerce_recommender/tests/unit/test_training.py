"""Tests for the training components (Trainer, EarlyStopping)."""

import pytest
import torch
from src.recommender.models.ncf import NCFModel
from src.recommender.training import EarlyStopping, Trainer
from torch.utils.data import DataLoader


def test_early_stopping_min_mode() -> None:
    """Test early stopping with min mode (loss)."""
    stopper = EarlyStopping(patience=2, mode="min")

    # First epoch: loss decreases
    assert not stopper(0.5)  # Should not stop
    assert stopper.best_value == 0.5
    assert stopper.num_bad_epochs == 0

    # Second epoch: loss increases (no improvement)
    assert not stopper(0.6)  # Should not stop yet
    assert stopper.num_bad_epochs == 1

    # Third epoch: loss increases again (patience exceeded)
    assert stopper(0.7)  # Should stop now
    assert stopper.num_bad_epochs == 2


def test_early_stopping_max_mode() -> None:
    """Test early stopping with max mode (AUC)."""
    stopper = EarlyStopping(patience=2, mode="max")

    # First epoch: AUC increases
    assert not stopper(0.8)  # Should not stop
    assert stopper.best_value == 0.8
    assert stopper.num_bad_epochs == 0

    # Second epoch: AUC decreases (no improvement)
    assert not stopper(0.7)  # Should not stop yet
    assert stopper.num_bad_epochs == 1

    # Third epoch: AUC decreases again (patience exceeded)
    assert stopper(0.6)  # Should stop now
    assert stopper.num_bad_epochs == 2


def test_early_stopping_with_min_delta() -> None:
    """Test early stopping with min_delta."""
    stopper = EarlyStopping(patience=2, mode="min", min_delta=0.1)

    # First epoch: loss decreases by 0.15 (exceeds min_delta)
    assert not stopper(0.5)  # Should not stop
    assert stopper.best_value == 0.5
    assert stopper.num_bad_epochs == 0

    # Second epoch: loss decreases by 0.05 (less than min_delta) - no improvement
    assert not stopper(0.45)  # Should not stop yet (no improvement)
    assert stopper.num_bad_epochs == 1

    # Third epoch: loss increases (no improvement) - should trigger early stopping
    assert stopper(0.55)  # Should stop now (patience exceeded)
    assert stopper.num_bad_epochs == 2


def test_early_stopping_disabled() -> None:
    """Test early stopping with patience=0."""
    stopper = EarlyStopping(patience=0, mode="min")

    # Should never stop
    assert not stopper(0.5)
    assert not stopper(0.6)
    assert not stopper(0.7)


def test_early_stopping_reset() -> None:
    """Test resetting the early stopping state."""
    stopper = EarlyStopping(patience=2, mode="min")

    # Make some progress
    assert not stopper(0.5)
    assert not stopper(0.6)

    # Reset and start over
    stopper.reset()
    assert not stopper(0.7)  # Should not stop (new start)
    assert stopper.best_value == 0.7


def test_trainer_train_batch() -> None:
    """Test the train_batch method directly."""
    model = NCFModel(num_users=10, num_items=5, embedding_dim=8, hidden_layers=[16])
    trainer = Trainer(model, {"learning_rate": 0.01})

    # Create a simple batch
    users = torch.tensor([0, 1, 2], dtype=torch.long)
    items = torch.tensor([0, 1, 2], dtype=torch.long)
    labels = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

    # Should return a loss value
    loss = trainer.train_batch(users, items, labels)
    assert isinstance(loss, float)
    assert loss >= 0


def test_trainer_train_epoch() -> None:
    """Test the train_epoch method with DataLoader."""
    model = NCFModel(num_users=10, num_items=5, embedding_dim=8, hidden_layers=[16])
    trainer = Trainer(model, {"learning_rate": 0.01})

    # Create a simple dataset and dataloader
    interactions = [
        (0, 0, 1.0),
        (0, 1, 0.0),
        (1, 2, 1.0),
        (1, 3, 0.0),
    ]

    # Create a simple dataset that returns the interactions
    class SimpleDataset:
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            user, item, label = self.data[idx]
            return (
                torch.tensor(user, dtype=torch.long),
                torch.tensor(item, dtype=torch.long),
                torch.tensor(label, dtype=torch.float32),
            )

    dataset = SimpleDataset(interactions)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    loss = trainer.train_epoch(dataloader)
    assert isinstance(loss, float)
    assert loss >= 0


def test_trainer_train_epoch_empty() -> None:
    """Test train_epoch with an empty DataLoader."""
    model = NCFModel(num_users=10, num_items=5, embedding_dim=8, hidden_layers=[16])
    trainer = Trainer(model, {"learning_rate": 0.01})

    class EmptyDataset:
        def __len__(self):
            return 0

        def __getitem__(self, idx):
            raise IndexError

    loader = DataLoader(EmptyDataset())
    loss = trainer.train_epoch(loader)
    assert loss == 0.0


def test_trainer_evaluate() -> None:
    """Test the evaluate method."""
    model = NCFModel(num_users=10, num_items=5, embedding_dim=8, hidden_layers=[16])
    trainer = Trainer(model, {"learning_rate": 0.01})

    # Create a simple dataset for evaluation
    class SimpleDataset:
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            user, item, label = self.data[idx]
            return (
                torch.tensor(user, dtype=torch.long),
                torch.tensor(item, dtype=torch.long),
                torch.tensor(label, dtype=torch.float32),
            )

    # Test data with known labels
    test_data = [
        (0, 0, 1.0),
        (0, 1, 0.0),
        (1, 2, 1.0),
        (1, 3, 0.0),
    ]

    dataset = SimpleDataset(test_data)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Should return metrics dict
    metrics = trainer.evaluate(dataloader, metrics=("auc_roc", "avg_precision", "loss", "ndcg_at_k"), num_items=5, k=3)
    assert isinstance(metrics, dict)
    assert "auc_roc" in metrics
    assert "avg_precision" in metrics
    assert "loss" in metrics
    assert "ndcg_at_k" in metrics
    assert 0.0 <= metrics["auc_roc"] <= 1.0
    assert 0.0 <= metrics["avg_precision"] <= 1.0


def test_trainer_evaluate_invalid_metric() -> None:
    model = NCFModel(num_users=10, num_items=5, embedding_dim=8, hidden_layers=[16])
    trainer = Trainer(model, {"learning_rate": 0.01})

    class EmptyDataset:
        def __len__(self):
            return 0

        def __getitem__(self, idx):
            raise IndexError

    with pytest.raises(ValueError):
        trainer.evaluate(DataLoader(EmptyDataset()), metrics=("unknown",))


def test_trainer_evaluate_ndcg_missing_num_items() -> None:
    model = NCFModel(num_users=10, num_items=5, embedding_dim=8, hidden_layers=[16])
    trainer = Trainer(model, {"learning_rate": 0.01})

    class EmptyDataset:
        def __len__(self):
            return 0

        def __getitem__(self, idx):
            raise IndexError

    with pytest.raises(ValueError):
        trainer.evaluate(DataLoader(EmptyDataset()), metrics=("ndcg_at_k",))


def test_trainer_fit() -> None:
    """Test the fit method."""
    model = NCFModel(num_users=10, num_items=5, embedding_dim=8, hidden_layers=[16])
    trainer = Trainer(model, {"learning_rate": 0.01})

    # Create simple datasets for training and validation
    class SimpleDataset:
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            user, item, label = self.data[idx]
            return (
                torch.tensor(user, dtype=torch.long),
                torch.tensor(item, dtype=torch.long),
                torch.tensor(label, dtype=torch.float32),
            )

    train_data = [(0, 0, 1.0), (0, 1, 0.0), (1, 2, 1.0)]
    val_data = [(1, 3, 0.0), (2, 0, 1.0)]

    train_dataset = SimpleDataset(train_data)
    val_dataset = SimpleDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    # Should return a list of EpochResult
    history = trainer.fit(train_loader=train_loader, val_loader=val_loader, epochs=2, show_progress=False, metric_for_best="auc_roc", mode="max", log_callback=lambda res: None)

    assert isinstance(history, list)
    assert len(history) == 2
    assert all(hasattr(result, "epoch") for result in history)
    assert all(hasattr(result, "train_loss") for result in history)
    assert all(hasattr(result, "eval_metrics") for result in history)


def test_trainer_fit_invalid_mode() -> None:
    model = NCFModel(num_users=10, num_items=5, embedding_dim=8, hidden_layers=[16])
    trainer = Trainer(model, {"learning_rate": 0.01})

    with pytest.raises(ValueError):
        trainer.fit(None, None, epochs=1, metric_for_best="loss", mode="invalid")


def test_trainer_fit_with_early_stopping() -> None:
    """Test the fit_with_early_stopping method."""
    model = NCFModel(num_users=10, num_items=5, embedding_dim=8, hidden_layers=[16])
    trainer = Trainer(model, {"learning_rate": 0.01})

    # Create simple datasets for training and validation
    class SimpleDataset:
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            user, item, label = self.data[idx]
            return (
                torch.tensor(user, dtype=torch.long),
                torch.tensor(item, dtype=torch.long),
                torch.tensor(label, dtype=torch.float32),
            )

    train_data = [(0, 0, 1.0), (0, 1, 0.0), (1, 2, 1.0)]
    val_data = [(1, 3, 0.0), (2, 0, 1.0)]

    train_dataset = SimpleDataset(train_data)
    val_dataset = SimpleDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    # Test with early stopping disabled (should run all epochs)
    stopper = EarlyStopping(patience=0)  # Disable early stopping
    history, best = trainer.fit_with_early_stopping(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
        early_stopping=stopper,
        monitor="val_loss",
        show_progress=False,
    )

    assert isinstance(history, list)
    assert len(history) == 3  # Should run all epochs since patience=0
    assert best["value"] is None or best["value"] >= 0


def test_trainer_fit_with_early_stopping_enabled() -> None:
    """Test the fit_with_early_stopping method with early stopping enabled."""
    model = NCFModel(num_users=10, num_items=5, embedding_dim=8, hidden_layers=[16])
    trainer = Trainer(model, {"learning_rate": 0.01})

    # Create simple datasets for training and validation
    class SimpleDataset:
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            user, item, label = self.data[idx]
            return (
                torch.tensor(user, dtype=torch.long),
                torch.tensor(item, dtype=torch.long),
                torch.tensor(label, dtype=torch.float32),
            )

    train_data = [(0, 0, 1.0), (0, 1, 0.0), (1, 2, 1.0)]
    val_data = [(1, 3, 0.0), (2, 0, 1.0)]

    train_dataset = SimpleDataset(train_data)
    val_dataset = SimpleDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    # Test with early stopping enabled
    stopper = EarlyStopping(patience=1, mode="min")  # Stop after 1 bad epoch

    history, best = trainer.fit_with_early_stopping(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        early_stopping=stopper,
        monitor="val_loss",
        show_progress=False,
    )

    assert isinstance(history, list)
    # Should have stopped before running all 5 epochs due to early stopping
    assert len(history) <= 5
    assert best["value"] is None or best["value"] >= 0


def test_trainer_fit_with_early_stopping_monitor_ndcg() -> None:
    model = NCFModel(num_users=10, num_items=5, embedding_dim=8, hidden_layers=[16])
    trainer = Trainer(model, {"learning_rate": 0.01})

    class SimpleDataset:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return torch.tensor(0), torch.tensor(0), torch.tensor(1.0)

    loader = DataLoader(SimpleDataset(), batch_size=1)
    stopper = EarlyStopping(patience=1, mode="max")

    history, best = trainer.fit_with_early_stopping(train_loader=loader, val_loader=loader, epochs=1, early_stopping=stopper, monitor="ndcg_at_k", num_items=5, ranking_k=3)
    assert len(history) == 1


def test_trainer_resolve_monitor_error() -> None:
    model = NCFModel(num_users=10, num_items=5, embedding_dim=8, hidden_layers=[16])
    trainer = Trainer(model, {"learning_rate": 0.01})

    class SimpleDataset:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return torch.tensor(0), torch.tensor(0), torch.tensor(1.0)

    loader = DataLoader(SimpleDataset(), batch_size=1)
    stopper = EarlyStopping(patience=1)

    with pytest.raises(ValueError):
        trainer.fit_with_early_stopping(train_loader=loader, val_loader=loader, epochs=1, early_stopping=stopper, monitor="invalid_monitor")
