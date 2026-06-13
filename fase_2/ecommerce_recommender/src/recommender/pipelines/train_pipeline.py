"""End-to-end training pipeline.

The pipeline is fully driven by configuration. Two design patterns
make the rest of the system pluggable:

* **Model Factory** - :class:`recommender.models.ModelFactory` produces
  the recommender model from a string identifier (``model.type`` in
  the YAML). Adding a new model is a one-line registration.
* **Strategy** - :class:`recommender.data.DataProcessorContext` selects
  a data-processing strategy (``model.processor`` in the YAML). The
  pipeline only calls ``context.process(...)`` and is unaware of the
  concrete processor.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, random_split

from ..data import (
    DataProcessorContext,
    RecommenderDataset,
    load_events,
)
from ..models import ModelFactory
from ..training import EarlyStopping, Trainer
from ..training.checkpoint import save_checkpoint
from ..training.evaluator import compute_ranking_metrics
from ..utils import resolve_device

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def run_training_pipeline(config_path: str = "configs/model.yaml") -> None:
    """Train a recommender model end-to-end using the config file.

    Args:
        config_path: Path to the model configuration YAML file. Defaults to "configs/model.yaml".
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)["model"]

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # --- 1. Data: load raw events -------------------------------------
    raw_path = cfg.get("raw_events_path", "data/raw/events.csv")
    logger.info("Loading events from %s ...", raw_path)
    events = load_events(raw_path)
    logger.info("  Total events: %d", len(events))

    # --- 2. Data: pick a processing strategy --------------------------
    processor_cfg = cfg.get("processor", "weighted")
    processor_kwargs = cfg.get("processor_kwargs", {}) or {}
    processor = DataProcessorContext(processor_cfg, **processor_kwargs)
    logger.info("Data processor: %s", processor.strategy_name)

    interactions, user2idx, item2idx = processor.process(
        events, min_interactions=cfg.get("min_interactions", 1)
    )
    num_users = len(user2idx)
    num_items = len(item2idx)
    logger.info("  Users: %d, Items: %d", num_users, num_items)

    # --- 3. Dataset with negative sampling ----------------------------
    logger.info("Generating dataset with negative sampling...")
    dataset = RecommenderDataset(
        interactions, num_items, num_negatives=cfg["num_negatives"]
    )
    logger.info("  Total samples: %d", len(dataset))

    # --- 4. Train/val split ------------------------------------------
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"])

    # --- 5. Model: use the factory -----------------------------------
    device = resolve_device()
    logger.info("Device: %s", device)

    model_type = cfg.get("type", "ncf")
    model = ModelFactory.create(
        model_type,
        num_users=num_users,
        num_items=num_items,
        **cfg.get("hyperparams", {}),
    )
    logger.info("Model: %s (%s)", model_type, model.model_name)

    # --- 6. Training loop --------------------------------------------
    trainer = Trainer(model, cfg, device=device)

    early_stopping_cfg = cfg.get("early_stopping", {}) or {}
    use_early_stopping = bool(early_stopping_cfg.get("enabled", False))

    logger.info("\nStarting training (%d epochs)...", cfg["epochs"])
    logger.info("-" * 60)

    last_epoch = cfg["epochs"]
    metrics: dict = {}
    train_loss = 0.0

    if use_early_stopping:
        stopper = EarlyStopping(
            patience=int(early_stopping_cfg.get("patience", 3)),
            mode=early_stopping_cfg.get("mode", "min"),
            min_delta=float(early_stopping_cfg.get("min_delta", 0.0)),
        )
        monitor = early_stopping_cfg.get("monitor", "val_loss")
        ranking_k = cfg.get("ranking_k", 10)

        # Pass num_items and ranking_k if monitoring NDCG
        history, best = trainer.fit_with_early_stopping(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=cfg["epochs"],
            early_stopping=stopper,
            monitor=monitor,
            show_progress=cfg.get("show_progress", True),
            num_items=num_items if monitor.startswith("ndcg") else None,
            ranking_k=ranking_k,
        )
        last_epoch = len(history)
        if history:
            metrics = history[-1].eval_metrics
            train_loss = history[-1].train_loss
        logger.info(
            "Early stopping (monitor=%s, mode=%s) - best=%s @ epoch %s | ran %d/%d epochs",
            monitor,
            stopper.mode,
            best["value"],
            best["epoch"],
            last_epoch,
            cfg["epochs"],
        )
    else:
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=cfg["epochs"],
            show_progress=cfg.get("show_progress", True),
        )
        if history:
            metrics = history[-1].eval_metrics
            train_loss = history[-1].train_loss
            logger.info(
                "Epoch %02d/%d | Loss: %.4f | AUC: %.4f | AP: %.4f",
                last_epoch,
                cfg["epochs"],
                train_loss,
                metrics["auc_roc"],
                metrics["avg_precision"],
            )

    # --- 7. Ranking metrics on the validation set ---------------------
    logger.info("-" * 60)
    logger.info("Calculando métricas de ranking...")

    hr, ndcg = compute_ranking_metrics(
        model=model,
        val_dataset=val_dataset,
        dataset=dataset,
        num_items=num_items,
        device=device,
        k=10,
        sample_limit=10000,
        positive_limit=1000,
    )

    logger.info("  Hit Rate@10: %.4f", hr)
    logger.info("  NDCG@10:     %.4f", ndcg)

    # --- 8. Persist model --------------------------------------------
    artifact_dir = Path(cfg.get("artifact_dir", "models"))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    metrics_dict: Dict[str, float] = {
        "loss": train_loss,
        "auc_roc": metrics["auc_roc"],
        "avg_precision": metrics["avg_precision"],
        "hit_rate_at_10": hr,
        "ndcg_at_10": ndcg,
    }

    early_stopping_info: Dict[str, Any] = {
        "best_epoch": last_epoch,
        "epochs_run": last_epoch,
    }

    artifact_path = save_checkpoint(
        model=model,
        model_type=model_type,
        processor_name=processor.strategy_name,
        user2idx=user2idx,
        item2idx=item2idx,
        config=cfg,
        metrics=metrics_dict,
        early_stopping_info=early_stopping_info,
        artifact_dir=artifact_dir,
    )

    logger.info("\nModelo salvo em %s", artifact_path)


if __name__ == "__main__":
    run_training_pipeline()
