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

from ..mlflow_toolkit import MLflowToolkit, create_mlflow_logger

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

    # Load MLflow config if available
    mlflow_config_path = Path(config_path).parent / "mlflow.yaml"
    mlflow_cfg = {}
    if mlflow_config_path.exists():
        with open(mlflow_config_path) as f:
            mlflow_cfg = yaml.safe_load(f).get("mlflow", {})

    import os
    # Optimize CPU threads for PyTorch operations
    num_threads = cfg.get("num_threads", os.cpu_count() or 1)
    torch.set_num_threads(num_threads)
    logger.info("Configuring PyTorch to use %d CPU threads (intra-op)", num_threads)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # --- 1. Dados: carrega eventos processados ou brutos ----------------
    processed_dir = Path("data/processed")
    interactions_path = processed_dir / "interactions.csv"
    user2idx_path = processed_dir / "user2idx.json"
    item2idx_path = processed_dir / "item2idx.json"

    import json
    import pandas as pd

    if interactions_path.exists() and user2idx_path.exists() and item2idx_path.exists():
        logger.info("Dados pré-processados encontrados. Carregando de %s ...", processed_dir)
        interactions = pd.read_csv(interactions_path)
        with open(user2idx_path) as f:
            user2idx = json.load(f)
        with open(item2idx_path) as f:
            item2idx = json.load(f)
        # Convert keys back to integers (JSON keys are always strings)
        user2idx = {int(k): v for k, v in user2idx.items()}
        item2idx = {int(k): v for k, v in item2idx.items()}
        processor_cfg = cfg.get("processor", "weighted")
        processor_name = processor_cfg
        num_users = len(user2idx)
        num_items = len(item2idx)
        raw_path = cfg.get("raw_events_path", "data/raw/events.csv")
    else:
        logger.info("Dados pré-processados não encontrados. Carregando dados brutos...")
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
        processor_name = processor.strategy_name
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

    # Configure DataLoader parallel workers
    num_workers = cfg.get("num_workers", min(4, os.cpu_count() or 1))
    logger.info("Configuring DataLoader to use %d workers", num_workers)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        num_workers=num_workers,
        pin_memory=True
    )

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

    # --- MLflow Setup ------------------------------------------------
    mlflow_toolkit = MLflowToolkit(
        tracking_uri=mlflow_cfg.get("tracking_uri"),
        experiment_name=mlflow_cfg.get("experiment_name", "ecommerce_recommender"),
        registry_uri=mlflow_cfg.get("registry_uri"),
    )
    mlflow_toolkit.setup()

    run_name = f"{model_type}-{processor_cfg}"
    with mlflow_toolkit.start_run(
        run_name=run_name,
        tags={"model_type": model_type, "processor": processor_cfg},
    ):
        # Log MLflow parameters
        mlflow_toolkit.log_params({
            "model_type": model_type,
            "processor": processor_cfg,
            "seed": cfg["seed"],
            "batch_size": cfg["batch_size"],
            "learning_rate": cfg["learning_rate"],
            "epochs": cfg["epochs"],
            "num_negatives": cfg["num_negatives"],
            "min_interactions": cfg.get("min_interactions", 1),
        })

        # Log MLflow dataset
        mlflow_toolkit.log_dataset(
            interactions,
            name=f"{processor_cfg}_interactions",
            source=str(raw_path),
            context="training",
        )

        # --- 6. Training loop --------------------------------------------
        trainer = Trainer(model, cfg, device=device)

        early_stopping_cfg = cfg.get("early_stopping", {}) or {}
        use_early_stopping = bool(early_stopping_cfg.get("enabled", False))

        logger.info("\nStarting training (%d epochs)...", cfg["epochs"])
        logger.info("-" * 60)

        last_epoch = cfg["epochs"]
        metrics: dict = {}
        train_loss = 0.0

        mlflow_logger = create_mlflow_logger(mlflow_toolkit)

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
                log_callback=mlflow_logger,
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

            # Log early stopping best epoch details to MLflow
            mlflow_toolkit.log_metrics({
                "best_epoch": int(best["epoch"]),
                "best_auc_roc": float(best["value"]) if monitor == "auc_roc" else 0.0,
                "epochs_run": last_epoch,
            })
        else:
            history = trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=cfg["epochs"],
                show_progress=cfg.get("show_progress", True),
                log_callback=mlflow_logger,
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

        # --- 8. Persist model locally ------------------------------------
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
            processor_name=processor_name,
            user2idx=user2idx,
            item2idx=item2idx,
            config=cfg,
            metrics=metrics_dict,
            early_stopping_info=early_stopping_info,
            artifact_dir=artifact_dir,
        )

        logger.info("\nModelo salvo localmente em %s", artifact_path)

        # Copiar para um caminho fixo para rastreamento no pipeline DVC
        import shutil
        dvc_model_path = artifact_dir / "model.pt"
        shutil.copyfile(artifact_path, dvc_model_path)
        logger.info("Cópia do modelo salva em %s para rastreamento DVC", dvc_model_path)

        # --- 9. Log Metrics and PyTorch Model to MLflow Server -----------
        mlflow_toolkit.log_metrics({
            "final_train_loss": train_loss,
            "final_auc_roc": metrics["auc_roc"],
            "final_avg_precision": metrics["avg_precision"],
            "hit_rate_at_10": hr,
            "ndcg_at_10": ndcg,
        })

        logger.info("Logging PyTorch model to MLflow server...")
        mlflow_toolkit.log_pytorch_model(
            model=model,
            artifact_path="model",
            registered_model_name=cfg.get("registered_model_name", "ecommerce_recommender"),
        )
        logger.info("MLflow logging completed successfully.")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run training pipeline.")
    parser.add_argument(
        "--config",
        default="configs/model.yaml",
        help="Path to the model configuration YAML file.",
    )
    args = parser.parse_args()
    run_training_pipeline(config_path=args.config)
