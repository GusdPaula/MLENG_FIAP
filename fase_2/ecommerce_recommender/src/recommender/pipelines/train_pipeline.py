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
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, random_split

from ..data import (
    DataProcessorContext,
    RecommenderDataset,
    load_events,
)
from ..mlflow_toolkit import MLflowToolkit, create_mlflow_logger
from ..models import ModelFactory
from ..models.baselines import LogisticRegressionRecommender, PopularityRecommender
from ..training import EarlyStopping, Trainer
from ..training.checkpoint import save_checkpoint
from ..training.evaluator import compute_ranking_metrics
from ..utils import resolve_device

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Training pipeline for recommender models."""

    def __init__(self, config_path: str):
        """Initialize pipeline with config."""
        self.config_path = config_path
        self.cfg = self._load_config()
        self.mlflow_cfg = self._load_mlflow_config()
        self.device = resolve_device()

    def _load_config(self) -> dict:
        """Load model configuration."""
        from ..config import get_settings

        get_settings()
        with open(self.config_path) as f:
            return yaml.safe_load(f)["model"]

    def _load_mlflow_config(self) -> dict:
        """Load MLflow configuration."""
        mlflow_config_path = Path(self.config_path).parent / "mlflow.yaml"
        if mlflow_config_path.exists():
            with open(mlflow_config_path) as f:
                return yaml.safe_load(f).get("mlflow", {})
        return {}

    def _setup_threads_and_seed(self):
        """Configure PyTorch threads and random seed."""
        num_threads = self.cfg.get("num_threads", os.cpu_count() or 1)
        torch.set_num_threads(num_threads)
        logger.info("Configuring PyTorch to use %d CPU threads", num_threads)
        torch.manual_seed(self.cfg["seed"])
        np.random.seed(self.cfg["seed"])

    def _load_or_process_data(self):
        """Load processed data or process raw events."""
        processed_dir = Path("data/processed")
        interactions_path = processed_dir / "interactions.csv"
        user2idx_path = processed_dir / "user2idx.json"
        item2idx_path = processed_dir / "item2idx.json"

        if all(p.exists() for p in [interactions_path, user2idx_path, item2idx_path]):
            return self._load_processed_data(
                interactions_path, user2idx_path, item2idx_path
            )
        else:
            return self._process_raw_data()

    def _load_processed_data(self, interactions_path, user2idx_path, item2idx_path):
        """Load pre-processed interactions and mappings."""
        import json

        import pandas as pd

        logger.info("Loading pre-processed data from data/processed")
        interactions = pd.read_csv(interactions_path)
        with open(user2idx_path) as f:
            user2idx = {int(k): v for k, v in json.load(f).items()}
        with open(item2idx_path) as f:
            item2idx = {int(k): v for k, v in json.load(f).items()}
        processor_cfg = self.cfg.get("processor", "weighted")
        raw_path = self.cfg.get("raw_events_path", "data/raw/events.csv")
        return interactions, user2idx, item2idx, processor_cfg, processor_cfg, raw_path

    def _process_raw_data(self):
        """Process raw events into interactions."""
        raw_path = self.cfg.get("raw_events_path", "data/raw/events.csv")
        logger.info("Loading events from %s", raw_path)
        events = load_events(raw_path)
        logger.info("Total events: %d", len(events))

        processor_cfg = self.cfg.get("processor", "weighted")
        processor_kwargs = self.cfg.get("processor_kwargs", {}) or {}
        processor = DataProcessorContext(processor_cfg, **processor_kwargs)
        logger.info("Data processor: %s", processor.strategy_name)

        interactions, user2idx, item2idx = processor.process(
            events, min_interactions=self.cfg.get("min_interactions", 1)
        )
        return (
            interactions,
            user2idx,
            item2idx,
            processor_cfg,
            processor.strategy_name,
            raw_path,
        )

    def _create_dataset(self, interactions, num_items):
        """Create dataset with negative sampling."""
        logger.info("Generating dataset with negative sampling...")
        dataset = RecommenderDataset(
            interactions, num_items, num_negatives=self.cfg["num_negatives"]
        )
        logger.info("Total samples: %d", len(dataset))
        return dataset

    def _split_dataset(self, dataset):
        """Split dataset into train and validation sets."""
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.cfg["seed"]),
        )
        return train_dataset, val_dataset

    def _create_data_loaders(self, train_dataset, val_dataset):
        """Create data loaders for training."""
        num_workers = self.cfg.get("num_workers", min(4, os.cpu_count() or 1))
        logger.info("Configuring DataLoader to use %d workers", num_workers)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        return train_loader, val_loader

    def _create_model(self, num_users, num_items, model_type):
        """Create model using factory."""
        model = ModelFactory.create(
            model_type,
            num_users=num_users,
            num_items=num_items,
            **self.cfg.get("hyperparams", {}),
        )
        logger.info("Model: %s (%s)", model_type, model.model_name)
        return model.to(self.device)

    def _setup_mlflow(self):
        """Setup MLflow toolkit."""
        mlflow_toolkit = MLflowToolkit(
            tracking_uri=self.mlflow_cfg.get("tracking_uri"),
            experiment_name=self.mlflow_cfg.get(
                "experiment_name", "ecommerce_recommender"
            ),
            registry_uri=self.mlflow_cfg.get("registry_uri"),
        )
        mlflow_toolkit.setup()
        return mlflow_toolkit

    def _log_params(self, mlflow_toolkit, model_type, processor_cfg):
        """Log training parameters to MLflow."""
        mlflow_toolkit.log_params(
            {
                "model_type": model_type,
                "processor": processor_cfg,
                "seed": self.cfg["seed"],
                "batch_size": self.cfg["batch_size"],
                "learning_rate": self.cfg["learning_rate"],
                "epochs": self.cfg["epochs"],
                "num_negatives": self.cfg["num_negatives"],
                "min_interactions": self.cfg.get("min_interactions", 1),
            }
        )

    def _log_dataset(self, mlflow_toolkit, interactions, processor_cfg, raw_path):
        """Log dataset to MLflow."""
        mlflow_toolkit.log_dataset(
            interactions,
            name=f"{processor_cfg}_interactions",
            source=str(raw_path),
            context="training",
        )

    def _train_model(self, model, train_loader, val_loader, mlflow_toolkit, num_items):
        """Train model with early stopping if enabled."""
        trainer = Trainer(model, self.cfg, device=self.device)
        early_stopping_cfg = self.cfg.get("early_stopping", {})
        use_early_stopping = bool(early_stopping_cfg.get("enabled", False))

        logger.info("\nStarting training (%d epochs)...", self.cfg["epochs"])
        logger.info("-" * 60)

        mlflow_logger = create_mlflow_logger(mlflow_toolkit)

        if use_early_stopping:
            return self._train_with_early_stopping(
                trainer,
                train_loader,
                val_loader,
                mlflow_logger,
                early_stopping_cfg,
                num_items,
            )
        else:
            return self._train_without_early_stopping(
                trainer, train_loader, val_loader, mlflow_logger
            )

    def _train_with_early_stopping(
        self,
        trainer,
        train_loader,
        val_loader,
        mlflow_logger,
        early_stopping_cfg,
        num_items,
    ):
        """Train model with early stopping."""

        stopper = self._create_early_stopper(early_stopping_cfg)
        monitor = early_stopping_cfg.get("monitor", "val_loss")
        ranking_k = self.cfg.get("ranking_k", 10)

        history, best = trainer.fit_with_early_stopping(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.cfg["epochs"],
            early_stopping=stopper,
            monitor=monitor,
            show_progress=self.cfg.get("show_progress", True),
            log_callback=mlflow_logger,
            num_items=num_items if monitor.startswith("ndcg") else None,
            ranking_k=ranking_k,
        )

        self._log_early_stopping_results(monitor, stopper, best, history)
        return history, best, monitor

    def _create_early_stopper(self, early_stopping_cfg):
        """Create early stopper from config."""

        return EarlyStopping(
            patience=int(early_stopping_cfg.get("patience", 3)),
            mode=early_stopping_cfg.get("mode", "min"),
            min_delta=float(early_stopping_cfg.get("min_delta", 0.0)),
        )

    def _log_early_stopping_results(self, monitor, stopper, best, history):
        """Log early stopping results."""
        logger.info(
            "Early stopping (monitor=%s, mode=%s) - best=%s @ epoch %s | ran %d/%d epochs",
            monitor,
            stopper.mode,
            best["value"],
            best["epoch"],
            len(history),
            self.cfg["epochs"],
        )

    def _train_without_early_stopping(
        self, trainer, train_loader, val_loader, mlflow_logger
    ):
        """Train model without early stopping."""
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.cfg["epochs"],
            show_progress=self.cfg.get("show_progress", True),
            log_callback=mlflow_logger,
        )
        if history:
            metrics = history[-1].eval_metrics
            train_loss = history[-1].train_loss
            logger.info(
                "Epoch %02d/%d | Loss: %.4f | AUC: %.4f | AP: %.4f",
                len(history),
                self.cfg["epochs"],
                train_loss,
                metrics["auc_roc"],
                metrics["avg_precision"],
            )
        return history, None, None

    def _compute_ranking_metrics(self, model, val_dataset, dataset, num_items):
        """Compute ranking metrics on validation set."""
        logger.info("-" * 60)
        logger.info("Calculating ranking metrics...")

        ranking = compute_ranking_metrics(
            model=model,
            val_dataset=val_dataset,
            dataset=dataset,
            num_items=num_items,
            device=self.device,
            k=10,
            sample_limit=10000,
            positive_limit=1000,
        )

        logger.info("  Hit Rate@10:    %.4f", ranking.hit_rate)
        logger.info("  NDCG@10:        %.4f", ranking.ndcg)
        logger.info("  Precision@10:   %.4f", ranking.precision)
        logger.info("  Recall@10:      %.4f", ranking.recall)
        logger.info("  MRR@10:         %.4f", ranking.mrr)
        return ranking

    def _save_model(
        self, model, model_type, processor_name, user2idx, item2idx, metrics
    ):
        """Save model checkpoint locally."""
        artifact_dir = Path(self.cfg.get("artifact_dir", "models"))
        artifact_dir.mkdir(parents=True, exist_ok=True)

        early_stopping_info = {"best_epoch": len(metrics), "epochs_run": len(metrics)}
        metrics_dict = self._create_metrics_dict(metrics)

        artifact_path = self._save_checkpoint(
            model,
            model_type,
            processor_name,
            user2idx,
            item2idx,
            metrics_dict,
            early_stopping_info,
            artifact_dir,
        )

        self._copy_for_dvc(artifact_path, artifact_dir)
        return artifact_path

    def _create_metrics_dict(self, metrics):
        """Create metrics dictionary for checkpoint."""
        return {
            "loss": metrics[-1].train_loss if metrics else 0.0,
            "auc_roc": metrics[-1].eval_metrics["auc_roc"] if metrics else 0.0,
            "avg_precision": metrics[-1].eval_metrics["avg_precision"]
            if metrics
            else 0.0,
        }

    def _save_checkpoint(
        self,
        model,
        model_type,
        processor_name,
        user2idx,
        item2idx,
        metrics_dict,
        early_stopping_info,
        artifact_dir,
    ):
        """Save model checkpoint."""
        artifact_path = save_checkpoint(
            model=model,
            model_type=model_type,
            processor_name=processor_name,
            user2idx=user2idx,
            item2idx=item2idx,
            config=self.cfg,
            metrics=metrics_dict,
            early_stopping_info=early_stopping_info,
            artifact_dir=artifact_dir,
        )
        logger.info("Model saved locally to %s", artifact_path)
        return artifact_path

    def _copy_for_dvc(self, artifact_path, artifact_dir):
        """Copy model for DVC tracking."""
        import shutil

        dvc_model_path = artifact_dir / "model.pt"
        shutil.copyfile(artifact_path, dvc_model_path)
        logger.info("Model copy saved to %s for DVC tracking", dvc_model_path)

    def _log_metrics_and_model(
        self, mlflow_toolkit, model, metrics, ranking, registered_model_name
    ):
        """Log metrics and model to MLflow."""
        os.environ["AWS_PROFILE"] = "aws"

        final_metrics = {
            "final_train_loss": metrics[-1].train_loss if metrics else 0.0,
            "final_auc_roc": metrics[-1].eval_metrics["auc_roc"] if metrics else 0.0,
            "final_avg_precision": metrics[-1].eval_metrics["avg_precision"]
            if metrics
            else 0.0,
            **ranking.to_dict(10),
        }

        mlflow_toolkit.log_metrics(final_metrics)
        logger.info("Logging PyTorch model to MLflow server...")

        mlflow_toolkit.log_pytorch_model(
            model=model, name="model", registered_model_name=registered_model_name
        )
        logger.info("MLflow logging completed successfully.")

    def _promote_to_staging(
        self, mlflow_toolkit, run_id, registered_model_name, monitor_metric=None
    ):
        """Promote model to staging if it's the best."""
        if not registered_model_name or mlflow_toolkit.is_offline:
            return

        logger.info("Evaluating model for Staging promotion...")
        monitor_metric = monitor_metric or "ndcg_at_10"
        higher_is_better = True

        if self.cfg.get("early_stopping", {}).get("enabled"):
            monitor_metric, higher_is_better = self._determine_monitor_metric()

        self._perform_staging_promotion(
            mlflow_toolkit,
            registered_model_name,
            run_id,
            monitor_metric,
            higher_is_better,
        )

    def _determine_monitor_metric(self):
        """Determine monitor metric from config."""
        monitor_val = self.cfg["early_stopping"].get("monitor", "val_loss")
        metric_map = {
            "ndcg_at_k": ("ndcg_at_10", True),
            "auc_roc": ("final_auc_roc", True),
            "avg_precision": ("final_avg_precision", True),
            "val_loss": ("final_train_loss", False),
        }
        return metric_map.get(monitor_val, ("ndcg_at_10", True))

    def _perform_staging_promotion(
        self, mlflow_toolkit, model_name, run_id, monitor_metric, higher_is_better
    ):
        """Perform staging promotion."""
        logger.info(
            "Using metric '%s' (higher_is_better=%s)", monitor_metric, higher_is_better
        )
        promoted = mlflow_toolkit.promote_best_to_staging(
            model_name=model_name,
            run_id=run_id,
            metric_name=monitor_metric,
            higher_is_better=higher_is_better,
        )

        if promoted:
            logger.info("Model version successfully promoted to Staging.")
        else:
            logger.info("Model version evaluated but not promoted to Staging.")

    def _train_single_model(self):
        """Train single model from config."""
        self._setup_threads_and_seed()
        data = self._load_or_process_data()
        logger.info("Users: %d, Items: %d", len(data[1]), len(data[2]))

        dataset = self._create_dataset(data[0], len(data[2]))
        train_dataset, val_dataset = self._split_dataset(dataset)
        train_loader, val_loader = self._create_data_loaders(train_dataset, val_dataset)

        model_type = self.cfg.get("type", "ncf")
        model = self._create_model(len(data[1]), len(data[2]), model_type)

        self._train_and_log_single_model(
            model,
            model_type,
            data[3],
            data[4],
            data[5],
            dataset,
            train_dataset,
            val_dataset,
            train_loader,
            val_loader,
            data[1],
            data[2],
            data[0],
            data[3],
        )

    def _train_and_log_single_model(
        self,
        model,
        model_type,
        processor_cfg,
        processor_name,
        raw_path,
        dataset,
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
        user2idx,
        item2idx,
        interactions,
        processor_type,
    ):
        """Train and log single model to MLflow."""
        mlflow_toolkit = self._setup_mlflow()
        run_name = f"{model_type}-{processor_cfg}"

        with mlflow_toolkit.start_run(
            run_name=run_name,
            tags={"model_type": model_type, "processor": processor_cfg},
        ) as run:
            self._log_params(mlflow_toolkit, model_type, processor_cfg)
            self._log_dataset(mlflow_toolkit, interactions, processor_cfg, raw_path)
            self._log_train_val_samples(
                mlflow_toolkit, dataset, train_dataset, val_dataset, processor_type
            )

            history, best, monitor = self._train_model(
                model, train_loader, val_loader, mlflow_toolkit
            )

            ranking = self._compute_ranking_metrics(
                model, val_dataset, dataset, len(item2idx)
            )
            self._save_model(
                model, model_type, processor_name, user2idx, item2idx, history
            )

            self._log_metrics_and_model(
                mlflow_toolkit,
                model,
                history,
                ranking,
                self.cfg.get("registered_model_name", "ecommerce_recommender"),
            )

            self._promote_to_staging(
                mlflow_toolkit,
                run.info.run_id,
                self.cfg.get("registered_model_name", "ecommerce_recommender"),
                monitor,
            )

    def _train_model_combination(
        self, model_type, processor_type, events, mlflow_toolkit
    ):
        """Train a single model combination."""
        run_name = f"{model_type}_{processor_type}"
        logger.info(
            "\n%s\nTraining %s with %s processor\n%s",
            "=" * 60,
            model_type,
            processor_type,
            "=" * 60,
        )

        with mlflow_toolkit.start_run(run_name=run_name):
            self._log_combination_params(mlflow_toolkit, model_type, processor_type)

            try:
                interactions, user2idx, item2idx = self._process_events_for_combination(
                    processor_type, events
                )
                raw_path = self.cfg.get("raw_events_path", "data/raw/events.csv")
                self._log_dataset(
                    mlflow_toolkit, interactions, processor_type, raw_path
                )

                dataset = self._create_dataset(interactions, len(item2idx))
                train_dataset, val_dataset = self._split_dataset(dataset)

                # Log train and validation samples
                self._log_train_val_samples(
                    mlflow_toolkit, dataset, train_dataset, val_dataset, processor_type
                )

                train_loader, val_loader = self._create_data_loaders(
                    train_dataset, val_dataset
                )

                model = self._create_model(len(user2idx), len(item2idx), model_type)
                history, _, _ = self._train_model(
                    model, train_loader, val_loader, mlflow_toolkit, len(item2idx)
                )

                if history:
                    self._log_combination_results(
                        mlflow_toolkit,
                        model,
                        history,
                        model_type,
                        processor_type,
                        val_dataset,
                        dataset,
                        len(item2idx),
                    )

            except Exception as e:
                logger.error(
                    "Failed to train %s with %s: %s", model_type, processor_type, e
                )
                mlflow_toolkit.log_params({"error": str(e)})

    def _log_combination_params(self, mlflow_toolkit, model_type, processor_type):
        """Log parameters for model combination."""
        mlflow_toolkit.log_params(
            {
                "model_type": model_type,
                "processor": processor_type,
                "batch_size": self.cfg["batch_size"],
                "learning_rate": self.cfg["learning_rate"],
                "epochs": self.cfg["epochs"],
                "num_negatives": self.cfg["num_negatives"],
                "min_interactions": self.cfg.get("min_interactions", 1),
            }
        )

    def _log_train_val_samples(
        self, mlflow_toolkit, dataset, train_dataset, val_dataset, processor_type
    ):
        """Log train and validation samples to MLflow."""
        import pandas as pd

        train_samples = np.array([dataset.samples[i] for i in train_dataset.indices])
        val_samples = np.array([dataset.samples[i] for i in val_dataset.indices])

        train_df = pd.DataFrame(train_samples, columns=["user_id", "item_id", "label"])
        val_df = pd.DataFrame(val_samples, columns=["user_id", "item_id", "label"])

        logging.getLogger(__name__).info(f"Logging train samples: {len(train_df)} rows")
        logging.getLogger(__name__).info(f"Logging val samples: {len(val_df)} rows")

        mlflow_toolkit.log_dataset(
            train_df,
            name=f"{processor_type}_train_samples",
            source=f"{processor_type}_interactions",
            context="training",
        )

        mlflow_toolkit.log_dataset(
            val_df,
            name=f"{processor_type}_val_samples",
            source=f"{processor_type}_interactions",
            context="validation",
        )

    def _process_events_for_combination(self, processor_type, events):
        """Process events for specific combination."""
        processor = DataProcessorContext(processor_type)
        interactions, user2idx, item2idx = processor.process(
            events, min_interactions=self.cfg.get("min_interactions", 1)
        )
        logger.info("Processed %d interactions", len(interactions))
        logger.info("Users: %d, Items: %d", len(user2idx), len(item2idx))
        return interactions, user2idx, item2idx

    def _log_combination_results(
        self,
        mlflow_toolkit,
        model,
        history,
        model_type,
        processor_type,
        val_dataset,
        dataset,
        num_items,
    ):
        """Log results for model combination."""
        metrics = history[-1].eval_metrics
        train_loss = history[-1].train_loss
        ranking = self._compute_ranking_metrics(model, val_dataset, dataset, num_items)

        mlflow_toolkit.log_metrics(
            {
                "final_train_loss": train_loss,
                "final_auc_roc": metrics["auc_roc"],
                "final_avg_precision": metrics["avg_precision"],
                **ranking.to_dict(10),
            }
        )

        mlflow_toolkit.log_pytorch_model(
            model=model,
            name=f"{model_type}_{processor_type}",
            registered_model_name=f"ecommerce_recommender_{model_type}_{processor_type}",
        )

        logger.info("Successfully trained and logged %s_%s", model_type, processor_type)

    def _get_existing_runs(self, mlflow_toolkit):
        """Get existing run names from MLflow experiment."""
        try:
            import mlflow

            experiment = mlflow.get_experiment_by_name(mlflow_toolkit.experiment_name)
            if not experiment:
                return set()

            runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            existing_runs = set()
            for _, run in runs_df.iterrows():
                run_name = run.get("tags.mlflow.runName", "")
                if run_name:
                    existing_runs.add(run_name)
            return existing_runs
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to get existing runs: {e}")
            return set()

    def _train_comprehensive_mode(self, mlflow_toolkit):
        """Train all model combinations."""
        logger.info(
            "\n%s\nCOMPREHENSIVE MODE: Training all model combinations\n%s",
            "=" * 80,
            "=" * 80,
        )

        os.environ["AWS_PROFILE"] = "aws"

        existing_runs = self._get_existing_runs(mlflow_toolkit)
        logger.info(f"Found {len(existing_runs)} existing runs in MLflow")

        model_types = ["ncf", "gmf", "matrix_factorization"]
        processors = ["weighted", "binary", "implicit"]

        raw_path = self.cfg.get("raw_events_path", "data/raw/events.csv")
        events = load_events(raw_path)
        logger.info("Loaded %d events from %s", len(events), raw_path)

        for model_type in model_types:
            for processor_type in processors:
                run_name = f"{model_type}_{processor_type}"
                if run_name in existing_runs:
                    logger.info(f"Skipping {run_name} - already trained")
                    continue

                self._train_model_combination(
                    model_type, processor_type, events, mlflow_toolkit
                )

        # Copy best model to model.pt for DVC tracking
        # Default to ncf_weighted if it exists, otherwise use the first trained model
        model_path = Path("models/ncf_weighted.pt")
        if not model_path.exists():
            for model_type in model_types:
                for processor_type in processors:
                    candidate_path = Path(f"models/{model_type}_{processor_type}.pt")
                    if candidate_path.exists():
                        model_path = candidate_path
                        break
                if model_path.exists():
                    break

        if model_path.exists():
            import shutil

            shutil.copy(model_path, Path("models/model.pt"))
            logger.info(f"Copied {model_path} to models/model.pt for DVC tracking")

        logger.info("\n%s\nCOMPREHENSIVE TRAINING COMPLETED\n%s", "=" * 80, "=" * 80)

    def _train_baseline_models(self, mlflow_toolkit, events):
        """Train baseline models."""
        logger.info("\n%s\nTraining baseline models\n%s", "=" * 60, "=" * 60)

        processor = DataProcessorContext("weighted")
        interactions, user2idx, item2idx = processor.process(
            events, min_interactions=self.cfg.get("min_interactions", 1)
        )

        raw_path = self.cfg.get("raw_events_path", "data/raw/events.csv")

        dataset = self._create_dataset(interactions, len(item2idx))
        train_dataset, val_dataset = self._split_dataset(dataset)

        # End current run before starting baseline runs
        import mlflow

        if mlflow.active_run():
            mlflow.end_run()

        # Start baseline runs with explicit names
        self._train_popularity_baseline(
            interactions,
            dataset,
            train_dataset,
            val_dataset,
            user2idx,
            item2idx,
            mlflow_toolkit,
            raw_path,
        )
        self._train_logistic_regression_baseline(
            dataset,
            train_dataset,
            val_dataset,
            user2idx,
            item2idx,
            mlflow_toolkit,
            raw_path,
        )

        logger.info("\n%s\nCOMPREHENSIVE TRAINING COMPLETED\n%s", "=" * 80, "=" * 80)

    def _train_popularity_baseline(
        self,
        interactions,
        dataset,
        train_dataset,
        val_dataset,
        user2idx,
        item2idx,
        mlflow_toolkit,
        raw_path,
    ):
        """Train popularity baseline model."""
        logger.info("Training Popularity Baseline...")

        with mlflow_toolkit.start_run(run_name="baseline_popularity"):
            mlflow_toolkit.log_params(
                {"model_type": "popularity_baseline", "processor": "weighted"}
            )
            self._log_dataset(mlflow_toolkit, interactions, "weighted", raw_path)
            self._log_train_val_samples(
                mlflow_toolkit, dataset, train_dataset, val_dataset, "weighted"
            )

            try:
                pop_recommender = PopularityRecommender()
                train_interactions = interactions[
                    interactions.index.isin(train_dataset.indices)
                ]
                pop_recommender.fit(train_interactions)

                val_samples = np.array(
                    [dataset.samples[i] for i in val_dataset.indices]
                )
                pop_metrics = self._compute_baseline_metrics(
                    pop_recommender, val_samples, len(item2idx)
                )

                mlflow_toolkit.log_metrics(
                    {
                        "final_auc_roc": pop_metrics["auc"],
                        "final_avg_precision": pop_metrics["ap"],
                        **pop_metrics["ranking"].to_dict(10),
                    }
                )

                logger.info(
                    "Popularity Baseline - AUC: %.4f, AP: %.4f",
                    pop_metrics["auc"],
                    pop_metrics["ap"],
                )

                # Log baseline model as artifact
                import joblib

                model_path = Path("models/baseline_popularity.joblib")
                model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(pop_recommender, model_path)
                mlflow_toolkit.log_artifact(str(model_path))

            except Exception as e:
                logger.error("Failed to train popularity baseline: %s", e)
                mlflow_toolkit.log_params({"error": str(e)})

    def _compute_baseline_metrics(
        self, recommender, val_samples, num_items, sample_users=1000
    ):
        """Compute baseline metrics with user sampling for speed."""
        from sklearn.metrics import average_precision_score, roc_auc_score

        # Sample users for faster evaluation if dataset is large
        unique_users = np.unique(val_samples[:, 0])
        if len(unique_users) > sample_users:
            sampled_users = np.random.choice(unique_users, sample_users, replace=False)
            mask = np.isin(val_samples[:, 0], sampled_users)
            val_samples = val_samples[mask]
            logging.getLogger(__name__).info(
                f"Sampled {sample_users} users for baseline evaluation"
            )

        val_users = val_samples[:, 0].astype(np.int64)
        val_items = val_samples[:, 1].astype(np.int64)
        val_labels = val_samples[:, 2]

        preds = recommender.predict(val_users, val_items)
        auc = float(roc_auc_score(val_labels, preds))
        ap = float(average_precision_score(val_labels, preds))

        positive_only = val_samples[val_samples[:, 2] == 1.0][:, :2].astype(np.int64)

        from .evaluate_pipeline import compute_baseline_ranking_metrics

        ranking = compute_baseline_ranking_metrics(
            lambda users, items: recommender.predict(users, items),
            positive_only,
            num_items,
            k=10,
        )

        return {"auc": auc, "ap": ap, "ranking": ranking}

    def _train_logistic_regression_baseline(
        self,
        dataset,
        train_dataset,
        val_dataset,
        user2idx,
        item2idx,
        mlflow_toolkit,
        raw_path,
    ):
        """Train logistic regression baseline model."""
        logger.info("Training Logistic Regression Baseline...")

        with mlflow_toolkit.start_run(run_name="baseline_logistic_regression"):
            mlflow_toolkit.log_params(
                {"model_type": "logistic_regression_baseline", "processor": "weighted"}
            )
            self._log_train_val_samples(
                mlflow_toolkit, dataset, train_dataset, val_dataset, "weighted"
            )

            try:
                lr_recommender = LogisticRegressionRecommender(
                    len(user2idx), len(item2idx)
                )
                train_samples = np.array(
                    [dataset.samples[i] for i in train_dataset.indices]
                )
                lr_recommender.fit(
                    train_samples[:, 0].astype(np.int64),
                    train_samples[:, 1].astype(np.int64),
                    train_samples[:, 2],
                )

                val_samples = np.array(
                    [dataset.samples[i] for i in val_dataset.indices]
                )
                lr_metrics = self._compute_baseline_metrics(
                    lr_recommender, val_samples, len(item2idx)
                )

                mlflow_toolkit.log_metrics(
                    {
                        "final_auc_roc": lr_metrics["auc"],
                        "final_avg_precision": lr_metrics["ap"],
                        **lr_metrics["ranking"].to_dict(10),
                    }
                )

                logger.info(
                    "Logistic Regression Baseline - AUC: %.4f, AP: %.4f",
                    lr_metrics["auc"],
                    lr_metrics["ap"],
                )

                # Log baseline model as artifact
                import joblib

                model_path = Path("models/baseline_logistic_regression.joblib")
                model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(lr_recommender, model_path)
                mlflow_toolkit.log_artifact(str(model_path))

            except Exception as e:
                logger.error("Failed to train logistic regression baseline: %s", e)
                mlflow_toolkit.log_params({"error": str(e)})

    def run(self, comprehensive: bool = False):
        """Run training pipeline.

        Args:
            comprehensive: If True, trains all model combinations plus baselines.
        """
        if comprehensive:
            mlflow_toolkit = self._setup_mlflow()
            self._train_comprehensive_mode(mlflow_toolkit)
        else:
            self._train_single_model()


def run_training_pipeline(
    config_path: str = "configs/model.yaml", comprehensive: bool = False
) -> None:
    """Train a recommender model end-to-end using the config file.

    Args:
        config_path: Path to the model configuration YAML file. Defaults to "configs/model.yaml".
        comprehensive: If True, trains all model combinations (3 models x 3 processors) plus baselines.
                     If False, trains single model from config (default behavior).
    """
    pipeline = TrainingPipeline(config_path)
    pipeline.run(comprehensive=comprehensive)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run training pipeline.")
    parser.add_argument(
        "--config",
        default="configs/model.yaml",
        help="Path to the model configuration YAML file.",
    )
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Train all model combinations (3 models x 3 processors) plus baselines.",
    )
    args = parser.parse_args()
    run_training_pipeline(config_path=args.config, comprehensive=args.comprehensive)
