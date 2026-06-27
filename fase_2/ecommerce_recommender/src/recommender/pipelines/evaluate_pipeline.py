"""Estágio de avaliação e comparação de baselines do pipeline para o DVC."""

import argparse
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, random_split

from ..data import RecommenderDataset
from ..mlflow_toolkit import MLflowToolkit
from ..models import ModelFactory
from ..models.baselines import LogisticRegressionRecommender, PopularityRecommender
from ..training import Trainer, load_checkpoint
from ..training.evaluator import compute_ranking_metrics
from ..utils import resolve_device

# Configura o logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BaselineRankingMetrics:
    """Container for baseline ranking metrics."""

    hit_rate: float
    ndcg: float
    precision: float
    recall: float
    mrr: float

    def to_dict(self, k: int = 10) -> dict[str, float]:
        """Return metrics as a dictionary."""
        return {
            f"hit_rate_{k}": self.hit_rate,
            f"ndcg_{k}": self.ndcg,
            f"precision_{k}": self.precision,
            f"recall_{k}": self.recall,
            f"mrr_{k}": self.mrr,
        }


def compute_baseline_ranking_metrics(
    predict_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    test_interactions: np.ndarray,
    num_items: int,
    k: int = 10,
) -> BaselineRankingMetrics:
    """Calcula métricas de ranking para a função de predição de um modelo baseline."""
    hits = 0
    total = 0
    ndcg_scores: list[float] = []
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    rr_scores: list[float] = []

    users_items: dict[int, list[int]] = {}
    for user, item in test_interactions:
        users_items.setdefault(int(user), []).append(int(item))

    for user_idx, true_items in users_items.items():
        user_arr = np.full(num_items, user_idx, dtype=np.int64)
        item_arr = np.arange(num_items, dtype=np.int64)

        scores = predict_fn(user_arr, item_arr)
        top_k_indices = np.argsort(scores)[::-1][:k]
        top_k_set = set(top_k_indices)

        user_hits = sum(1 for item in true_items if item in top_k_set)
        hits += user_hits
        total += len(true_items)

        precision_scores.append(user_hits / k)
        recall_scores.append(user_hits / len(true_items))

        rr = 0.0
        dcg = 0.0
        for rank, item_id in enumerate(top_k_indices):
            if item_id in true_items:
                dcg += 1.0 / np.log2(rank + 2)
                if rr == 0.0:
                    rr = 1.0 / (rank + 1)
        rr_scores.append(rr)

        ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_items), k)))
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        ndcg_scores.append(ndcg)

    return BaselineRankingMetrics(
        hit_rate=hits / total if total > 0 else 0.0,
        ndcg=float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        precision=float(np.mean(precision_scores)) if precision_scores else 0.0,
        recall=float(np.mean(recall_scores)) if recall_scores else 0.0,
        mrr=float(np.mean(rr_scores)) if rr_scores else 0.0,
    )


class EvaluationPipeline:
    """Evaluation pipeline for comparing PyTorch model against baselines."""

    def __init__(self, config_path: str):
        """Initialize pipeline with config."""
        self.config_path = config_path
        self.cfg = self._load_config()
        self.device = resolve_device()

    def _load_config(self) -> dict:
        """Load model configuration."""
        from ..config import get_settings

        get_settings()
        with open(self.config_path) as f:
            return yaml.safe_load(f)["model"]

    def _load_processed_data(self):
        """Load pre-processed data."""
        processed_dir = Path("data/processed")
        interactions_path = processed_dir / "interactions.csv"
        user2idx_path = processed_dir / "user2idx.json"
        item2idx_path = processed_dir / "item2idx.json"

        if not all(
            p.exists() for p in [interactions_path, user2idx_path, item2idx_path]
        ):
            raise FileNotFoundError(
                f"Processed data not found in {processed_dir}. Run preprocessing first."
            )

        interactions = pd.read_csv(interactions_path)
        with open(user2idx_path) as f:
            user2idx = {int(k): v for k, v in json.load(f).items()}
        with open(item2idx_path) as f:
            item2idx = {int(k): v for k, v in json.load(f).items()}

        return interactions, user2idx, item2idx

    def _create_dataset(self, interactions, num_items):
        """Create dataset with negative sampling."""
        dataset = RecommenderDataset(
            interactions, num_items, num_negatives=self.cfg["num_negatives"]
        )
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.cfg["seed"]),
        )
        return dataset, train_dataset, val_dataset

    def _load_pytorch_model(self, num_users, num_items):
        """Load trained PyTorch model from checkpoint."""
        checkpoint_path = Path(self.cfg.get("artifact_dir", "models")) / "model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

        return self._load_pytorch_model_from_file(checkpoint_path, num_users, num_items)

    def _load_pytorch_model_from_file(self, model_path, num_users, num_items):
        """Load trained PyTorch model from a specific file path."""
        checkpoint = load_checkpoint(str(model_path))

        # Use hyperparameters from checkpoint if available, otherwise from config
        hyperparams = checkpoint.get("config", {}).get("hyperparams", {})
        embedding_dim = hyperparams.get(
            "embedding_dim", self.cfg.get("embedding_dim", 64)
        )
        hidden_dims = hyperparams.get(
            "hidden_dims", self.cfg.get("hidden_dims", [128, 64])
        )
        dropout = hyperparams.get("dropout", self.cfg.get("dropout", 0.2))

        model = ModelFactory.create(
            checkpoint["model_type"],
            num_users,
            num_items,
            embedding_dim=embedding_dim,
            hidden_layers=hidden_dims,
            dropout=dropout,
        )
        # Use strict=False to handle architecture mismatches
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model = model.to(self.device)
        model.eval()
        return model, checkpoint

    def _evaluate_pytorch_model(self, model, val_dataset, dataset, num_items):
        """Evaluate PyTorch model on validation set."""
        logger.info("Evaluating PyTorch model on validation set...")
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=False,
            num_workers=self.cfg.get("num_workers", 1),
            pin_memory=True,
        )

        trainer = Trainer(model, self.cfg, device=self.device)
        pytorch_eval = trainer.evaluate(
            val_loader, metrics=("auc_roc", "avg_precision")
        )

        pytorch_ranking = compute_ranking_metrics(
            model=model,
            val_dataset=val_dataset,
            dataset=dataset,
            num_items=num_items,
            device=self.device,
            k=10,
            sample_limit=10000,
            positive_limit=1000,
        )

        return pytorch_eval, pytorch_ranking

    def _extract_samples(self, dataset, train_dataset, val_dataset):
        """Extract train and validation samples for baselines."""
        logger.info("Extracting samples for baseline training...")
        train_indices = train_dataset.indices
        train_samples = np.array([dataset.samples[i] for i in train_indices])

        val_indices = val_dataset.indices
        val_samples = np.array([dataset.samples[i] for i in val_indices])

        return train_samples, val_samples

    def _train_popularity_baseline(self, train_samples, val_samples):
        """Train and evaluate popularity baseline with MLflow logging."""
        logger.info("Training and evaluating Popularity Baseline...")

        # Setup MLflow for baseline logging
        mlflow_config_path = Path("configs/mlflow.yaml")
        mlflow_cfg = {}
        if mlflow_config_path.exists():
            with open(mlflow_config_path) as f:
                mlflow_cfg = yaml.safe_load(f).get("mlflow", {})

        import os

        os.environ["AWS_PROFILE"] = "aws"
        mlflow_toolkit = MLflowToolkit(**mlflow_cfg)

        with mlflow_toolkit.start_run(run_name="baseline_popularity"):
            mlflow_toolkit.log_params(
                {"model_type": "popularity_baseline", "processor": "weighted"}
            )

            train_pos_df = pd.DataFrame(
                {
                    "user_idx": train_samples[:, 0][train_samples[:, 2] == 1.0].astype(
                        np.int64
                    ),
                    "item_idx": train_samples[:, 1][train_samples[:, 2] == 1.0].astype(
                        np.int64
                    ),
                }
            )

            # Log train and validation samples
            train_df = pd.DataFrame(
                train_samples, columns=["user_id", "item_id", "label"]
            )
            val_df = pd.DataFrame(val_samples, columns=["user_id", "item_id", "label"])
            mlflow_toolkit.log_dataset(
                train_df, name="baseline_train_samples", context="training"
            )
            mlflow_toolkit.log_dataset(
                val_df, name="baseline_val_samples", context="validation"
            )

            # Measure training time
            import time

            training_start = time.time()

            pop_recommender = PopularityRecommender()
            pop_recommender.fit(train_pos_df)

            training_latency = time.time() - training_start

            pop_preds = pop_recommender.predict(
                val_samples[:, 0].astype(np.int64), val_samples[:, 1].astype(np.int64)
            )

            from sklearn.metrics import average_precision_score, roc_auc_score

            pop_auc = float(roc_auc_score(val_samples[:, 2], pop_preds))
            pop_ap = float(average_precision_score(val_samples[:, 2], pop_preds))

            mlflow_toolkit.log_metrics(
                {
                    "final_auc_roc": pop_auc,
                    "final_avg_precision": pop_ap,
                    "training_latency": training_latency,
                }
            )

            # Log baseline model using MLflowToolkit
            logger.info("Logging baseline model to MLflow...")
            mlflow_toolkit.log_sklearn_model(
                model=pop_recommender, name="popularity_baseline"
            )
            logger.info("Baseline model logged successfully")

        return pop_recommender, pop_auc, pop_ap, training_latency

    def _train_logistic_regression_baseline(
        self, train_samples, val_samples, num_users, num_items
    ):
        """Train and evaluate logistic regression baseline with MLflow logging."""
        logger.info("Training and evaluating Logistic Regression Baseline...")

        # Setup MLflow for baseline logging
        mlflow_config_path = Path("configs/mlflow.yaml")
        mlflow_cfg = {}
        if mlflow_config_path.exists():
            with open(mlflow_config_path) as f:
                mlflow_cfg = yaml.safe_load(f).get("mlflow", {})

        import os

        os.environ["AWS_PROFILE"] = "aws"
        mlflow_toolkit = MLflowToolkit(**mlflow_cfg)

        with mlflow_toolkit.start_run(run_name="baseline_logistic_regression"):
            mlflow_toolkit.log_params(
                {"model_type": "logistic_regression_baseline", "processor": "weighted"}
            )

            # Log train and validation samples
            train_df = pd.DataFrame(
                train_samples, columns=["user_id", "item_id", "label"]
            )
            val_df = pd.DataFrame(val_samples, columns=["user_id", "item_id", "label"])
            mlflow_toolkit.log_dataset(
                train_df, name="baseline_train_samples", context="training"
            )
            mlflow_toolkit.log_dataset(
                val_df, name="baseline_val_samples", context="validation"
            )

            # Measure training time
            import time

            training_start = time.time()

            lr_recommender = LogisticRegressionRecommender(num_users, num_items)
            lr_recommender.fit(
                train_samples[:, 0].astype(np.int64),
                train_samples[:, 1].astype(np.int64),
                train_samples[:, 2],
            )

            training_latency = time.time() - training_start

            lr_preds = lr_recommender.predict(
                val_samples[:, 0].astype(np.int64), val_samples[:, 1].astype(np.int64)
            )

            from sklearn.metrics import average_precision_score, roc_auc_score

            lr_auc = float(roc_auc_score(val_samples[:, 2], lr_preds))
            lr_ap = float(average_precision_score(val_samples[:, 2], lr_preds))

            mlflow_toolkit.log_metrics(
                {
                    "final_auc_roc": lr_auc,
                    "final_avg_precision": lr_ap,
                    "training_latency": training_latency,
                }
            )

            # Log baseline model using MLflowToolkit
            logger.info("Logging baseline model to MLflow...")
            mlflow_toolkit.log_sklearn_model(
                model=lr_recommender, name="logistic_regression_baseline"
            )
            logger.info("Baseline model logged successfully")

        return lr_recommender, lr_auc, lr_ap, training_latency

    def _compute_baseline_rankings(
        self, pop_recommender, lr_recommender, positive_only_val, num_items
    ):
        """Compute ranking metrics for baselines."""
        pop_ranking = compute_baseline_ranking_metrics(
            pop_recommender.predict, positive_only_val, num_items, k=10
        )
        lr_ranking = compute_baseline_ranking_metrics(
            lr_recommender.predict, positive_only_val, num_items, k=10
        )
        return pop_ranking, lr_ranking

    def _print_comparison_table(
        self,
        mlflow_model_metrics,
        pop_auc,
        pop_ap,
        pop_ranking,
        lr_auc,
        lr_ap,
        lr_ranking,
        pop_latency,
        lr_latency,
    ):
        """Print comparison table with all models."""
        header = (
            f"{'Modelo':<30} | {'Loss':<8} | {'AUC-ROC':<8} | {'Avg Prec':<8} | "
            f"{'HR@10':<8} | {'NDCG@10':<8} | {'Prec@10':<8} | {'Rec@10':<8} | {'MRR@10':<8} | {'TrainLat(s)':<10}"
        )
        logger.info("\n" + "=" * 120)
        logger.info("TABELA COMPARATIVA (Conjunto de Validação)")
        logger.info("=" * 120)
        logger.info(header)
        logger.info("-" * 120)

        # Print all PyTorch models from MLflow
        for metrics in mlflow_model_metrics:
            model_label = f"{metrics['name']} ({metrics['type'].upper()})"
            logger.info(
                f"{model_label:<30} | {metrics['final_train_loss']:<8.4f} | {metrics['auc_roc']:<8.4f} | {metrics['avg_precision']:<8.4f} | "
                f"{metrics['hit_rate_10']:<8.4f} | {metrics['ndcg_10']:<8.4f} | "
                f"{metrics['precision_10']:<8.4f} | {metrics['recall_10']:<8.4f} | {metrics['mrr_10']:<8.4f} | {metrics['training_latency']:<10.2f}"
            )

        # Print baselines
        logger.info(
            f"{'Popularidade':<30} | {'N/A':<8} | {pop_auc:<8.4f} | {pop_ap:<8.4f} | "
            f"{pop_ranking.hit_rate:<8.4f} | {pop_ranking.ndcg:<8.4f} | "
            f"{pop_ranking.precision:<8.4f} | {pop_ranking.recall:<8.4f} | {pop_ranking.mrr:<8.4f} | {pop_latency:<10.4f}"
        )
        logger.info(
            f"{'Regressão Logística':<30} | {'N/A':<8} | {lr_auc:<8.4f} | {lr_ap:<8.4f} | "
            f"{lr_ranking.hit_rate:<8.4f} | {lr_ranking.ndcg:<8.4f} | "
            f"{lr_ranking.precision:<8.4f} | {lr_ranking.recall:<8.4f} | {lr_ranking.mrr:<8.4f} | {lr_latency:<10.4f}"
        )
        logger.info("=" * 120)

    def _save_metrics(
        self,
        mlflow_model_metrics,
        pop_auc,
        pop_ap,
        pop_ranking,
        lr_auc,
        lr_ap,
        lr_ranking,
        pop_latency,
        lr_latency,
    ):
        """Save metrics to JSON file."""
        metrics_dict = {
            "popularity_auc_roc": pop_auc,
            "popularity_avg_precision": pop_ap,
            "popularity_training_latency": pop_latency,
            **{f"popularity_{k}": v for k, v in pop_ranking.to_dict(10).items()},
            "logistic_regression_auc_roc": lr_auc,
            "logistic_regression_avg_precision": lr_ap,
            "logistic_regression_training_latency": lr_latency,
            **{
                f"logistic_regression_{k}": v for k, v in lr_ranking.to_dict(10).items()
            },
        }

        # Add metrics for all PyTorch models from MLflow
        for metrics in mlflow_model_metrics:
            model_name = metrics["name"]
            metrics_dict[f"{model_name}_auc_roc"] = metrics["auc_roc"]
            metrics_dict[f"{model_name}_avg_precision"] = metrics["avg_precision"]
            metrics_dict[f"{model_name}_final_train_loss"] = metrics["final_train_loss"]
            metrics_dict[f"{model_name}_training_latency"] = metrics["training_latency"]
            metrics_dict[f"{model_name}_hit_rate_10"] = metrics["hit_rate_10"]
            metrics_dict[f"{model_name}_ndcg_10"] = metrics["ndcg_10"]
            metrics_dict[f"{model_name}_precision_10"] = metrics["precision_10"]
            metrics_dict[f"{model_name}_recall_10"] = metrics["recall_10"]
            metrics_dict[f"{model_name}_mrr_10"] = metrics["mrr_10"]

        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = reports_dir / "metrics.json"

        with open(metrics_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        logger.info("Metrics saved successfully to %s", metrics_path)

    def _log_baselines_to_mlflow(
        self, pop_auc, pop_ap, pop_ranking, lr_auc, lr_ap, lr_ranking
    ):
        """Log baseline metrics to MLflow."""
        mlflow_config_path = Path("configs/mlflow.yaml")
        mlflow_cfg = {}
        if mlflow_config_path.exists():
            with open(mlflow_config_path) as f:
                mlflow_cfg = yaml.safe_load(f).get("mlflow", {})

        try:
            mlflow_toolkit = MLflowToolkit(
                tracking_uri=mlflow_cfg.get("tracking_uri"),
                experiment_name=mlflow_cfg.get(
                    "experiment_name", "ecommerce_recommender"
                ),
                registry_uri=mlflow_cfg.get("registry_uri"),
            )
            mlflow_toolkit.setup()

            with mlflow_toolkit.start_run(run_name="baseline-popularity"):
                mlflow_toolkit.log_params({"model_type": "popularity_baseline"})
                mlflow_toolkit.log_metrics(
                    {
                        "final_auc_roc": pop_auc,
                        "final_avg_precision": pop_ap,
                        **pop_ranking.to_dict(10),
                    }
                )

            with mlflow_toolkit.start_run(run_name="baseline-logistic-regression"):
                mlflow_toolkit.log_params(
                    {"model_type": "logistic_regression_baseline"}
                )
                mlflow_toolkit.log_metrics(
                    {
                        "final_auc_roc": lr_auc,
                        "final_avg_precision": lr_ap,
                        **lr_ranking.to_dict(10),
                    }
                )

            logger.info("Baseline metrics logged successfully to MLflow Server.")
        except Exception as e:
            logger.warning("Failed to log baseline metrics to MLflow: %s", e)

    def _extract_mlflow_metrics(self):
        """Extract metrics for all trained models from MLflow."""
        mlflow_config_path = Path("configs/mlflow.yaml")
        mlflow_cfg = {}
        if mlflow_config_path.exists():
            with open(mlflow_config_path) as f:
                mlflow_cfg = yaml.safe_load(f).get("mlflow", {})

        import mlflow

        mlflow.set_tracking_uri(
            mlflow_cfg.get("tracking_uri", "https://mlflow.asgardprint.com.br")
        )
        experiment_name = mlflow_cfg.get(
            "experiment_name", "ecommerce_recommender_fiap_5"
        )

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            logger.error(f"Experiment {experiment_name} not found")
            return []

        runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        model_metrics = []
        model_types = ["ncf", "gmf", "matrix_factorization"]
        processors = ["weighted", "binary", "implicit"]

        for model_type in model_types:
            for processor in processors:
                run_name = f"{model_type}_{processor}"
                matching_runs = runs_df[runs_df["tags.mlflow.runName"] == run_name]

                if not matching_runs.empty:
                    run = matching_runs.iloc[0]
                    # MLflow automatically tracks run duration in milliseconds
                    run_duration_ms = run.get("duration", 0)
                    run_duration_s = (
                        run_duration_ms / 1000.0 if run_duration_ms else 0.0
                    )

                    metrics = {
                        "name": run_name,
                        "auc_roc": run.get("metrics.final_auc_roc", 0.0),
                        "avg_precision": run.get("metrics.final_avg_precision", 0.0),
                        "hit_rate_10": run.get("metrics.hit_rate_10", 0.0),
                        "ndcg_10": run.get("metrics.ndcg_10", 0.0),
                        "precision_10": run.get("metrics.precision_10", 0.0),
                        "recall_10": run.get("metrics.recall_10", 0.0),
                        "mrr_10": run.get("metrics.mrr_10", 0.0),
                        "final_train_loss": run.get("metrics.final_train_loss", 0.0),
                        "training_latency": run_duration_s,
                        "type": model_type,
                    }
                    model_metrics.append(metrics)
                    logger.info(
                        f"Extracted metrics for {run_name}: AUC={metrics['auc_roc']:.4f}, Latency={metrics['training_latency']:.2f}s"
                    )

        return model_metrics

    def run(self):
        """Run evaluation pipeline."""
        logger.info("Starting evaluation using config: %s", self.config_path)

        interactions, user2idx, item2idx = self._load_processed_data()
        num_users = len(user2idx)
        num_items = len(item2idx)

        dataset, train_dataset, val_dataset = self._create_dataset(
            interactions, num_items
        )

        # Extract metrics for all 9 trained models from MLflow
        mlflow_model_metrics = self._extract_mlflow_metrics()

        train_samples, val_samples = self._extract_samples(
            dataset, train_dataset, val_dataset
        )
        positive_only_val = val_samples[val_samples[:, 2] == 1.0][:, :2].astype(
            np.int64
        )[:1000]

        pop_recommender, pop_auc, pop_ap, pop_latency = self._train_popularity_baseline(
            train_samples, val_samples
        )
        lr_recommender, lr_auc, lr_ap, lr_latency = (
            self._train_logistic_regression_baseline(
                train_samples, val_samples, num_users, num_items
            )
        )

        pop_ranking, lr_ranking = self._compute_baseline_rankings(
            pop_recommender, lr_recommender, positive_only_val, num_items
        )

        self._print_comparison_table(
            mlflow_model_metrics,
            pop_auc,
            pop_ap,
            pop_ranking,
            lr_auc,
            lr_ap,
            lr_ranking,
            pop_latency,
            lr_latency,
        )

        self._save_metrics(
            mlflow_model_metrics,
            pop_auc,
            pop_ap,
            pop_ranking,
            lr_auc,
            lr_ap,
            lr_ranking,
            pop_latency,
            lr_latency,
        )


def run_evaluation_pipeline(config_path: str = "configs/model.yaml") -> None:
    """Avalia o modelo PyTorch treinado contra baselines do Scikit-Learn e registra as métricas de comparação.

    Args:
        config_path: Caminho para o arquivo YAML de configuração do modelo. Padrão é "configs/model.yaml".
    """
    pipeline = EvaluationPipeline(config_path)
    pipeline.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa o pipeline de avaliação.")
    parser.add_argument(
        "--config",
        default="configs/model.yaml",
        help="Caminho para o arquivo YAML de configuração do modelo.",
    )
    args = parser.parse_args()
    run_evaluation_pipeline(config_path=args.config)
