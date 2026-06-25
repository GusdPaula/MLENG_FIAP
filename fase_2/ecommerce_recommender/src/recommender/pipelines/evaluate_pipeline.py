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
            f"hit_rate@{k}": self.hit_rate,
            f"ndcg@{k}": self.ndcg,
            f"precision@{k}": self.precision,
            f"recall@{k}": self.recall,
            f"mrr@{k}": self.mrr,
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


def run_evaluation_pipeline(config_path: str = "configs/model.yaml") -> None:
    """Avalia o modelo PyTorch treinado contra baselines do Scikit-Learn e registra as métricas de comparação.

    Args:
        config_path: Caminho para o arquivo YAML de configuração do modelo. Padrão é "configs/model.yaml".
    """
    from ..config import get_settings
    get_settings()  # validates and loads .env into the process

    logger.info("Iniciando estágio de avaliação usando a configuração: %s", config_path)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)["model"]

    # Carrega os dados pré-processados
    processed_dir = Path("data/processed")
    interactions_path = processed_dir / "interactions.csv"
    user2idx_path = processed_dir / "user2idx.json"
    item2idx_path = processed_dir / "item2idx.json"

    if not (
        interactions_path.exists() and user2idx_path.exists() and item2idx_path.exists()
    ):
        raise FileNotFoundError(
            f"Dados processados não encontrados em {processed_dir}. Execute o estágio de pré-processamento primeiro."
        )

    interactions = pd.read_csv(interactions_path)
    with open(user2idx_path) as f:
        user2idx = json.load(f)
    with open(item2idx_path) as f:
        item2idx = json.load(f)

    # Converte as chaves de volta para inteiros
    user2idx = {int(k): v for k, v in user2idx.items()}
    item2idx = {int(k): v for k, v in item2idx.items()}

    num_users = len(user2idx)
    num_items = len(item2idx)

    # Recria o Dataset e os splits (idênticos ao split de treino devido à seed fixa)
    logger.info("Gerando dataset e divisões de validação...")
    dataset = RecommenderDataset(
        interactions, num_items, num_negatives=cfg["num_negatives"]
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )

    # --- Carrega o modelo PyTorch -------------------------------------------
    checkpoint_path = Path(cfg.get("artifact_dir", "models")) / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint do modelo treinado não encontrado em {checkpoint_path}"
        )

    logger.info("Carregando checkpoint do modelo PyTorch de %s...", checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)

    device = resolve_device()
    model = ModelFactory.create(
        checkpoint["model_type"],
        num_users=num_users,
        num_items=num_items,
        **checkpoint["config"].get("hyperparams", {}),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # --- Avalia o Modelo PyTorch ---------------------------------------
    logger.info("Avaliando o modelo PyTorch no conjunto de validação...")
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 1),
        pin_memory=True,
    )

    trainer = Trainer(model, cfg, device=device)
    pytorch_eval = trainer.evaluate(val_loader, metrics=("auc_roc", "avg_precision"))
    pytorch_auc = pytorch_eval["auc_roc"]
    pytorch_ap = pytorch_eval["avg_precision"]

    pytorch_ranking = compute_ranking_metrics(
        model=model,
        val_dataset=val_dataset,
        dataset=dataset,
        num_items=num_items,
        device=device,
        k=10,
        sample_limit=10000,
        positive_limit=1000,
    )

    # --- Carrega as amostras de Treino e Validação para os Baselines ----------------
    logger.info("Extraindo amostras para o treinamento dos baselines...")
    train_indices = train_dataset.indices
    train_samples = np.array([dataset.samples[i] for i in train_indices])
    train_users = train_samples[:, 0].astype(np.int64)
    train_items = train_samples[:, 1].astype(np.int64)
    train_labels = train_samples[:, 2].astype(np.float32)

    val_indices = val_dataset.indices
    val_samples = np.array([dataset.samples[i] for i in val_indices])
    val_users = val_samples[:, 0].astype(np.int64)
    val_items = val_samples[:, 1].astype(np.int64)
    val_labels = val_samples[:, 2].astype(np.float32)

    # Avalia as métricas de ranking apenas em interações positivas de validação
    positive_only_val = val_samples[val_samples[:, 2] == 1.0][:, :2].astype(np.int64)
    # Limita ao limite positivo = 1000 conforme feito para o avaliador do PyTorch
    positive_only_val = positive_only_val[:1000]

    # --- 1. Baseline de Popularidade ---------------------------------------
    logger.info("Treinando e avaliando o Baseline de Popularidade...")
    train_pos_df = pd.DataFrame(
        {
            "user_idx": train_users[train_labels == 1.0],
            "item_idx": train_items[train_labels == 1.0],
        }
    )
    pop_recommender = PopularityRecommender()
    pop_recommender.fit(train_pos_df)
    pop_preds = pop_recommender.predict(val_users, val_items)

    from sklearn.metrics import average_precision_score, roc_auc_score

    pop_auc = float(roc_auc_score(val_labels, pop_preds))
    pop_ap = float(average_precision_score(val_labels, pop_preds))
    pop_ranking = compute_baseline_ranking_metrics(
        pop_recommender.predict, positive_only_val, num_items, k=10
    )

    # --- 2. Baseline de Regressão Logística ------------------------------
    logger.info("Treinando e avaliando o Baseline de Regressão Logística...")
    lr_recommender = LogisticRegressionRecommender(
        num_users=num_users, num_items=num_items
    )
    lr_recommender.fit(train_users, train_items, train_labels)
    lr_preds = lr_recommender.predict(val_users, val_items)

    lr_auc = float(roc_auc_score(val_labels, lr_preds))
    lr_ap = float(average_precision_score(val_labels, lr_preds))
    lr_ranking = compute_baseline_ranking_metrics(
        lr_recommender.predict, positive_only_val, num_items, k=10
    )

    # --- Registra os resultados no terminal ------------------------------------------
    header = (
        f"{'Modelo':<30} | {'AUC-ROC':<8} | {'Avg Prec':<8} | "
        f"{'HR@10':<8} | {'NDCG@10':<8} | {'Prec@10':<8} | {'Rec@10':<8} | {'MRR@10':<8}"
    )
    logger.info("\n" + "=" * 100)
    logger.info("TABELA COMPARATIVA (Conjunto de Validação)")
    logger.info("=" * 100)
    logger.info(header)
    logger.info("-" * 100)
    model_label = f"PyTorch ({checkpoint['model_type'].upper()})"
    logger.info(
        f"{model_label:<30} | {pytorch_auc:<8.4f} | {pytorch_ap:<8.4f} | "
        f"{pytorch_ranking.hit_rate:<8.4f} | {pytorch_ranking.ndcg:<8.4f} | "
        f"{pytorch_ranking.precision:<8.4f} | {pytorch_ranking.recall:<8.4f} | {pytorch_ranking.mrr:<8.4f}"
    )
    logger.info(
        f"{'Popularidade':<30} | {pop_auc:<8.4f} | {pop_ap:<8.4f} | "
        f"{pop_ranking.hit_rate:<8.4f} | {pop_ranking.ndcg:<8.4f} | "
        f"{pop_ranking.precision:<8.4f} | {pop_ranking.recall:<8.4f} | {pop_ranking.mrr:<8.4f}"
    )
    logger.info(
        f"{'Regressão Logística':<30} | {lr_auc:<8.4f} | {lr_ap:<8.4f} | "
        f"{lr_ranking.hit_rate:<8.4f} | {lr_ranking.ndcg:<8.4f} | "
        f"{lr_ranking.precision:<8.4f} | {lr_ranking.recall:<8.4f} | {lr_ranking.mrr:<8.4f}"
    )
    logger.info("=" * 100)

    # --- Grava o arquivo metrics.json -------------------------------------------
    metrics_dict = {
        "pytorch_auc_roc": pytorch_auc,
        "pytorch_avg_precision": pytorch_ap,
        **{f"pytorch_{k}": v for k, v in pytorch_ranking.to_dict(10).items()},
        "popularity_auc_roc": pop_auc,
        "popularity_avg_precision": pop_ap,
        **{f"popularity_{k}": v for k, v in pop_ranking.to_dict(10).items()},
        "logistic_regression_auc_roc": lr_auc,
        "logistic_regression_avg_precision": lr_ap,
        **{f"logistic_regression_{k}": v for k, v in lr_ranking.to_dict(10).items()},
    }

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "metrics.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    logger.info("Métricas salvas com sucesso em %s", metrics_path)

    # --- Registra os Baselines no MLflow --------------------------------------
    mlflow_config_path = Path("configs/mlflow.yaml")
    mlflow_cfg = {}
    if mlflow_config_path.exists():
        with open(mlflow_config_path) as f:
            mlflow_cfg = yaml.safe_load(f).get("mlflow", {})

    try:
        mlflow_toolkit = MLflowToolkit(
            tracking_uri=mlflow_cfg.get("tracking_uri"),
            experiment_name=mlflow_cfg.get("experiment_name", "ecommerce_recommender"),
            registry_uri=mlflow_cfg.get("registry_uri"),
        )
        mlflow_toolkit.setup()

        # Registra o Baseline de Popularidade
        with mlflow_toolkit.start_run(run_name="baseline-popularity"):
            mlflow_toolkit.log_params({"model_type": "popularity_baseline"})
            mlflow_toolkit.log_metrics(
                {
                    "final_auc_roc": pop_auc,
                    "final_avg_precision": pop_ap,
                    **pop_ranking.to_dict(10),
                }
            )

        # Registra o Baseline de Regressão Logística
        with mlflow_toolkit.start_run(run_name="baseline-logistic-regression"):
            mlflow_toolkit.log_params({"model_type": "logistic_regression_baseline"})
            mlflow_toolkit.log_metrics(
                {
                    "final_auc_roc": lr_auc,
                    "final_avg_precision": lr_ap,
                    **lr_ranking.to_dict(10),
                }
            )
        logger.info("Métricas dos baselines registradas com sucesso no MLflow Server.")
    except Exception as e:
        logger.warning("Falha ao registrar métricas dos baselines no MLflow: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa o pipeline de avaliação.")
    parser.add_argument(
        "--config",
        default="configs/model.yaml",
        help="Caminho para o arquivo YAML de configuração do modelo.",
    )
    args = parser.parse_args()
    run_evaluation_pipeline(config_path=args.config)
