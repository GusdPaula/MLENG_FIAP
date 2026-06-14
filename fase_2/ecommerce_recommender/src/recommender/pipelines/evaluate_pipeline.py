"""Estágio de avaliação e comparação de baselines do pipeline para o DVC."""

import argparse
import json
import logging
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


def compute_baseline_ranking_metrics(
    predict_fn,
    test_interactions: np.ndarray,
    num_items: int,
    k: int = 10,
) -> tuple[float, float]:
    """Calcula Hit Rate@K e NDCG@K para a função de predição de um modelo baseline."""
    hits = 0
    total = 0
    ndcg_scores = []

    users_items: dict[int, list[int]] = {}
    for user, item in test_interactions:
        users_items.setdefault(int(user), []).append(int(item))

    for user_idx, true_items in users_items.items():
        user_arr = np.full(num_items, user_idx, dtype=np.int64)
        item_arr = np.arange(num_items, dtype=np.int64)

        # Prediz scores para todos os itens do catálogo
        scores = predict_fn(user_arr, item_arr)

        # Obtém os top-K itens
        top_k_indices = np.argsort(scores)[::-1][:k]
        top_k_set = set(top_k_indices)

        for item in true_items:
            if item in top_k_set:
                hits += 1
            total += 1

        dcg = 0.0
        for rank, item_id in enumerate(top_k_indices):
            if item_id in true_items:
                dcg += 1.0 / np.log2(rank + 2)

        ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_items), k)))
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        ndcg_scores.append(ndcg)

    hr = hits / total if total > 0 else 0.0
    ndcg_val = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
    return hr, ndcg_val


def run_evaluation_pipeline(config_path: str = "configs/model.yaml") -> None:
    """Avalia o modelo PyTorch treinado contra baselines do Scikit-Learn e registra as métricas de comparação.

    Args:
        config_path: Caminho para o arquivo YAML de configuração do modelo. Padrão é "configs/model.yaml".
    """
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

    pytorch_hr, pytorch_ndcg = compute_ranking_metrics(
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
    pop_hr, pop_ndcg = compute_baseline_ranking_metrics(
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
    lr_hr, lr_ndcg = compute_baseline_ranking_metrics(
        lr_recommender.predict, positive_only_val, num_items, k=10
    )

    # --- Registra os resultados no terminal ------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("TABELA COMPARATIVA (Conjunto de Validação)")
    logger.info("=" * 80)
    logger.info(
        f"{'Modelo':<30} | {'AUC-ROC':<10} | {'Avg Prec':<10} | {'Hit Rate@10':<12} | {'NDCG@10':<10}"
    )
    logger.info("-" * 80)
    logger.info(
        f"{f'Modelo PyTorch ({checkpoint["model_type"].upper()})':<30} | {pytorch_auc:<10.4f} | {pytorch_ap:<10.4f} | {pytorch_hr:<12.4f} | {pytorch_ndcg:<10.4f}"
    )
    logger.info(
        f"{'Recomendador Popularidade':<30} | {pop_auc:<10.4f} | {pop_ap:<10.4f} | {pop_hr:<12.4f} | {pop_ndcg:<10.4f}"
    )
    logger.info(
        f"{'Regressão Logística':<30} | {lr_auc:<10.4f} | {lr_ap:<10.4f} | {lr_hr:<12.4f} | {lr_ndcg:<10.4f}"
    )
    logger.info("=" * 80)

    # --- Grava o arquivo metrics.json -------------------------------------------
    metrics_dict = {
        "pytorch_auc_roc": pytorch_auc,
        "pytorch_avg_precision": pytorch_ap,
        "pytorch_hit_rate_at_10": pytorch_hr,
        "pytorch_ndcg_at_10": pytorch_ndcg,
        "popularity_auc_roc": pop_auc,
        "popularity_avg_precision": pop_ap,
        "popularity_hit_rate_at_10": pop_hr,
        "popularity_ndcg_at_10": pop_ndcg,
        "logistic_regression_auc_roc": lr_auc,
        "logistic_regression_avg_precision": lr_ap,
        "logistic_regression_hit_rate_at_10": lr_hr,
        "logistic_regression_ndcg_at_10": lr_ndcg,
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
                    "hit_rate_at_10": pop_hr,
                    "ndcg_at_10": pop_ndcg,
                }
            )

        # Registra o Baseline de Regressão Logística
        with mlflow_toolkit.start_run(run_name="baseline-logistic-regression"):
            mlflow_toolkit.log_params({"model_type": "logistic_regression_baseline"})
            mlflow_toolkit.log_metrics(
                {
                    "final_auc_roc": lr_auc,
                    "final_avg_precision": lr_ap,
                    "hit_rate_at_10": lr_hr,
                    "ndcg_at_10": lr_ndcg,
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
