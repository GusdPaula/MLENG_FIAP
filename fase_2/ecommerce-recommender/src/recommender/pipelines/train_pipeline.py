import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split

from recommender.data.dataset import (
    load_events,
    create_interaction_matrix,
    RecommenderDataset,
)
from recommender.models.ncf import NCFModel
from recommender.training.trainer import Trainer
from recommender.training.metrics import hit_rate_at_k, ndcg_at_k


def run_training_pipeline(config_path: str = "configs/model.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)["model"]

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # 1. Carregar dados
    print("Carregando eventos...")
    events = load_events("data/raw/events.csv")
    print(f"  Total de eventos: {len(events)}")

    # 2. Filtrar cold-start
    min_interactions = config["min_interactions"]
    user_counts = events["visitorid"].value_counts()
    item_counts = events["itemid"].value_counts()
    events = events[
        events["visitorid"].isin(user_counts[user_counts >= min_interactions].index)
        & events["itemid"].isin(item_counts[item_counts >= min_interactions].index)
    ]
    print(f"  Após filtro cold-start (min {min_interactions}): {len(events)} eventos")

    # 3. Criar mapeamentos
    events, user2idx, item2idx = create_interaction_matrix(events)
    num_users = len(user2idx)
    num_items = len(item2idx)
    print(f"  Users: {num_users}, Items: {num_items}")

    # 4. Dataset com negative sampling
    print("Gerando dataset com negative sampling...")
    dataset = RecommenderDataset(
        events, num_items, num_negatives=config["num_negatives"]
    )
    print(f"  Total de samples (positivos + negativos): {len(dataset)}")

    # 5. Split treino/validação
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config["seed"]),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    # 6. Modelo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = NCFModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=config["embedding_dim"],
        hidden_layers=config["hidden_layers"],
        dropout=config["dropout"],
    )

    # 7. Treinar
    trainer = Trainer(model, config, device=device)

    print(f"\nIniciando treino ({config['epochs']} epochs)...")
    print("-" * 60)

    for epoch in range(config["epochs"]):
        train_loss = trainer.train_epoch(train_loader)
        metrics = trainer.evaluate(val_loader)
        print(
            f"Epoch {epoch + 1:02d}/{config['epochs']} | "
            f"Loss: {train_loss:.4f} | "
            f"AUC: {metrics['auc_roc']:.4f} | "
            f"AP: {metrics['avg_precision']:.4f}"
        )

    # 8. Métricas de ranking no validation set
    print("-" * 60)
    print("Calculando métricas de ranking...")
    val_indices = val_dataset.indices
    val_samples = np.array(
        [dataset.samples[i] for i in val_indices[: min(10000, len(val_indices))]]
    )
    positive_only = val_samples[val_samples[:, 2] == 1.0][:, :2].astype(np.int64)

    hr = hit_rate_at_k(model, positive_only[:1000], num_items, k=10, device=device)
    ndcg = ndcg_at_k(model, positive_only[:1000], num_items, k=10, device=device)
    print(f"  Hit Rate@10: {hr:.4f}")
    print(f"  NDCG@10: {ndcg:.4f}")

    # 9. Salvar modelo
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "user2idx": user2idx,
            "item2idx": item2idx,
            "config": config,
            "metrics": {
                "auc_roc": metrics["auc_roc"],
                "avg_precision": metrics["avg_precision"],
                "hit_rate_at_10": hr,
                "ndcg_at_10": ndcg,
            },
        },
        "models/ncf_model.pt",
    )
    print("\nModelo salvo em models/ncf_model.pt")


if __name__ == "__main__":
    run_training_pipeline()
