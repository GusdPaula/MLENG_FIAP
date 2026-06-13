def train_one_experiment(
    processor_data: dict,
    model_type: str,
    processor_name: str,
    seed: int,
) -> dict:

    interactions = processor_data["interactions"]
    user2idx = processor_data["user2idx"]
    item2idx = processor_data["item2idx"]
    processed_path = processor_data["path"]

    print(f"Training model={model_type}, processor={processor_name}")

    mlflow_toolkit.log_dataset(
        interactions,
        name=f"{processor_name}_interactions",
        source=str(processed_path),
        context="training",
    )

    num_users = len(user2idx)
    num_items = len(item2idx)

    dataset = RecommenderDataset(
        interactions,
        num_items,
        num_negatives=base_cfg["num_negatives"],
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=base_cfg["batch_size"],
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=base_cfg["batch_size"],
    )

    device = resolve_device()

    print(
        f"Device: {device} | "
        f"samples={len(dataset):,} | "
        f"train={train_size:,} | "
        f"val={val_size:,}"
    )

    model = ModelFactory.create(
        model_type,
        num_users=num_users,
        num_items=num_items,
        **build_model_hyperparams(base_cfg),
    )

    trainer = Trainer(model, base_cfg, device=device)

    early_stopping = EarlyStopping(
        patience=5,
        mode="max",
        min_delta=1e-3,
    )

    # ------------------------------------------------------------
    # MLflow callback (called immediately after every epoch)
    # ------------------------------------------------------------
    def mlflow_logger(epoch_result):
        mlflow_toolkit.log_metrics(
            {
                "train_loss": float(epoch_result.train_loss),
                "learning_rate": float(epoch_result.learning_rate),
                **{
                    k: float(v)
                    for k, v in epoch_result.eval_metrics.items()
                },
            },
            step=epoch_result.epoch,
        )

    history, best = trainer.fit_with_early_stopping(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=base_cfg["epochs"],
        early_stopping=early_stopping,
        monitor="auc_roc",
        show_progress=base_cfg.get("show_progress", True),
        log_callback=mlflow_logger,
    )

    # ------------------------------------------------------------
    # Metrics from BEST epoch
    # ------------------------------------------------------------
    best_result = next(
        r for r in history if r.epoch == best["epoch"]
    )

    best_loss = float(best_result.train_loss)
    best_metrics = best_result.eval_metrics

    mlflow_toolkit.log_metrics(
        {
            "best_epoch": int(best["epoch"]),
            "best_auc_roc": float(best["value"]),
            "epochs_run": len(history),
        }
    )

    # ------------------------------------------------------------
    # Ranking metrics (model already restored to best weights)
    # ------------------------------------------------------------
    val_indices = val_dataset.indices

    val_samples = np.array(
        [dataset.samples[i] for i in val_indices[: min(10000, len(val_indices))]]
    )

    positive_only = (
        val_samples[val_samples[:, 2] == 1.0][:, :2]
        .astype(np.int64)
    )

    hr = hit_rate_at_k(
        model,
        positive_only[:1000],
        num_items,
        k=10,
        device=device,
    )

    ndcg = ndcg_at_k(
        model,
        positive_only[:1000],
        num_items,
        k=10,
        device=device,
    )

    checkpoint_path = ARTIFACT_DIR / f"{model_type}_{processor_name}.pt"

    torch.save(
        {
            "model_type": model_type,
            "processor": processor_name,
            "model_state_dict": model.state_dict(),
            "user2idx": user2idx,
            "item2idx": item2idx,
            "config": base_cfg,
            "metrics": {
                "loss": best_loss,
                "auc_roc": float(best_metrics["auc_roc"]),
                "avg_precision": float(best_metrics["avg_precision"]),
                "hit_rate_at_10": float(hr),
                "ndcg_at_10": float(ndcg),
            },
            "early_stopping": {
                "best_epoch": best["epoch"],
                "best_auc_roc": best["value"],
                "epochs_run": len(history),
            },
        },
        checkpoint_path,
    )

    mlflow_toolkit.log_artifact(checkpoint_path)

    mlflow_toolkit.log_metrics(
        {
            "final_train_loss": best_loss,
            "final_auc_roc": float(best_metrics["auc_roc"]),
            "final_avg_precision": float(best_metrics["avg_precision"]),
            "hit_rate_at_10": float(hr),
            "ndcg_at_10": float(ndcg),
        }
    )

    print(f"Saved model artifact: {checkpoint_path}")

    result = {
        "model_type": model_type,
        "processor": processor_name,
        "artifact": str(checkpoint_path),
        "processed_data": str(processed_path),
        "train_loss": best_loss,
        "auc_roc": float(best_metrics["auc_roc"]),
        "avg_precision": float(best_metrics["avg_precision"]),
        "hit_rate_at_10": float(hr),
        "ndcg_at_10": float(ndcg),
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
    }

    del trainer
    del model
    del train_loader
    del val_loader
    del train_dataset
    del val_dataset
    del dataset

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return result
