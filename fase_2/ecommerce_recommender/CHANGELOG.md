# Changelog

## Unreleased

### Added

- New `recommender.mlflow_toolkit` package with `MLflowToolkit` for experiment setup, dataset logging, model logging, and registration.
- MLflow dependency added to `requirements.txt`.
- **Early stopping** support in the `Trainer` class via a new `EarlyStopping` helper. The training pipeline now supports early stopping when configured in `configs/model.yaml`.
- **Batch processing** support in `RecommenderDataset` with a new `streaming` parameter that enables memory-efficient negative sampling.
- New `BatchCollator` and `make_batches` utilities for explicit batch handling.
- New `EpochResult` dataclass to track training metrics per epoch.

### Changed

- Root README updated to describe the project structure, models, training flow, and MLflow support.
- `src/README.md` updated to reflect the current package layout and training module names.
- The `Trainer` class now supports both simple epoch-by-epoch training and early stopping with `fit_with_early_stopping`.
- The `RecommenderDataset` now supports a streaming mode for memory-efficient batch processing.
- Training pipeline now supports early stopping via configuration in `configs/model.yaml`.

### Fixed

- Training pipeline import now points to `recommender.training.trainer`.
- **ModelFactory** now filters hyperparameters per model type via `MODEL_PARAM_MAP` to prevent passing invalid parameters to models (e.g., GMF now rejects `hidden_layers`).
- **Early stopping** metric changed from AUC to NDCG@10 for better ranking evaluation. Added lightweight NDCG@K computation to trainer's evaluate method with sampling for efficiency.
- **.gitignore** updated to properly handle model artifacts with unignore hierarchy for mlflow_experiments while keeping specific .pt models in Git as needed.
- **.pre-commit-config.yaml** fixed to ensure hooks are triggered in commits by running from correct directory where pyproject.toml is located.
