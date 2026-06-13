# Changelog

All relevant changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## Unreleased

### Added

- New `recommender.mlflow_toolkit` package with `MLflowToolkit` for experiment setup, dataset logging, model logging, and registration.
- MLflow dependency added to `requirements.txt`.
- **Early stopping** support in the `Trainer` class via a new `EarlyStopping` helper. The training pipeline now supports early stopping when configured in `configs/model.yaml`.
- **Batch processing** support in `RecommenderDataset` with a new `streaming` parameter that enables memory-efficient negative sampling.
- New `BatchCollator` and `make_batches` utilities for explicit batch handling.
- New `EpochResult` dataclass to track training metrics per epoch.
- Poetry project initialized with runtime/dev dependencies
- Local virtual environment `.venv` created
- `pyproject.toml` and `poetry.lock` added
- `data_pipeline` package added with Kaggle download and BigQuery upload support
- `run_pipeline.py` added as module entry point and README
- `bigquery_query.py` added for BigQuery extraction and DVC versioning/export
- `ecommerce_recommender/notebooks/data_pipeline_eda.ipynb` added for exploratory data analysis
- `ecommerce_recommender/README.md` added with package overview and data_pipeline documentation
- Dedicated `fase_2/.pre-commit-config.yaml` configuration added to run `ruff` and `pytest` in fase_2 context
- `fase_2/ruff.toml` adjusted to exclude notebooks and use correct lint configuration
- Comprehensive metrics documentation (`METRICS_DOCUMENTATION.md`) explaining metric interpretation and why multiple metrics are necessary

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
- **DVC integration** improved for data versioning and tracking. Added raw and processed datasets to DVC cache, updated .gitignore to track dvc.lock, created basic dvc.yaml configuration.
- **Test coverage** improved from 87% to 91% by adding comprehensive unit tests for metrics.py (100% coverage), evaluator.py (100% coverage), checkpoint.py (81% coverage), and callbacks.py (100% coverage).
- **Documentation and style** improved by converting Portuguese docstrings to English Google-style across all modules and adding missing return type hints.
- **Model evaluation** verified that all experiments track AUC-ROC, Average Precision, HitRate@K, and NDCG@K.
- `BigQueryQuery` fixed to accept explicit `dvc_repo_path` and allow local versioning with DVC
- **Architecture** improved by extracting common weight initialization patterns into BaseRecommenderModel (_init_embeddings, _init_linear_layers) to reduce code duplication across GMF, NCF, and MatrixFactorization models.
- **Architecture** improved by extracting common min_interactions filtering logic into DataProcessor base class (_filter_by_min_interactions) to reduce code duplication across WeightedEventProcessor, BinaryInteractionProcessor, and ImplicitFeedbackProcessor.
