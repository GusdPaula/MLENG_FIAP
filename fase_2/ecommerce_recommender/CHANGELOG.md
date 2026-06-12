# Changelog

## Unreleased

### Added

- New `recommender.mlflow_toolkit` package with `MLflowToolkit` for experiment setup, dataset logging, model logging, and registration.
- MLflow dependency added to `requirements.txt`.

### Changed

- Root README updated to describe the project structure, models, training flow, and MLflow support.
- `src/README.md` updated to reflect the current package layout and training module names.

### Fixed

- Training pipeline import now points to `recommender.training.trainer`.
