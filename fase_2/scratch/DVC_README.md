# DVC Integration

This project uses DVC (Data Version Control) for data versioning and experiment tracking.

## Setup

DVC is installed via Poetry:
```bash
poetry install
```

## Data Versioning

### Tracked Data Files

**Raw Data:**
- `ecommerce_recommender/data/raw/events.csv`
- `ecommerce_recommender/data/raw/category_tree.csv`
- `ecommerce_recommender/data/raw/item_properties.csv`

**Processed Data:**
- `ecommerce_recommender/data/processed/mlflow_experiments/binary_interactions.csv`
- `ecommerce_recommender/data/processed/mlflow_experiments/implicit_interactions.csv`
- `ecommerce_recommender/data/processed/mlflow_experiments/weighted_interactions.csv`

### Common DVC Commands

```bash
# Add a new file to DVC
poetry run dvc add path/to/file.csv

# Checkout data files from cache
poetry run dvc checkout

# Check status of tracked files
poetry run dvc status

# Push data to remote storage (if configured)
poetry run dvc push

# Pull data from remote storage (if configured)
poetry run dvc pull
```

## Experiment Tracking

### Running Experiments

```bash
# Run training with DVC experiment tracking
poetry run dvc experiments run

# List experiments
poetry run dvc experiments list

# Show experiment results
poetry run dvc experiments show
```

### Configuration

- DVC configuration in `.dvc/config`
- Pipeline stages defined in `dvc.yaml`
- `.dvcignore` excludes files from DVC tracking

## Notes

- `.dvc/` directory contains DVC cache and internal files
- `dvc.lock` is tracked by Git to track pipeline state
- Model artifacts (.pt files) are tracked via MLflow, not DVC
- Use MLflow for experiment tracking and model registry
- Use DVC for data versioning and dataset tracking
