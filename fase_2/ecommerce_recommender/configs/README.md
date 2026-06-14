# Configs Directory

This directory contains YAML configuration files for the ecommerce recommender system.

## Purpose

Configuration files control the behavior of models, training pipelines, and MLflow experiments. All configs follow a hierarchical structure that allows for easy parameter tuning without modifying code.

## Configuration Files

### base.yaml

Base configuration file that contains common settings shared across different experiments. This file typically includes:
- Default paths
- Common hyperparameters
- Shared settings

### mlflow.yaml

MLflow-specific configuration for experiment tracking:
- Tracking URI
- Experiment name
- MLflow server settings
- Artifact storage location

### model.yaml

Main model configuration file that defines:
- **Model type**: `ncf`, `gmf`, or `matrix_factorization`
- **Training parameters**: batch_size, learning_rate, epochs, seed
- **Data processing**: processor strategy (weighted, binary, implicit)
- **Negative sampling**: num_negatives, min_interactions
- **Early stopping**: monitor metric, patience, mode
- **Model hyperparameters**: embedding_dim, hidden_layers, dropout

Example structure:
```yaml
model:
  type: ncf
  seed: 42
  batch_size: 256
  learning_rate: 0.001
  epochs: 10
  processor: weighted
  hyperparams:
    embedding_dim: 64
    hidden_layers: [128, 64, 32]
    dropout: 0.2
```

### model_gmf.yaml

Configuration specific to GMF (Generalized Matrix Factorization) models. Inherits from base settings and adds GMF-specific hyperparameters.

### model_mf.yaml

Configuration specific to Matrix Factorization models. Inherits from base settings and adds MF-specific hyperparameters.

## Models Configured

The configuration system supports three recommender models:

1. **NCF (Neural Collaborative Filtering)**
   - Most expressive model
   - Uses MLP to learn complex user-item interactions
   - Configurable hidden layers and dropout

2. **GMF (Generalized Matrix Factorization)**
   - Neural version of collaborative filtering
   - Element-wise product of embeddings
   - Lightweight and fast

3. **Matrix Factorization**
   - Classic collaborative filtering baseline
   - User and item embeddings with biases
   - Simple and interpretable

## Metrics Configured

The early stopping configuration can monitor:
- `auc_roc`: Area Under ROC Curve (0.0 to 1.0, higher is better)
- `val_loss`: Validation loss (lower is better)
- `avg_precision`: Average Precision (0.0 to 1.0, higher is better)
- `ndcg_at_k`: Normalized Discounted Cumulative Gain at K (0.0 to 1.0, higher is better)

## Code Integration

Configuration files are loaded and used by:
- `src/recommender/pipelines/train_pipeline.py`: Main training pipeline
- `src/recommender/training/experiment.py`: Experiment orchestration
- `src/recommender/models/factory.py`: Model instantiation

The configuration system uses YAML for human readability and easy modification without code changes.
