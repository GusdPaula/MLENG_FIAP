# Processed Data Directory

This directory contains final, cleaned data ready for model training and evaluation.

## Purpose

The `processed/` directory stores data that has been fully processed and is ready for use in training recommender models. This is the final stage of the data pipeline.

## Contents

### mlflow_experiments/

Subdirectory containing MLflow experiment outputs and processed data artifacts from model training runs.

## Data Processing

Data in this directory has undergone:
1. **Loading**: From raw CSV files in `data/raw/`
2. **Cleaning**: Removal of invalid entries, handling missing values
3. **Processing**: Strategy-based transformation (weighted, binary, or implicit)
4. **Filtering**: Removing users/items with insufficient interactions
5. **Indexing**: Creating user2idx and item2idx mappings
6. **Sampling**: Negative sampling for training pairs

## Processing Strategies

The data is processed using one of three strategies from `src/recommender/data/processors.py`:

1. **WeightedEventProcessor**
   - Assigns weights: view=1, addtocart=2, transaction=3
   - Keeps all event types
   - Preserves interaction strength

2. **BinaryInteractionProcessor**
   - Keeps only addtocart and transaction events
   - Treats all as positive interactions (label=1.0)
   - Simplifies to binary classification

3. **ImplicitFeedbackProcessor**
   - Keeps all events
   - Treats all as positive with weight=1.0
   - Standard implicit feedback approach

## Models Using This Data

The processed data feeds three recommender models from `src/recommender/models/`:

1. **NCF (Neural Collaborative Filtering)**
   - Uses user-item pairs with binary labels
   - Negative sampling: 4 negatives per positive
   - Learns complex non-linear interactions

2. **GMF (Generalized Matrix Factorization)**
   - Uses user-item interaction matrix
   - Element-wise product of embeddings
   - Neural extension of collaborative filtering

3. **Matrix Factorization**
   - Classic collaborative filtering
   - User and item embeddings with biases
   - Interpretable baseline model

## Metrics Computed

Models trained on this data are evaluated using metrics from `src/recommender/training/metrics.py`:

- **AUC-ROC**: Overall discrimination ability (0.0 to 1.0)
  - Measures ability to distinguish positive from negative items
  - Computed during training on validation set

- **Average Precision**: Precision-recall tradeoff (0.0 to 1.0)
  - Summarizes precision-recall curve
  - More sensitive to class imbalance

- **Hit Rate@K**: User-focused recommendation success (0.0 to 1.0)
  - Proportion of users with at least one relevant item in top-K
  - K=10 is typical for e-commerce
  - Computed at end of training with sampling

- **NDCG@K**: Ranking quality with position awareness (0.0 to 1.0)
  - Accounts for position of relevant items in top-K
  - Logarithmic discounting: higher-ranked items more valuable
  - Best metric for ranking quality in recommenders

## Code References

- Dataset creation: `src/recommender/data/dataset.py`
- Data processing: `src/recommender/data/processors.py`
- Model training: `src/recommender/training/trainer.py`
- Metrics computation: `src/recommender/training/metrics.py`
- Pipeline orchestration: `src/recommender/pipelines/train_pipeline.py`

## MLflow Integration

Processed data and experiment results are tracked by MLflow:
- Experiment tracking via `src/recommender/mlflow_toolkit/`
- Artifacts stored in `mlflow_experiments/` subdirectory
- Metrics and parameters logged for each run
