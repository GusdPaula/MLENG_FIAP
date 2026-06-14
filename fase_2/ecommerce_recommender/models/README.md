# Models Directory

This directory contains trained model artifacts and experiment outputs from the ecommerce recommender system.

## Purpose

The `models/` directory stores:
- Trained PyTorch model weights (.pt files)
- MLflow experiment artifacts
- Model checkpoints
- Serialized model configurations

## Contents

### mlflow_experiments/

Subdirectory containing trained model artifacts for different model architectures and processing strategies:

**GMF Models (Generalized Matrix Factorization)**
- `gmf_binary.pt`: GMF trained with binary interaction processor
- `gmf_implicit.pt`: GMF trained with implicit feedback processor
- `gmf_weighted.pt`: GMF trained with weighted event processor

**Matrix Factorization Models**
- `matrix_factorization_binary.pt`: MF trained with binary interaction processor
- `matrix_factorization_implicit.pt`: MF trained with implicit feedback processor
- `matrix_factorization_weighted.pt`: MF trained with weighted event processor

**NCF Models (Neural Collaborative Filtering)**
- `ncf_binary.pt`: NCF trained with binary interaction processor
- `ncf_implicit.pt`: NCF trained with implicit feedback processor
- `ncf_weighted.pt`: NCF trained with weighted event processor

## Model Architectures

### 1. NCF (Neural Collaborative Filtering)

**Implementation**: `src/recommender/models/ncf.py`

**Architecture**:
- User embedding layer
- Item embedding layer
- Concatenation of embeddings
- Multi-layer perceptron (MLP) with configurable hidden layers
- Dropout for regularization
- Sigmoid output layer

**Hyperparameters** (configurable in `configs/model.yaml`):
- `embedding_dim`: Size of user/item embeddings (default: 64)
- `hidden_layers`: MLP layer sizes (default: [128, 64, 32])
- `dropout`: Dropout rate (default: 0.2)

**Strengths**:
- Can learn complex non-linear user-item interactions
- Most expressive model in the suite
- Flexible architecture through configurable hidden layers

**Use cases**:
- When interaction patterns are complex
- When you have sufficient data for deep learning
- When maximizing ranking quality is priority

### 2. GMF (Generalized Matrix Factorization)

**Implementation**: `src/recommender/models/gmf.py`

**Architecture**:
- User embedding layer
- Item embedding layer
- Element-wise product of embeddings
- Optional projection layer
- Dropout for regularization
- Linear output layer
- Sigmoid activation

**Hyperparameters**:
- `embedding_dim`: Size of user/item embeddings
- Optional projection dimensions

**Strengths**:
- Neural extension of classic matrix factorization
- Lightweight and fast to train
- Good balance of simplicity and expressiveness

**Use cases**:
- When you want a neural approach but need speed
- As a middle ground between MF and NCF
- For large-scale recommendation systems

### 3. Matrix Factorization

**Implementation**: `src/recommender/models/matrix_factorization.py`

**Architecture**:
- User embedding layer
- Item embedding layer
- User bias term
- Item bias term
- Global bias term
- Dot product of embeddings + biases
- Sigmoid output

**Hyperparameters**:
- `embedding_dim`: Size of user/item embeddings

**Strengths**:
- Classic collaborative filtering baseline
- Highly interpretable
- Fast training and inference
- Strong baseline for comparison

**Use cases**:
- As a baseline for comparison
- When interpretability is important
- For quick prototyping
- When computational resources are limited

## Processing Strategies

Each model is trained with one of three data processing strategies from `src/recommender/data/processors.py`:

### Weighted Event Processor
- Assigns weights: view=1, addtocart=2, transaction=3
- Preserves interaction strength
- Best when event type matters

### Binary Interaction Processor
- Keeps only addtocart and transaction events
- All treated as positive (label=1.0)
- Simplifies to binary classification
- Best for clear positive/negative signals

### Implicit Feedback Processor
- Keeps all events
- All treated as positive with weight=1.0
- Standard implicit feedback approach
- Best when all interactions provide signal

## Metrics

Models are evaluated using four metrics from `src/recommender/training/metrics.py`:

### AUC-ROC (Area Under ROC Curve)
- **Range**: 0.0 to 1.0 (higher is better)
- **Purpose**: Measures overall discrimination ability
- **Computation**: During training on validation set
- **Interpretation**: Probability that positive item ranks higher than negative
- **Limitation**: Doesn't measure top-K performance

### Average Precision (AP)
- **Range**: 0.0 to 1.0 (higher is better)
- **Purpose**: Summarizes precision-recall curve
- **Computation**: During training on validation set
- **Interpretation**: Quality of positive predictions
- **Strength**: More sensitive to class imbalance

### Hit Rate@K (HR@K)
- **Range**: 0.0 to 1.0 (higher is better)
- **Purpose**: User-focused recommendation success
- **Computation**: At end of training with sampling
- **Typical K**: 10 for e-commerce
- **Interpretation**: Proportion of users with relevant item in top-K
- **Strength**: Directly measures user experience

### NDCG@K (Normalized Discounted Cumulative Gain)
- **Range**: 0.0 to 1.0 (higher is better)
- **Purpose**: Ranking quality with position awareness
- **Computation**: At end of training with sampling
- **Typical K**: 10 for e-commerce
- **Interpretation**: Accounts for position of relevant items in top-K
- **Strength**: Best metric for ranking quality in recommenders

## Model Artifacts

Each `.pt` file contains:
- Model type (ncf, gmf, or matrix_factorization)
- Trained model weights
- `user2idx` mapping (user IDs to indices)
- `item2idx` mapping (item IDs to indices)
- Training configuration
- Validation metrics

## Code References

- Model implementations: `src/recommender/models/`
- Model factory: `src/recommender/models/factory.py`
- Training loop: `src/recommender/training/trainer.py`
- Metrics computation: `src/recommender/training/metrics.py`
- Experiment orchestration: `src/recommender/training/experiment.py`
- MLflow integration: `src/recommender/mlflow_toolkit/`
- Training pipeline: `src/recommender/pipelines/train_pipeline.py`

## Model Selection

When choosing a model:
1. **Start with Matrix Factorization** as a baseline
2. **Try GMF** for a neural approach with good speed
3. **Use NCF** for maximum expressiveness if data permits
4. **Compare metrics** across all models
5. **Consider processing strategy** impact on performance

## Loading Models

To load a trained model:
```python
import torch
artifact = torch.load('models/mlflow_experiments/ncf_weighted.pt')
# Contains: model weights, mappings, config, metrics
```

## MLflow Integration

Models are tracked with MLflow:
- Experiments logged to MLflow tracking server
- Artifacts stored in this directory
- Metrics and parameters recorded for each run
- Model registration supported via MLflow toolkit
