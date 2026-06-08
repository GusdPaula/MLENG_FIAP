# AGENT.md

## Project Overview

This project implements a product recommendation system for an e-commerce platform using user behavior data.

The solution must:

* Train a recommendation model using PyTorch
* Track experiments with MLflow
* Version datasets and pipelines using DVC
* Run inside Docker containers
* Follow Clean Code and SOLID principles
* Be fully reproducible from scratch

---

# Agent Responsibilities

## 1. Data Engineering Agent

### Responsibilities

* Load raw interaction data
* Validate schema
* Handle missing values
* Generate user-item interaction matrices
* Create train/validation/test splits
* Persist processed datasets

### Input

Raw dataset from DVC storage

### Output

Processed datasets stored in:

data/processed/

---

## 2. Feature Engineering Agent

### Responsibilities

* Generate user features
* Generate item features
* Create embeddings metadata
* Encode categorical variables
* Normalize numerical variables

### Input

Processed datasets

### Output

Feature-ready datasets

---

## 3. Training Agent

### Responsibilities

* Train recommendation models
* Compare architectures
* Log experiments into MLflow

Supported models:

* MLP Recommender
* Embedding-based Recommender

### Requirements

* Fixed random seeds
* Early stopping
* Model checkpointing

### Output

Trained model artifacts

models/

---

## 4. Evaluation Agent

### Responsibilities

Evaluate:

* Precision@K
* Recall@K
* F1 Score
* ROC AUC
* MAP@K (optional)

Compare results against baseline models.

### Output

Evaluation reports

reports/

---

## 5. Registry Agent

### Responsibilities

* Register best model in MLflow Registry
* Promote model through stages

Stages:

* Development
* Staging
* Production

---

## 6. Deployment Agent

### Responsibilities

* Build Docker image
* Validate runtime environment
* Package production artifacts

Optional:

* Deploy to AWS
* Deploy to Azure
* Deploy to GCP

---

# Architecture

src/
│
├── config/
├── data/
├── features/
├── models/
├── evaluation/
├── registry/
├── deployment/
├── pipelines/
└── utils/

tests/
data/
models/
configs/
scripts/

---

# Design Patterns

The project must use at least one design pattern.

Recommended patterns:

## Factory Pattern

Used for model creation.

Example:

ModelFactory.create("mlp")

ModelFactory.create("embedding")

---

## Strategy Pattern

Used for preprocessing strategies.

Examples:

* StandardScalerStrategy
* MinMaxScalerStrategy
* EmbeddingEncodingStrategy

---

# Coding Standards

## Clean Code

Requirements:

* Functions ≤ 20 lines whenever possible
* Descriptive naming
* Single Responsibility Principle
* Small modules
* Explicit dependencies

---

## Type Hints

All public methods must contain type hints.

Example:

def train_model(
train_data: pd.DataFrame,
config: TrainingConfig
) -> nn.Module:

---

## Docstrings

Google Style required.

Example:

def train_model():
"""
Train recommendation model.

```
Args:
    train_data: Training dataset.

Returns:
    Trained PyTorch model.
"""
```

---

# Reproducibility

## Dependency Management

Use:

* Poetry or uv

Files:

* pyproject.toml
* lock file

---

## Environment Variables

Configuration must be loaded from:

.env

Example:

MLFLOW_TRACKING_URI=
DATA_PATH=
MODEL_PATH=

---

## Environment Validation

scripts/validate_env.py

Checks:

* Python version
* Required packages
* Environment variables
* DVC installation

---

# DVC Pipeline

Required stages:

1. preprocess
2. feature_eng
3. train
4. evaluate

Example flow:

raw_data
↓
preprocess
↓
feature_eng
↓
train
↓
evaluate

Pipeline execution:

dvc repro

---

# MLflow Tracking

Every run must log:

## Parameters

* learning_rate
* batch_size
* epochs
* embedding_dim

## Metrics

* precision
* recall
* f1
* auc

## Artifacts

* trained model
* plots
* evaluation reports

---

# Docker Standards

Use multi-stage builds.

Stages:

## Builder

Install dependencies

## Runtime

Minimal image containing:

* application
* model artifacts
* runtime dependencies

Required files:

* Dockerfile
* docker-compose.yml
* .dockerignore

---

# Testing Requirements

Framework:

pytest

Minimum coverage targets:

* Data pipeline
* Feature engineering
* Training
* Evaluation

---

# Repository Standards

Required files:

README.md
AGENT.md
pyproject.toml
poetry.lock
.env.example
Dockerfile
docker-compose.yml
dvc.yaml

---

# Success Criteria

The project is considered complete when:

* Dataset is versioned with DVC
* Pipeline reproduces end-to-end
* Recommendation model trains successfully
* Experiments are tracked in MLflow
* Best model is registered
* Docker containers execute successfully
* Repository follows Clean Code principles
* README provides full installation instructions

---

# STAR Presentation Alignment

## Situation

E-commerce recommendation problem.

## Task

Build a reproducible recommendation system.

## Action

Use:

* PyTorch
* DVC
* MLflow
* Docker
* Clean Architecture

## Result

Deliver:

* Trained model
* Reproducible pipeline
* Registry-managed model
* Production-ready repository
