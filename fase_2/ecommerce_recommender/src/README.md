# `recommender` package overview

This folder contains the source code for a small recommender-system project built with PyTorch.

The code is organized around three main ideas:

1. Load raw ecommerce events.
2. Turn those events into training samples.
3. Train one of several recommender models through a configuration file.

## Folder structure

```text
src/recommender/
  data/         # loading, cleaning, and preprocessing helpers
  models/       # recommender model implementations and factory
  training/     # training loop and evaluation metrics
  mlflow_toolkit/ # MLflow wrapper for experiments, datasets, and models
  pipelines/    # end-to-end orchestration
```

## End-to-end flow

The main entry point is [`pipelines/train_pipeline.py`](recommender/pipelines/train_pipeline.py).

At a high level, the pipeline does the following:

1. Reads a YAML config file.
2. Loads raw events from CSV.
3. Applies a data processing strategy.
4. Builds a dataset with negative sampling.
5. Splits the dataset into train and validation sets.
6. Creates the model through a factory.
7. Trains the model and computes metrics.
8. Saves the trained model artifact to disk.

## Data handling

### `load_events`

Defined in [`data/dataset.py`](recommender/data/dataset.py), this function reads the raw CSV file into a pandas DataFrame and adds a simple event weight:

- `view` -> `1`
- `addtocart` -> `2`
- `transaction` -> `3`

### `DataProcessor` strategies

The more flexible preprocessing logic lives in [`data/processors.py`](recommender/data/processors.py).

The project uses a strategy pattern so the pipeline can switch preprocessing behavior through config.

Available strategies:

- `WeightedEventProcessor`
- `BinaryInteractionProcessor`
- `ImplicitFeedbackProcessor`

What each one does:

- `WeightedEventProcessor`: keeps all events and assigns different weights depending on the event type.
- `BinaryInteractionProcessor`: keeps only `addtocart` and `transaction` events and treats them as positive interactions.
- `ImplicitFeedbackProcessor`: keeps all events and treats every interaction as positive with weight `1.0`.

Each processor also:

- builds `user2idx` and `item2idx` mappings
- creates `user_idx` and `item_idx` columns
- optionally filters users/items with too few interactions

### `RecommenderDataset`

Also in [`data/dataset.py`](recommender/data/dataset.py), the `RecommenderDataset` class prepares training samples using negative sampling.

For every positive interaction `(user, item)`:

- it adds one positive sample with label `1.0`
- it adds `num_negatives` sampled items that the user did not interact with, each labeled `0.0`

This makes the training problem a binary classification task: the model learns to predict whether a `(user, item)` pair is likely to be a real interaction.

## Models

All models inherit from [`models/base.py`](recommender/models/base.py).

The shared contract is simple:

- each model must implement `forward(user_ids, item_ids)`
- each model must expose a `model_name`

### 1. Matrix Factorization

Defined in [`models/matrix_factorization.py`](recommender/models/matrix_factorization.py).

This is the classic collaborative filtering baseline.

It learns:

- a user embedding
- an item embedding
- a user bias
- an item bias
- a global bias

The score is computed as:

```text
score(u, i) = global_bias + user_bias + item_bias + dot(user_embedding, item_embedding)
```

Then the score is passed through a sigmoid so the output is in `[0, 1]`.

Why it is useful:

- simple
- fast
- strong baseline for recommendation tasks

### 2. GMF

Defined in [`models/gmf.py`](recommender/models/gmf.py).

GMF stands for Generalized Matrix Factorization.

It learns user and item embeddings, combines them with element-wise multiplication, optionally projects that representation to another size, and then predicts with a final linear layer.

In simple terms:

- user embedding + item embedding
- element-wise product
- optional projection
- dropout
- linear output
- sigmoid

Why it is useful:

- more flexible than plain matrix factorization
- still relatively lightweight
- works well as a neural version of collaborative filtering

### 3. NCF

Defined in [`models/ncf.py`](recommender/models/ncf.py).

NCF stands for Neural Collaborative Filtering.

This model:

- learns a user embedding
- learns an item embedding
- concatenates both embeddings
- sends them through an MLP
- applies sigmoid to produce the final score

The hidden layer sizes are configurable through `hidden_layers`.

Why it is useful:

- can learn more complex user-item interactions than MF or GMF
- typically the most expressive model in this package

## Model factory

The file [`models/factory.py`](recommender/models/factory.py) contains `ModelFactory`.

The factory is a registry that maps a string name to a model class.

Current built-in model keys:

- `ncf`
- `gmf`
- `matrix_factorization`

This is what makes the training pipeline config-driven:

```yaml
model:
  type: ncf
```

If you want to add a new model:

1. Create a subclass of `BaseRecommenderModel`.
2. Register it in `ModelFactory`.
3. Select it in the config file.

## Training

The training loop is in [`training/trainer.py`](recommender/training/trainer.py).

What the trainer does:

- uses `BCELoss`
- optimizes with Adam
- trains one epoch at a time
- evaluates with:
  - ROC AUC
  - Average Precision

Important detail:

- the models output sigmoid probabilities
- because of that, `BCELoss` is a natural fit here

## MLflow Toolkit

The MLflow helper lives in [`mlflow_toolkit/toolkit.py`](recommender/mlflow_toolkit/toolkit.py).

It is responsible for MLflow-specific tasks:

- configuring the tracking URI
- selecting or creating an experiment
- starting runs
- logging params and metrics
- logging datasets
- logging and registering PyTorch models

This keeps the training pipeline clean and avoids mixing MLflow code with model logic.

## Ranking metrics

[`training/metrics.py`](recommender/training/metrics.py) adds recommendation-style metrics:

- `hit_rate_at_k`
- `ndcg_at_k`

These are computed on the validation set after training.

They answer a different question from AUC/AP:

- AUC/AP measure classification quality
- Hit Rate and NDCG measure how good the ranked top recommendations are

## Pipeline configuration

The pipeline expects a YAML file with a `model` section.

Typical values used by the code:

- `seed`
- `raw_events_path`
- `processor`
- `processor_kwargs`
- `min_interactions`
- `num_negatives`
- `batch_size`
- `epochs`
- `learning_rate`
- `type`
- `hyperparams`
- `artifact_dir`

Example shape:

```yaml
model:
  seed: 42
  raw_events_path: data/raw/events.csv
  processor: weighted
  min_interactions: 1
  num_negatives: 4
  batch_size: 256
  epochs: 10
  learning_rate: 0.001
  type: ncf
  hyperparams:
    embedding_dim: 64
    hidden_layers: [128, 64, 32]
    dropout: 0.2
  artifact_dir: models
```

## Output artifact

At the end of training, the pipeline saves a `.pt` file containing:

- the model type
- the trained model weights
- `user2idx`
- `item2idx`
- the training config
- validation metrics

This makes it possible to reload the model later with the same user/item mapping.

## Small implementation notes

- The pipeline imports the public package APIs from `recommender.data`, `recommender.models`, and `recommender.training`.
- `create_interaction_matrix` is available in [`data/dataset.py`](recommender/data/dataset.py), but the main pipeline currently uses the processor strategies instead.
- The code is built around implicit-feedback style recommendation, not a full explicit rating system.

## In one sentence

This package loads ecommerce interaction logs, converts them into user-item training pairs, trains a configurable recommendation model, and stores the trained artifact with the mappings needed for inference later.
