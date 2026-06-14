# Data Directory

This directory contains all data files for the ecommerce recommender system, organized by processing stage.

## Purpose

The data directory follows a standard data science project structure:
- **raw/**: Original, immutable data from external sources
- **external/**: Data from third-party sources
- **interim/**: Intermediate data that has been transformed but not finalized
- **processed/**: Final, cleaned data ready for modeling

## Data Flow

```
raw/ → external/ → interim/ → processed/ → models/
```

## Subdirectories

### raw/

Contains original data files from the primary data source:
- `events.csv`: User interaction events (views, addtocart, transactions)
- `item_properties.csv`: Item metadata and properties
- `category_tree.csv`: Category hierarchy information

These files are tracked with DVC (Data Version Control) using `.dvc` files.

### external/

Reserved for data from third-party sources or APIs. Currently empty (placeholder).

### interim/

Intermediate data files created during data processing and feature engineering. Currently empty (placeholder).

### processed/

Final processed data ready for model training:
- Cleaned and transformed datasets
- User-item interaction matrices
- Feature-engineered data
- MLflow experiment outputs

## Data Processing

The data is processed through the following pipeline:

1. **Loading**: Raw CSV files are loaded using `data_pipeline/kaggle_data_loader.py` or `data_pipeline/bigquery_query.py`
2. **Cleaning**: Data cleaning and preprocessing in `src/recommender/data/dataset.py`
3. **Processing**: Strategy-based processing in `src/recommender/data/processors.py`:
   - `WeightedEventProcessor`: Assigns weights to different event types
   - `BinaryInteractionProcessor`: Converts to binary positive/negative interactions
   - `ImplicitFeedbackProcessor`: Treats all interactions as positive

## Models Using This Data

The processed data feeds into three recommender models:

1. **NCF (Neural Collaborative Filtering)**
   - Requires user-item pairs with labels
   - Uses negative sampling for training

2. **GMF (Generalized Matrix Factorization)**
   - Requires user-item interaction matrix
   - Uses embedding-based approach

3. **Matrix Factorization**
   - Requires user-item ratings/interactions
   - Classic collaborative filtering approach

## Metrics Computed

The data is used to compute the following metrics during model evaluation:

- **AUC-ROC**: Overall discrimination ability (0.0 to 1.0)
- **Average Precision**: Precision-recall tradeoff (0.0 to 1.0)
- **Hit Rate@K**: Proportion of users with at least one relevant item in top-K (0.0 to 1.0)
- **NDCG@K**: Ranking quality with position awareness (0.0 to 1.0)

## Code References

- Data loading: `src/recommender/data/dataset.py`
- Data processing: `src/recommender/data/processors.py`
- Pipeline orchestration: `data_pipeline/pipeline.py`
- BigQuery integration: `data_pipeline/bigquery_query.py`

## Data Version Control

Large data files are tracked with DVC to avoid committing them to Git. The `.dvc` files in this directory are committed to version control, while the actual data files are stored in remote storage.
