# Raw Data Directory

This directory contains original, immutable data files from external sources.

## Purpose

The `raw/` directory stores the original data in its unmodified form. This data should never be altered directly to ensure reproducibility and data provenance.

## Contents

### events.csv

User interaction events from the ecommerce platform.
- **Size**: ~97 MB
- **Format**: CSV with columns for user_id, item_id, event_type, timestamp
- **Event types**: view, addtocart, transaction
- **Purpose**: Primary source for user-item interactions

### item_properties.csv

Product metadata and properties.
- **Size**: ~891 MB
- **Format**: CSV with item properties and attributes
- **Purpose**: Item features and metadata for enrichment

### category_tree.csv

Category hierarchy information.
- **Size**: ~16 KB
- **Format**: CSV with category IDs and relationships
- **Purpose**: Category structure for hierarchical recommendations

## Data Version Control

Large data files are tracked with DVC (Data Version Control):
- `.dvc` files are committed to Git
- Actual data files are stored in remote storage
- Use `dvc pull` to fetch data
- Use `dvc push` to upload data

## Data Source

This dataset is from the E-commerce Events dataset (likely Kaggle or similar source), containing:
- User interactions with products
- Item properties and metadata
- Category information
- Timestamps for temporal analysis

## Data Loading

Raw data is loaded using:
- `data_pipeline/kaggle_data_loader.py`: For local CSV files
- `data_pipeline/bigquery_query.py`: For cloud-based data
- `src/recommender/data/dataset.py`: Main data loading logic

## Data Processing Pipeline

Raw data flows through the following processing steps:

1. **Load events.csv** → Create interaction DataFrame
2. **Load item_properties.csv** → Extract item features
3. **Load category_tree.csv** → Build category hierarchy
4. **Apply processing strategy** (weighted, binary, or implicit)
5. **Filter users/items** with minimum interactions
6. **Create user2idx and item2idx mappings**
7. **Generate training pairs** with negative sampling
8. **Split into train/validation sets**

## Models Using This Data

The raw data supports three recommender models from `src/recommender/models/`:

1. **NCF (Neural Collaborative Filtering)**
   - Requires: user-item interaction pairs
   - Uses: Event types for weighted processing
   - Benefits: Can leverage temporal patterns from timestamps

2. **GMF (Generalized Matrix Factorization)**
   - Requires: User-item interaction matrix
   - Uses: Binary or weighted interactions
   - Benefits: Fast training on large datasets

3. **Matrix Factorization**
   - Requires: User-item ratings/interactions
   - Uses: Event weights as implicit ratings
   - Benefits: Interpretable user/item embeddings

## Metrics Impact

Raw data characteristics affect evaluation metrics:

- **AUC-ROC**: Higher with more diverse interaction patterns
- **Average Precision**: Improved with balanced event types
- **Hit Rate@K**: Better with sufficient user history
- **NDCG@K**: Enhanced with clear interaction preferences

## Data Characteristics

- **Sparsity**: Typical user-item interaction matrix is very sparse
- **Imbalance**: View events dominate, transactions are rare
- **Temporal**: Timestamps enable time-based splitting
- **Multi-type**: Different event types provide signal strength

## Code References

- Data loading: `src/recommender/data/dataset.py`
- Data processing: `src/recommender/data/processors.py`
- Pipeline: `data_pipeline/pipeline.py`
- BigQuery integration: `data_pipeline/bigquery_query.py`
- Kaggle loader: `data_pipeline/kaggle_data_loader.py`

## Important Notes

- Never modify files in this directory directly
- Always work with copies in `interim/` or `processed/`
- Use DVC commands for data operations
- Document any data quality issues discovered
