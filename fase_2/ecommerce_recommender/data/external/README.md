# External Data Directory

This directory is reserved for data from third-party sources, external APIs, or supplementary datasets.

## Purpose

The `external/` directory stores data that comes from sources outside the primary dataset. This may include:
- Third-party API responses
- Publicly available datasets
- Reference data for enrichment
- External features or metadata

## Current State

Currently empty (placeholder directory).

## Intended Use Cases

Potential external data sources for the ecommerce recommender system:
- Product image embeddings
- User demographic data
- Seasonal trends data
- Competitor pricing
- Social media sentiment
- Geographic location data

## Integration

When adding external data:
1. Place raw files in this directory
2. Document the source and update frequency
3. Add processing logic to `data_pipeline/`
4. Integrate with the main data processing pipeline in `src/recommender/data/`

## Data Processing

External data would be processed through:
- `data_pipeline/bigquery_query.py`: For cloud-based external data
- `data_pipeline/kaggle_data_loader.py`: For public datasets
- Custom loaders in `src/recommender/data/`

## Models

External data could enhance the recommender models by:
- **NCF**: Adding side information to embeddings
- **GMF**: Enriching item representations
- **Matrix Factorization**: Incorporating user/item features

## Metrics

External features may impact all evaluation metrics:
- AUC-ROC: Improved discrimination with richer features
- Average Precision: Better precision with additional context
- Hit Rate@K: Higher hit rates with better personalization
- NDCG@K: Improved ranking with feature-aware recommendations
