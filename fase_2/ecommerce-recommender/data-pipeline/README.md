# Data Pipeline

This `data-pipeline` module contains the logic to download e-commerce data from Kaggle and upload it into Google BigQuery.

## Modules

- `kaggle_data_loader.py`
  - `KaggleDataLoader`: downloads the dataset from Kaggle using `kagglehub.dataset_download`.
  - `combine_item_properties()`: merges `item_properties_part1.csv` and `item_properties_part2.csv` into a single `item_properties.csv` file.
  - `collect_files()`: returns the expected CSV files for upload.

- `bigquery_uploader.py`
  - `BigQueryUploader`: connects to Google BigQuery and uploads CSV files.
  - `ensure_dataset()`: creates the destination dataset if it does not already exist.
  - `upload_csv()`: uploads a single CSV file to the specified table.
  - `upload_files()`: uploads multiple files and returns their BigQuery table identifiers.

- `pipeline.py`
  - `DataPipeline`: orchestrates the full workflow by downloading Kaggle data, combining item properties, and uploading all dataset files to BigQuery.

- `run_pipeline.py`
  - A runnable entrypoint for the pipeline.
  - Supports command-line arguments and environment variables for configuration.

## Usage

From the `ecommerce-recommender/data-pipeline` directory, run:

```bash
python run_pipeline.py \
  --gcp-project YOUR_PROJECT_ID \
  --gcp-dataset YOUR_DATASET_ID \
  --location US \
  --table-prefix ecommerce
```

You can also configure the pipeline using environment variables:

- `KAGGLE_DATASET`
- `GCP_PROJECT_ID`
- `BIGQUERY_DATASET`
- `BIGQUERY_LOCATION`
- `TABLE_PREFIX`

## Data files

The pipeline expects the following files to exist after downloading the Kaggle dataset:

- `category_tree.csv`
- `events.csv`
- `item_properties_part1.csv`
- `item_properties_part2.csv`

The two item properties parts are combined into `item_properties.csv` before upload.

## Notes

- The BigQuery client uses standard Google Cloud authentication.
- This module is intentionally kept small so it can be tested independently.
