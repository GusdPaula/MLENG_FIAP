import argparse
import logging
import os

from bigquery_uploader import BigQueryUploader
from dotenv import find_dotenv, load_dotenv
from kaggle_data_loader import KaggleDataLoader
from pipeline import DataPipeline

load_dotenv(find_dotenv())
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ecommerce data pipeline: download from Kaggle and upload to BigQuery."
    )
    parser.add_argument(
        "--kaggle-dataset",
        default=os.getenv("KAGGLE_DATASET", "retailrocket/ecommerce-dataset"),
        help="Kaggle dataset identifier to download.",
    )
    parser.add_argument(
        "--gcp-project",
        default=os.getenv("GCP_PROJECT"),
        help="Google Cloud project id for BigQuery.",
    )
    parser.add_argument(
        "--gcp-dataset",
        default=os.getenv("BIGQUERY_DATASET", os.getenv("GCP_DATASET", "ecommerce_dataset")),
        help="BigQuery dataset id where tables will be loaded.",
    )
    parser.add_argument(
        "--location",
        default=os.getenv("BIGQUERY_LOCATION", os.getenv("GCP_REGION", "US")),
        help="Google BigQuery dataset location.",
    )
    parser.add_argument(
        "--table-prefix",
        default=os.getenv("TABLE_PREFIX", ""),
        help="Optional prefix to add to BigQuery table names.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    kaggle_loader = KaggleDataLoader(dataset_name=args.kaggle_dataset)
    bigquery_uploader = BigQueryUploader(
        project_id=args.gcp_project,
        dataset_id=args.gcp_dataset,
        location=args.location,
    )

    pipeline = DataPipeline(
        kaggle_loader=kaggle_loader,
        bigquery_uploader=bigquery_uploader,
        table_prefix=args.table_prefix,
    )

    results = pipeline.run()

    logger.info("Uploaded tables:")
    for name, destination in results.items():
        logger.info("- %s: %s", name, destination)


if __name__ == "__main__":
    main()
