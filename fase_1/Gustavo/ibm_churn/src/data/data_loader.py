import logging

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

from .bg_settings import settings


class BigQueryDataLoader:
    """Class to load data from Google BigQuery."""

    def __init__(
        self, project_id="ibm-churn", dataset_id="ibmchurn", table_id="ibm_churn"
    ):
        load_dotenv()  # load environment variables from .env file

        self.project_id = settings.google_cloud_project
        self.dataset_id = settings.bq_dataset_id
        self.table_id = settings.bq_table_id
        self.df = pd.DataFrame()  # Initialize an empty DataFrame

        self.client = bigquery.Client(project=self.project_id)

    def load_data(self, query=None) -> None:
        """Loads data from BigQuery and returns it as a pandas DataFrame."""
        if query is None:
            query = f"""
                SELECT *
                FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`
            """

        try:
            logging.info("Starting to load data from BigQuery...")
            self.df = self.client.query(query).to_dataframe()
            logging.info(f"Success! {len(self.df)} rows loaded.")

        except Exception as e:
            logging.error(f"Error occurred while querying BigQuery: {e}")

        return None

    def parse_data(self) -> pd.DataFrame:
        """Parses and preprocesses the DataFrame as needed."""
        pass
