from pathlib import Path

from google.cloud import bigquery


class BigQueryUploader:
    """Upload local CSV files into Google BigQuery tables."""

    def __init__(self, project_id: str, dataset_id: str, location: str = "US") -> None:
        """Create a BigQuery uploader for a target project and dataset.

        Args:
            project_id: Google Cloud project id.
            dataset_id: BigQuery dataset id.
            location: Location for the dataset.
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.location = location
        self.client = bigquery.Client(project=self.project_id)

    def ensure_dataset(self) -> None:
        """Ensure the destination dataset exists in BigQuery."""
        dataset_ref = bigquery.Dataset(f"{self.project_id}.{self.dataset_id}")
        dataset_ref.location = self.location
        self.client.create_dataset(dataset_ref, exists_ok=True)

    def upload_csv(self, source_path: Path, table_name: str) -> str:
        """Upload a single CSV file to a BigQuery table.

        Args:
            source_path: Local path to the CSV file.
            table_name: Destination BigQuery table name.

        Returns:
            The fully qualified BigQuery table identifier.

        Raises:
            FileNotFoundError: If the source CSV does not exist.
        """
        if not source_path.exists():
            raise FileNotFoundError(f"CSV file not found: {source_path}")

        dataset_ref = self.client.dataset(self.dataset_id)
        table_ref = dataset_ref.table(table_name)
        job_config = bigquery.LoadJobConfig(
            autodetect=True,
            skip_leading_rows=1,
            source_format=bigquery.SourceFormat.CSV,
        )

        with source_path.open("rb") as source_file:
            load_job = self.client.load_table_from_file(
                source_file,
                table_ref,
                job_config=job_config,
            )
            load_job.result()

        return f"{self.project_id}.{self.dataset_id}.{table_name}"

    def upload_files(self, source_files: dict[str, Path]) -> dict[str, str]:
        """Upload multiple CSV files to BigQuery.

        Args:
            source_files: Mapping of table names to local CSV file paths.

        Returns:
            Mapping of table names to fully qualified BigQuery table identifiers.
        """
        self.ensure_dataset()
        results: dict[str, str] = {}

        for table_name, path in source_files.items():
            results[table_name] = self.upload_csv(path, table_name)

        return results
