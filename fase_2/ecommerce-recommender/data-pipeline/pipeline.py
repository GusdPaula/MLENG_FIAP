import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from bigquery_uploader import BigQueryUploader
from kaggle_data_loader import KaggleDataLoader


class DataPipeline:
    """Orchestrate Kaggle dataset download and BigQuery upload."""

    def __init__(
        self,
        kaggle_loader: KaggleDataLoader,
        bigquery_uploader: BigQueryUploader,
        table_prefix: str = "",
    ) -> None:
        """Initialize the data pipeline orchestration.

        Args:
            kaggle_loader: Instance that downloads Kaggle data.
            bigquery_uploader: Instance that uploads files to BigQuery.
            table_prefix: Optional prefix for destination BigQuery tables.
        """
        self.kaggle_loader = kaggle_loader
        self.bigquery_uploader = bigquery_uploader
        self.table_prefix = table_prefix

    def run(self) -> dict[str, str]:
        """Execute the full data pipeline.

        Returns:
            Mapping of logical dataset names to BigQuery table identifiers.
        """
        download_path = self.kaggle_loader.download_dataset()
        combined_file_path = self.kaggle_loader.combine_item_properties(download_path)

        source_files = {
            "category_tree": download_path / "category_tree.csv",
            "events": download_path / "events.csv",
            "item_properties": combined_file_path,
        }

        if self.table_prefix:
            prefixed_source_files = {
                f"{self.table_prefix}_{key}": path for key, path in source_files.items()
            }
        else:
            prefixed_source_files = source_files

        return self.bigquery_uploader.upload_files(prefixed_source_files)
