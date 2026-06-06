import csv
import subprocess
from pathlib import Path

from google.cloud import bigquery


def _find_dvc_root(start_path: Path) -> Path | None:
    current = start_path.resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".dvc").exists():
            return parent
    return None


class BigQueryQuery:
    """Extract data from BigQuery and version exported files with DVC."""

    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        output_dir: str | Path,
        dvc_repo_path: str | Path | None = None,
    ) -> None:
        """Create a BigQuery query client that writes CSV results and versions them.

        Args:
            project_id: Google Cloud project id.
            dataset_id: BigQuery dataset id.
            output_dir: Local directory where query results are stored.
            dvc_repo_path: Path to the DVC repository root. Defaults to the current working directory.
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dvc_repo_path = Path(dvc_repo_path) if dvc_repo_path else Path.cwd()
        self.dvc_repo_root = (
            self.dvc_repo_path
            if dvc_repo_path is not None
            else _find_dvc_root(self.dvc_repo_path)
        )
        self.client = bigquery.Client(project=self.project_id)

    def extract_table(self, table_name: str, destination_name: str | None = None) -> Path:
        """Extract an entire BigQuery table to a local CSV file and version it with DVC."""
        destination_name = destination_name or f"{table_name}.csv"
        query = f"SELECT * FROM `{self.project_id}.{self.dataset_id}.{table_name}`"
        return self.extract_query(query, destination_name)

    def extract_query(self, query: str, destination_name: str) -> Path:
        """Run an arbitrary SQL query, write the results to CSV, and version the output."""
        destination_path = self.output_dir / destination_name
        self._write_query_results_to_csv(query, destination_path)
        self._version_with_dvc(destination_path)
        return destination_path

    def _write_query_results_to_csv(self, query: str, destination_path: Path) -> None:
        query_job = self.client.query(query)
        result = query_job.result()

        headers = [field.name for field in result.schema]
        with destination_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(headers)
            for row in result:
                writer.writerow([row[field] for field in headers])

    def _version_with_dvc(self, destination_path: Path) -> None:
        if self.dvc_repo_root is None:
            raise RuntimeError(
                "DVC repository was not found. "
                f"Checked '{self.dvc_repo_path}' and its parents, but no .dvc directory was found. "
                "Initialize DVC with `dvc init` in your repository root or pass a valid `dvc_repo_path`."
            )

        try:
            completed = subprocess.run(
                ["dvc", "add", str(destination_path)],
                cwd=str(self.dvc_repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("DVC executable was not found. Ensure DVC is installed and on PATH.") from exc

        if completed.returncode != 0:
            raise RuntimeError(
                f"DVC add failed for {destination_path}: {completed.stderr.strip() or completed.stdout.strip()}"
            )
