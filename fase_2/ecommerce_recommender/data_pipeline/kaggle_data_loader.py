from pathlib import Path

import kagglehub
import pandas as pd


class KaggleDataLoader:
    """Download and prepare e-commerce dataset files from Kaggle."""

    def __init__(self, dataset_name: str = "retailrocket/ecommerce-dataset") -> None:
        """Initialize the loader with a Kaggle dataset identifier.

        Args:
            dataset_name: The Kaggle dataset slug.
        """
        self._dataset_name = dataset_name

    def download_dataset(self) -> Path:
        """Download the latest version of the configured Kaggle dataset.

        Returns:
            Path to the downloaded dataset root directory.

        Raises:
            FileNotFoundError: If the downloaded path does not exist.
        """
        downloaded_path = Path(kagglehub.dataset_download(self._dataset_name))
        if not downloaded_path.exists():
            raise FileNotFoundError(
                f"Downloaded dataset path not found: {downloaded_path}"
            )
        return downloaded_path

    def combine_item_properties(
        self, download_root: Path, combined_filename: str = "item_properties.csv"
    ) -> Path:
        """Combine item properties parts into a single CSV file.

        Args:
            download_root: Root directory where dataset files were downloaded.
            combined_filename: Name of the combined output CSV.

        Returns:
            Path to the combined item properties CSV file.

        Raises:
            FileNotFoundError: If either part file is missing.
        """
        part1_path = download_root / "item_properties_part1.csv"
        part2_path = download_root / "item_properties_part2.csv"

        if not part1_path.exists():
            raise FileNotFoundError(f"Missing file: {part1_path}")
        if not part2_path.exists():
            raise FileNotFoundError(f"Missing file: {part2_path}")

        data_frame_1 = pd.read_csv(part1_path)
        data_frame_2 = pd.read_csv(part2_path)
        combined_data_frame = pd.concat([data_frame_1, data_frame_2], ignore_index=True)

        combined_path = download_root / combined_filename
        combined_data_frame.to_csv(combined_path, index=False)
        return combined_path

    def collect_files(self, download_root: Path) -> dict[str, Path]:
        """Collect the dataset files that should be uploaded.

        Args:
            download_root: Root directory where the dataset files exist.

        Returns:
            A mapping between logical dataset names and file paths.
        """
        category_tree = download_root / "category_tree.csv"
        events = download_root / "events.csv"
        item_properties = download_root / "item_properties.csv"

        return {
            "category_tree": category_tree,
            "events": events,
            "item_properties": item_properties,
        }
