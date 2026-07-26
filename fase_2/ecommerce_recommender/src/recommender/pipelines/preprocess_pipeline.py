"""Estágio de pré-processamento do pipeline para o DVC."""

import argparse
import json
import logging
from pathlib import Path

import yaml

from ..data import DataProcessorContext, load_events

# Configura o logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class PreprocessPipeline:
    """Preprocessing pipeline for raw events data."""

    def __init__(self, config_path: str):
        """Initialize pipeline with config."""
        self.config_path = config_path
        self.cfg = self._load_config()

    def _load_config(self) -> dict:
        """Load model configuration."""
        with open(self.config_path) as f:
            return yaml.safe_load(f)["model"]

    def _load_events(self):
        """Load raw events from file."""
        raw_path = self.cfg.get("raw_events_path", "data/raw/events.csv")
        logger.info("Loading raw events from %s...", raw_path)
        events = load_events(raw_path)
        logger.info("  Total events loaded: %d", len(events))
        return events

    def _process_events(self, events):
        """Process events using configured processor."""
        processor_cfg = self.cfg.get("processor", "weighted")
        processor_kwargs = self.cfg.get("processor_kwargs", {}) or {}
        processor = DataProcessorContext(processor_cfg, **processor_kwargs)
        logger.info("Applying data processor strategy: %s", processor.strategy_name)

        interactions, user2idx, item2idx = processor.process(events, min_interactions=self.cfg.get("min_interactions", 1))

        logger.info("Preprocessing completed.")
        logger.info("  Unique Users: %d", len(user2idx))
        logger.info("  Unique Items: %d", len(item2idx))
        logger.info("  Interactions shape: %s", str(interactions.shape))

        # Calculate popular items for cold start fallback
        popular_items = self._calculate_popular_items(interactions)
        logger.info("  Popular items calculated: %d items", len(popular_items))

        return interactions, user2idx, item2idx, popular_items

    def _calculate_popular_items(self, interactions):
        """Calculate item popularity scores from interactions.

        Args:
            interactions: DataFrame with user-item interactions and weights.

        Returns:
            Dictionary mapping item IDs to popularity scores.
        """
        # Group by item_id and sum the weights to get popularity
        if "weight" in interactions.columns:
            item_popularity = interactions.groupby("itemid")["weight"].sum().to_dict()
        else:
            # If no weight column, use count as popularity
            item_popularity = interactions.groupby("itemid").size().to_dict()

        # Normalize popularity scores to [0, 1] range
        if item_popularity:
            max_pop = max(item_popularity.values())
            if max_pop > 0:
                item_popularity = {k: v / max_pop for k, v in item_popularity.items()}

        return item_popularity

    def _save_processed_data(self, interactions, user2idx, item2idx, popular_items):
        """Save processed data to files."""
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)

        interactions_path = processed_dir / "interactions.csv"
        user2idx_path = processed_dir / "user2idx.json"
        item2idx_path = processed_dir / "item2idx.json"
        popular_items_path = processed_dir / "popular_items.json"

        logger.info("Saving processed files to %s...", processed_dir)

        interactions.to_csv(interactions_path, index=False)

        with open(user2idx_path, "w") as f:
            json.dump(user2idx, f)

        with open(item2idx_path, "w") as f:
            json.dump(item2idx, f)

        with open(popular_items_path, "w") as f:
            json.dump(popular_items, f)

        logger.info("Preprocessing artifacts saved successfully.")

    def run(self):
        """Run preprocessing pipeline."""
        logger.info("Starting preprocessing using config: %s", self.config_path)

        events = self._load_events()
        interactions, user2idx, item2idx, popular_items = self._process_events(events)
        self._save_processed_data(interactions, user2idx, item2idx, popular_items)


def run_preprocess_pipeline(config_path: str = "configs/model.yaml") -> None:
    """Carrega eventos brutos, aplica a estratégia de processamento e salva os artefatos processados.

    Args:
        config_path: Caminho para o arquivo YAML de configuração do modelo. Padrão é "configs/model.yaml".
    """
    pipeline = PreprocessPipeline(config_path)
    pipeline.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa o pipeline de pré-processamento.")
    parser.add_argument(
        "--config",
        default="configs/model.yaml",
        help="Caminho para o arquivo YAML de configuração do modelo.",
    )
    args = parser.parse_args()
    run_preprocess_pipeline(config_path=args.config)
