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


def run_preprocess_pipeline(config_path: str = "configs/model.yaml") -> None:
    """Carrega eventos brutos, aplica a estratégia de processamento e salva os artefatos processados.

    Args:
        config_path: Caminho para o arquivo YAML de configuração do modelo. Padrão é "configs/model.yaml".
    """
    logger.info("Iniciando estágio de pré-processamento usando a configuração: %s", config_path)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)["model"]

    raw_path = cfg.get("raw_events_path", "data/raw/events.csv")
    logger.info("Carregando eventos brutos de %s...", raw_path)
    events = load_events(raw_path)
    logger.info("  Total de eventos carregados: %d", len(events))

    processor_cfg = cfg.get("processor", "weighted")
    processor_kwargs = cfg.get("processor_kwargs", {}) or {}
    processor = DataProcessorContext(processor_cfg, **processor_kwargs)
    logger.info("Aplicando a estratégia do processador de dados: %s", processor.strategy_name)

    interactions, user2idx, item2idx = processor.process(
        events, min_interactions=cfg.get("min_interactions", 1)
    )

    logger.info("Pré-processamento concluído.")
    logger.info("  Usuários Únicos: %d", len(user2idx))
    logger.info("  Itens Únicos: %d", len(item2idx))
    logger.info("  Formato das interações: %s", str(interactions.shape))

    # Salva em data/processed
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    interactions_path = processed_dir / "interactions.csv"
    user2idx_path = processed_dir / "user2idx.json"
    item2idx_path = processed_dir / "item2idx.json"

    logger.info("Salvando arquivos processados em %s...", processed_dir)

    # Salva o DataFrame de interações
    interactions.to_csv(interactions_path, index=False)

    # Salva os mapeamentos de usuário/item para índice
    with open(user2idx_path, "w") as f:
        json.dump(user2idx, f)

    with open(item2idx_path, "w") as f:
        json.dump(item2idx, f)

    logger.info("Artefatos de pré-processamento salvos com sucesso.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa o pipeline de pré-processamento.")
    parser.add_argument(
        "--config",
        default="configs/model.yaml",
        help="Caminho para o arquivo YAML de configuração do modelo.",
    )
    args = parser.parse_args()
    run_preprocess_pipeline(config_path=args.config)
