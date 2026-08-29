import logging
import sys
from pathlib import Path

from treino_modelo.data.ingest import load_data
from treino_modelo.data.validate import validate_data
from treino_modelo.train.evaluate import evaluate_model
from treino_modelo.train.train import train_model

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_pipeline() -> None:
    """Executa a pipeline completa: Ingestão → Validação → Treinamento → Avaliação.

    Encerra o processo com código 1 em caso de falha em qualquer etapa.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Etapa 1 — Ingestão
    logger.info("=== Etapa 1/4: Ingestão de dados ===")
    try:
        data = load_data()
    except Exception as exc:
        logger.error(f"Falha na etapa de Ingestão: {exc}")
        sys.exit(1)

    # Etapa 2 — Validação
    logger.info("=== Etapa 2/4: Validação de dados ===")
    try:
        validate_data(data["train"], data["test"])
    except Exception as exc:
        logger.error(f"Falha na etapa de Validação: {exc}")
        sys.exit(1)

    # Etapa 3 — Treinamento
    logger.info("=== Etapa 3/4: Treinamento do modelo ===")
    try:
        pipeline = train_model(data["train"])
    except Exception as exc:
        logger.error(f"Falha na etapa de Treinamento: {exc}")
        sys.exit(1)

    # Etapa 4 — Avaliação
    logger.info("=== Etapa 4/4: Avaliação do modelo ===")
    try:
        results = evaluate_model(pipeline, data["test"])
    except Exception as exc:
        logger.error(f"Falha na etapa de Avaliação: {exc}")
        sys.exit(1)

    # Etapa 5 — Otimização ONNX
    logger.info("=== Etapa 5/5: Exportação e Otimização ONNX ===")
    onnx_path = ARTIFACTS_DIR / "model.onnx"
    try:
        from treino_modelo.optimize.onnx_exporter import export_pipeline_to_onnx
        export_pipeline_to_onnx(pipeline, output_path=onnx_path)
    except Exception as exc:
        logger.warning(f"Não foi possível exportar modelo para ONNX: {exc}")

    # Resumo final
    model_path = ARTIFACTS_DIR / "model.pkl"
    print("\n" + "=" * 50)
    print("PIPELINE CONCLUÍDA COM SUCESSO")
    print(f"Acurácia do modelo : {results['accuracy']:.4f}")
    print(f"Artefato Sklearn   : {model_path}")
    if onnx_path.exists():
        print(f"Artefato ONNX      : {onnx_path}")
    print("=" * 50)


if __name__ == "__main__":
    run_pipeline()
