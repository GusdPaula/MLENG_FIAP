"""Módulo para conversão de pipelines Scikit-Learn em formato ONNX (ONNX Runtime)."""

import logging
from pathlib import Path
from typing import Any

from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import StringTensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from skl2onnx.operator_converters.linear_classifier import convert_sklearn_linear_classifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)

# Registra LinearSVC com suporte às opções de saída do skl2onnx
try:
    update_registered_converter(
        LinearSVC,
        "SklearnLinearSVC",
        calculate_linear_classifier_output_shapes,
        convert_sklearn_linear_classifier,
        options={
            "nocl": [True, False],
            "zipmap": [True, False, "columns"],
            "raw_scores": [True, False],
            "output_class_labels": [True, False],
        },
    )
except Exception:  # pragma: no cover
    pass

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
DEFAULT_ONNX_PATH = ARTIFACTS_DIR / "model.onnx"


def export_pipeline_to_onnx(
    pipeline: Pipeline,
    output_path: Path | str = DEFAULT_ONNX_PATH,
    target_opset: int = 17,
) -> Path:
    """Converte uma pipeline Scikit-Learn treinada para o formato ONNX.

    Args:
        pipeline: Pipeline scikit-learn treinada (ex.: TF-IDF + LinearSVC).
        output_path: Caminho de saída para o arquivo .onnx.
        target_opset: Versão do ONNX opset a ser utilizada (padrão: 17).

    Returns:
        Path do arquivo .onnx gerado.

    Raises:
        ValueError: Se a pipeline for nula ou não for uma instância de Pipeline.
        Exception: Se a conversão via skl2onnx falhar.
    """
    if not isinstance(pipeline, Pipeline):
        msg = f"Objeto fornecido não é uma instância de sklearn.pipeline.Pipeline: {type(pipeline)}"
        logger.error(msg)
        raise ValueError(msg)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Iniciando conversão da pipeline para ONNX (opset={target_opset})...")

    # Define o tipo de entrada: tensor 2D de strings [batch_size, 1] ou [batch_size]
    initial_type = [("input_text", StringTensorType([None, 1]))]

    try:
        # Opções adicionais para compatibilidade (zipmap apenas se suportado)
        options: dict[Any, Any] = {}
        if "clf" in pipeline.named_steps:
            clf_step = pipeline.named_steps["clf"]
            if hasattr(clf_step, "predict_proba"):
                options[type(clf_step)] = {"zipmap": False}

        onnx_model = convert_sklearn(
            pipeline,
            name="MedicalTextClassifier",
            initial_types=initial_type,
            target_opset=target_opset,
            options=options if options else None,
        )

        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        logger.info(f"Modelo ONNX exportado com sucesso em: '{output_path}'")
        return output_path
    except Exception as exc:
        logger.error(f"Erro durante a conversão da pipeline para ONNX: {exc}")
        raise
