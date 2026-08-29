"""Módulo de abstração de inferência (Scikit-Learn e ONNX Runtime)."""

from abc import ABC, abstractmethod
import json
import logging
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
DEFAULT_SKLEARN_PATH = ARTIFACTS_DIR / "model.pkl"
DEFAULT_ONNX_PATH = ARTIFACTS_DIR / "model.onnx"
DEFAULT_METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

DEFAULT_LABEL_TO_CLASS = {
    1: "neoplasms",
    2: "digestive system diseases",
    3: "nervous system diseases",
    4: "cardiovascular diseases",
    5: "general pathological conditions",
}


def load_label_mapping(metadata_path: Path | str = DEFAULT_METADATA_PATH) -> dict[int, str]:
    """Carrega o mapeamento de rótulos a partir do JSON de metadados se existir."""
    path = Path(metadata_path)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                mapping = data.get("label_to_class", {})
                return {int(k): str(v) for k, v in mapping.items()}
        except Exception as exc:
            logger.warning(f"Não foi possível carregar metadados em '{path}': {exc}. Usando mapeamento padrão.")
    return DEFAULT_LABEL_TO_CLASS


class BaseInferenceEngine(ABC):
    """Classe base abstrata para motores de inferência."""

    def __init__(self, label_to_class: dict[int, str] | None = None) -> None:
        self.label_to_class = label_to_class or load_label_mapping()

    @abstractmethod
    def predict(self, texts: list[str] | str) -> list[dict[str, Any]]:
        """Executa a predição para um texto ou lista de textos.

        Args:
            texts: String única ou lista de strings contendo laudos médicos.

        Returns:
            Lista de dicionários com chaves 'label', 'class_name' e 'latency_ms'.
        """
        pass


class SklearnInferenceEngine(BaseInferenceEngine):
    """Motor de inferência baseado na pipeline Scikit-Learn (.pkl)."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_SKLEARN_PATH,
        pipeline: Any | None = None,
        label_to_class: dict[int, str] | None = None,
    ) -> None:
        super().__init__(label_to_class=label_to_class)
        if pipeline is not None:
            self.pipeline = pipeline
        else:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Modelo Scikit-Learn não encontrado em: '{model_path}'")
            self.pipeline = joblib.load(model_path)

    def predict(self, texts: list[str] | str) -> list[dict[str, Any]]:
        if isinstance(texts, str):
            texts = [texts]

        start_time = time.perf_counter()
        raw_predictions = self.pipeline.predict(texts)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        unit_latency_ms = elapsed_ms / max(len(texts), 1)

        results = []
        for pred in raw_predictions:
            label_id = int(pred)
            results.append({
                "label": label_id,
                "class_name": self.label_to_class.get(label_id, f"unknown ({label_id})"),
                "latency_ms": round(unit_latency_ms, 4),
            })
        return results


class ONNXInferenceEngine(BaseInferenceEngine):
    """Motor de inferência otimizado baseado no ONNX Runtime (.onnx)."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_ONNX_PATH,
        label_to_class: dict[int, str] | None = None,
    ) -> None:
        super().__init__(label_to_class=label_to_class)
        import onnxruntime as ort

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo ONNX não encontrado em: '{model_path}'")

        # Configura a sessão de inferência otimizada
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 1

        self.session = ort.InferenceSession(str(model_path), sess_options=opts)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, texts: list[str] | str) -> list[dict[str, Any]]:
        if isinstance(texts, str):
            texts = [texts]

        # Formata o input para 2D array compatível com o tensor do grafo ONNX
        input_data = np.array(texts, dtype=object).reshape(-1, 1)

        start_time = time.perf_counter()
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        unit_latency_ms = elapsed_ms / max(len(texts), 1)

        raw_predictions = outputs[0]
        results = []
        for pred in raw_predictions:
            label_id = int(pred)
            results.append({
                "label": label_id,
                "class_name": self.label_to_class.get(label_id, f"unknown ({label_id})"),
                "latency_ms": round(unit_latency_ms, 4),
            })
        return results


def get_inference_engine(
    backend: str = "onnx",
    model_path: Path | str | None = None,
) -> BaseInferenceEngine:
    """Factory para instanciar o motor de inferência desejado ('onnx' ou 'sklearn')."""
    backend_lower = backend.lower().strip()
    if backend_lower == "onnx":
        path = model_path or DEFAULT_ONNX_PATH
        return ONNXInferenceEngine(model_path=path)
    elif backend_lower in ("sklearn", "scikit-learn", "pkl"):
        path = model_path or DEFAULT_SKLEARN_PATH
        return SklearnInferenceEngine(model_path=path)
    else:
        raise ValueError(f"Backend desconhecido: '{backend}'. Opções válidas: 'onnx', 'sklearn'.")
