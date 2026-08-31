"""Model loading and inference.

Wraps an ONNX Runtime session behind a small interface (predict_batch,
is_ready, warm_up) so the rest of the app never touches onnxruntime,
numpy, or the raw model output format directly.

Assumes the graph was exported with ZipMap disabled (see
ml/convert_to_onnx.py) — outputs are plain dense arrays aligned with
class_labels order, not a per-sample dict. Dicts are more convenient out
of the box but cost real time to build per sample; a dense float array
plus a fixed label list is faster to parse, which matters here since
this is the code path Etapa 4's latency comparison measures.
"""

import logging
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from app.core.exceptions import InferenceError
from app.monitoring.metrics import (
    observe_batch_size,
    observe_inference_duration,
    record_classification,
)
from app.schemas.classify import BatchClassifyResponse, PredictionResult
from app.services.preprocessing import clean_texts

logger = logging.getLogger(__name__)

# Fixed by ml/convert_to_onnx.py when the graph is built — changing these
# means updating the conversion script too, not just this file.
_INPUT_NAME = "input_text"
_LABEL_OUTPUT = "output_label"
_PROBA_OUTPUT = "output_probability"


class InferenceService:
    """Loads an ONNX model once at construction and serves batched
    predictions from it. One instance lives on app.state for the life
    of the process (see main.py's lifespan).
    """

    def __init__(self, model_path: str, class_labels: list[str], model_version: str = None) -> None:
        self._model_path = Path(model_path)
        self._class_labels = class_labels
        self._model_version = model_version
        self._session: ort.InferenceSession | None = None

        self._load()

    def _load(self) -> None:
        if not self._model_path.exists():
            raise InferenceError(f"Model artifact not found at {self._model_path}")

        self._session = ort.InferenceSession(
            str(self._model_path),
            providers=["CPUExecutionProvider"],
        )
        logger.info("model_loaded path=%s version=%s", self._model_path, self._model_version)

    def is_ready(self) -> bool:
        return self._session is not None

    def warm_up(self) -> None:
        """Fires one dummy inference at startup so the first real request
        doesn't pay ONNX Runtime's lazy-initialization cost.
        """
        try:
            self.predict_batch(["warm up"])
        except Exception:
            # Warm-up failing shouldn't crash startup — a real request
            # will surface the same error properly through InferenceError.
            logger.exception("warm_up_failed")

    def predict_batch(self, texts: list[str]) -> BatchClassifyResponse:
        """Runs the whole list of texts through a single session.run()
        call. This is the one and only inference code path — the single-
        item /v1/classify endpoint calls this with a list of length 1.
        """
        if self._session is None:
            raise InferenceError("Model session is not loaded")

        cleaned = clean_texts(texts)

        start = time.perf_counter()
        try:
            # Format input as 2D array with shape (batch_size, 1) as expected by the model
            input_array = np.array([[text] for text in cleaned], dtype=object)

            label_out, proba_out = self._session.run(
                [_LABEL_OUTPUT, _PROBA_OUTPUT],
                {_INPUT_NAME: input_array},
            )

            results = [
                self._to_prediction(label_row, proba_row)
                for label_row, proba_row in zip(label_out, proba_out, strict=True)
            ]
        except Exception as exc:
            raise InferenceError(f"ONNX Runtime inference failed: {exc}") from exc
        duration_s = time.perf_counter() - start

        results = [
            self._to_prediction(label_row, proba_row)
            for label_row, proba_row in zip(label_out, proba_out, strict=True)
        ]

        observe_inference_duration(duration_s)
        observe_batch_size(len(texts))
        for prediction in results:
            record_classification(prediction.label)

        return BatchClassifyResponse(
            results=results,
            model_version=self._model_version,
            batch_size=len(texts),
            inference_ms=duration_s * 1000,
        )

    def _to_prediction(self, label_row: int, proba_row: dict) -> PredictionResult:
        """Converts model output into a PredictionResult.

        The model returns:
        - label_row: integer class label (1-5)
        - proba_row: dictionary with logit scores for each class
        """
        import math

        # Convert logits to probabilities using softmax
        logits = list(proba_row.values())
        exp_logits = [math.exp(logit) for logit in logits]
        sum_exp = sum(exp_logits)
        probabilities = [exp_logit / sum_exp for exp_logit in exp_logits]

        # Map to class labels
        scores = {
            label: float(prob)
            for label, prob in zip(self._class_labels, probabilities, strict=True)
        }

        # Map numeric label to class name (labels are 1-indexed)
        label_index = int(label_row) - 1  # Convert to 0-indexed
        if 0 <= label_index < len(self._class_labels):
            best_label = self._class_labels[label_index]
        else:
            # Fallback to highest probability if label is out of range
            best_label = max(scores, key=scores.get)

        return PredictionResult(label=best_label, confidence=scores[best_label], scores=scores)
