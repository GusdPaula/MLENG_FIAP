"""Testes unitários para o módulo de otimização ONNX e motores de inferência."""

from pathlib import Path

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from treino_modelo.optimize.inference import (
    DEFAULT_LABEL_TO_CLASS,
    ONNXInferenceEngine,
    SklearnInferenceEngine,
    get_inference_engine,
    load_label_mapping,
)
from treino_modelo.optimize.onnx_exporter import export_pipeline_to_onnx


@pytest.fixture
def trained_toy_pipeline(tmp_path):
    """Cria e treina uma pipeline TF-IDF + LinearSVC em dados sintéticos para testes rápidos."""
    texts = [
        "Patient presents symptoms of malignant neoplasms and tumor growth",
        "Digestive disorder with abdominal pain and acute gastritis",
        "Nervous system stroke and neurological motor deficiency",
        "Cardiovascular infarction with elevated blood pressure",
        "General pathological condition with fever and fatigue",
    ]
    labels = [1, 2, 3, 4, 5]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 1), lowercase=True)),
        ("clf", LinearSVC(C=1.0, random_state=42)),
    ])
    pipeline.fit(texts, labels)
    return pipeline, texts, labels


def test_export_pipeline_to_onnx_invalid_input():
    """Valida se export_pipeline_to_onnx rejeita entradas que não sejam Pipeline."""
    with pytest.raises(ValueError, match="não é uma instância de sklearn.pipeline.Pipeline"):
        export_pipeline_to_onnx("not_a_pipeline")


def test_export_and_onnx_inference(tmp_path, trained_toy_pipeline):
    """Testa o fluxo completo de exportação para ONNX e execução da inferência."""
    pipeline, texts, labels = trained_toy_pipeline
    onnx_path = tmp_path / "test_model.onnx"

    exported_path = export_pipeline_to_onnx(pipeline, output_path=onnx_path)
    assert exported_path.exists()
    assert exported_path.stat().st_size > 0

    onnx_engine = ONNXInferenceEngine(model_path=exported_path)
    results = onnx_engine.predict("Acute stroke and neurological disorder")
    assert len(results) == 1
    assert "label" in results[0]
    assert "class_name" in results[0]
    assert "latency_ms" in results[0]
    assert results[0]["label"] in labels


def test_prediction_parity_between_sklearn_and_onnx(tmp_path, trained_toy_pipeline):
    """Valida que a inferência do ONNX produz exatamente as mesmas predições do Scikit-Learn."""
    pipeline, texts, labels = trained_toy_pipeline
    onnx_path = tmp_path / "test_model.onnx"
    export_pipeline_to_onnx(pipeline, output_path=onnx_path)

    sklearn_engine = SklearnInferenceEngine(pipeline=pipeline)
    onnx_engine = ONNXInferenceEngine(model_path=onnx_path)

    test_queries = [
        "Patient shows malignant cancer and cell neoplasms",
        "Gastric pain and digestive issues",
        "Severe cerebral stroke and nervous disorder",
        "High blood pressure and cardiovascular failure",
        "General symptoms of malaise and fever",
    ]

    sklearn_preds = sklearn_engine.predict(test_queries)
    onnx_preds = onnx_engine.predict(test_queries)

    assert len(sklearn_preds) == len(onnx_preds) == len(test_queries)
    for sk_res, onnx_res in zip(sklearn_preds, onnx_preds):
        assert sk_res["label"] == onnx_res["label"]
        assert sk_res["class_name"] == onnx_res["class_name"]


def test_load_label_mapping(tmp_path):
    """Testa o carregamento de mapeamento de classes padrão e a partir de JSON."""
    default_mapping = load_label_mapping(tmp_path / "non_existent.json")
    assert default_mapping == DEFAULT_LABEL_TO_CLASS

    custom_json = tmp_path / "meta.json"
    custom_json.write_text('{"label_to_class": {"1": "urgente", "2": "normal"}}', encoding="utf-8")
    loaded = load_label_mapping(custom_json)
    assert loaded[1] == "urgente"
    assert loaded[2] == "normal"


def test_get_inference_engine_factory(tmp_path, trained_toy_pipeline):
    """Testa a factory de motores de inferência e validação de parâmetros."""
    pipeline, _, _ = trained_toy_pipeline
    onnx_path = tmp_path / "test.onnx"
    export_pipeline_to_onnx(pipeline, output_path=onnx_path)

    onnx_eng = get_inference_engine(backend="onnx", model_path=onnx_path)
    assert isinstance(onnx_eng, ONNXInferenceEngine)

    with pytest.raises(ValueError, match="Backend desconhecido"):
        get_inference_engine(backend="unsupported_backend")
