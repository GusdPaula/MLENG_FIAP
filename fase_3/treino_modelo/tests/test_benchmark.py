"""Testes unitários para o módulo de benchmark (treino_modelo/optimize/benchmark.py)."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from treino_modelo.optimize.benchmark import (
    compute_prediction_parity,
    generate_markdown_report,
    measure_latencies,
    run_benchmark,
)
from treino_modelo.optimize.onnx_exporter import export_pipeline_to_onnx


@pytest.fixture
def mock_engine():
    """Engine simulada para testes rápidos de medição."""
    engine = MagicMock()
    engine.predict.return_value = [{"label": 1, "class_name": "neoplasms", "latency_ms": 0.5}]
    return engine


def test_measure_latencies_success(mock_engine):
    """Testa o cálculo das estatísticas de latência com a engine simulada."""
    texts = ["Sample medical report 1", "Sample medical report 2"]
    metrics = measure_latencies(mock_engine, texts, iterations=20, warmup=5)

    expected_keys = {
        "mean_ms", "std_ms", "min_ms", "p50_ms",
        "p90_ms", "p95_ms", "p99_ms", "max_ms", "throughput_rps"
    }
    assert expected_keys.issubset(metrics.keys())
    assert metrics["p50_ms"] >= 0
    assert metrics["throughput_rps"] > 0
    assert mock_engine.predict.call_count == 25  # 5 warmup + 20 iterations


def test_measure_latencies_empty_texts(mock_engine):
    """Testa erro ao passar lista de textos vazia."""
    with pytest.raises(ValueError, match="não pode estar vazia"):
        measure_latencies(mock_engine, [], iterations=10, warmup=2)


def test_compute_prediction_parity():
    """Testa o cálculo da taxa de paridade entre duas engines."""
    sk_engine = MagicMock()
    onnx_engine = MagicMock()

    # 3 amostras: 2 iguais, 1 diferente -> 66.66%
    sk_engine.predict.return_value = [{"label": 1}, {"label": 2}, {"label": 3}]
    onnx_engine.predict.return_value = [{"label": 1}, {"label": 2}, {"label": 4}]

    rate = compute_prediction_parity(sk_engine, onnx_engine, ["t1", "t2", "t3"])
    assert round(rate, 2) == 66.67

    # Lista vazia retorna 100.0%
    assert compute_prediction_parity(sk_engine, onnx_engine, []) == 100.0


def test_generate_markdown_report():
    """Valida se o relatório em Markdown contém tabela formatada e seções esperadas."""
    sk_metrics = {"mean_ms": 2.0, "p50_ms": 1.9, "p90_ms": 2.2, "p95_ms": 2.5, "p99_ms": 3.0, "throughput_rps": 500.0}
    onnx_metrics = {"mean_ms": 0.8, "p50_ms": 0.7, "p90_ms": 0.9, "p95_ms": 1.0, "p99_ms": 1.2, "throughput_rps": 1200.0}

    report = generate_markdown_report(sk_metrics, onnx_metrics, parity_rate=100.0, sample_count=10, iterations=100)
    assert "# Relatório de Benchmark de Latência" in report
    assert "Scikit-Learn (Baseline)" in report
    assert "ONNX Runtime (Otimizado)" in report
    assert "100.00%" in report


def test_run_benchmark_end_to_end(tmp_path):
    """Testa a execução completa de run_benchmark em arquivos temporários."""
    import joblib

    texts = [
        "Patient presents neoplasms and tumor symptoms",
        "Digestive disorder and severe gastritis pain",
        "Nervous system stroke and neurological deficiency",
        "Cardiovascular failure and hypertension",
        "General pathological condition and malaise",
    ]
    labels = [1, 2, 3, 4, 5]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 1))),
        ("clf", LinearSVC(C=1.0, random_state=42)),
    ])
    pipeline.fit(texts, labels)

    sk_path = tmp_path / "model.pkl"
    onnx_path = tmp_path / "model.onnx"
    report_path = tmp_path / "report.md"

    joblib.dump(pipeline, sk_path)
    export_pipeline_to_onnx(pipeline, output_path=onnx_path)

    results = run_benchmark(
        sklearn_path=sk_path,
        onnx_path=onnx_path,
        report_path=report_path,
        iterations=10,
        warmup=2,
        test_texts=texts,
    )

    assert report_path.exists()
    assert results["parity_rate"] == 100.0
    assert "sklearn" in results
    assert "onnx" in results
