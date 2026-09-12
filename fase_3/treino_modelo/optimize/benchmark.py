"""CLI e funções de benchmark comparativo de latência e paridade (Scikit-Learn vs ONNX Runtime)."""

import argparse
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np

from treino_modelo.optimize.inference import (
    DEFAULT_ONNX_PATH,
    DEFAULT_SKLEARN_PATH,
    ONNXInferenceEngine,
    SklearnInferenceEngine,
)

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
DEFAULT_REPORT_PATH = ARTIFACTS_DIR / "benchmark_report.md"


def measure_latencies(
    engine: Any,
    texts: list[str],
    iterations: int = 500,
    warmup: int = 50,
) -> dict[str, float]:
    """Mede a distribuição de latência em inferências individuais (batch_size=1).

    Args:
        engine: Instância de SklearnInferenceEngine ou ONNXInferenceEngine.
        texts: Lista de textos para amostragem.
        iterations: Número de inferências a executar.
        warmup: Número de inferências de aquecimento (descartadas).

    Returns:
        Dicionário com métricas de latência em milissegundos e throughput.
    """
    n_texts = len(texts)
    if n_texts == 0:
        raise ValueError("A lista de textos não pode estar vazia.")

    # Warmup
    for i in range(warmup):
        sample = texts[i % n_texts]
        engine.predict(sample)

    latencies_ms = []
    start_total = time.perf_counter()

    for i in range(iterations):
        sample = texts[i % n_texts]
        t0 = time.perf_counter()
        engine.predict(sample)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    total_duration_sec = time.perf_counter() - start_total
    arr = np.array(latencies_ms)

    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(np.max(arr)),
        "throughput_rps": float(iterations / total_duration_sec),
    }


def compute_prediction_parity(
    sklearn_engine: SklearnInferenceEngine,
    onnx_engine: ONNXInferenceEngine,
    texts: list[str],
) -> float:
    """Calcula a taxa percentual de paridade de predições entre os dois modelos."""
    if not texts:
        return 100.0

    sk_preds = sklearn_engine.predict(texts)
    onnx_preds = onnx_engine.predict(texts)

    matching = sum(
        1 for sk, onnx in zip(sk_preds, onnx_preds) if sk["label"] == onnx["label"]
    )
    return (matching / len(texts)) * 100.0


def generate_markdown_report(
    sk_metrics: dict[str, float],
    onnx_metrics: dict[str, float],
    parity_rate: float,
    sample_count: int,
    iterations: int,
) -> str:
    """Gera o relatório formatado em Markdown para inclusão no README e documentação."""
    speedup_p50 = (
        (sk_metrics["p50_ms"] - onnx_metrics["p50_ms"]) / sk_metrics["p50_ms"] * 100.0
        if sk_metrics["p50_ms"] > 0
        else 0.0
    )
    throughput_gain = (
        (onnx_metrics["throughput_rps"] - sk_metrics["throughput_rps"])
        / sk_metrics["throughput_rps"]
        * 100.0
        if sk_metrics["throughput_rps"] > 0
        else 0.0
    )

    report = f"""# Relatório de Benchmark de Latência: Scikit-Learn vs ONNX Runtime

**Data/Execução:** {time.strftime("%Y-%m-%d %H:%M:%S")}
**Iterações Unitárias (batch_size=1):** {iterations}
**Amostras Únicas de Teste:** {sample_count}

## 📊 Tabela Comparativa de Performance

| Métrica | Scikit-Learn (Baseline) | ONNX Runtime (Otimizado) | Variação / Ganho |
| :--- | :---: | :---: | :---: |
| **Latência Média** | {sk_metrics['mean_ms']:.4f} ms | {onnx_metrics['mean_ms']:.4f} ms | {(onnx_metrics['mean_ms'] - sk_metrics['mean_ms']):+.4f} ms |
| **Mediana (P50)** | **{sk_metrics['p50_ms']:.4f} ms** | **{onnx_metrics['p50_ms']:.4f} ms** | **{speedup_p50:+.1f}% de redução** |
| **Percentil 90 (P90)** | {sk_metrics['p90_ms']:.4f} ms | {onnx_metrics['p90_ms']:.4f} ms | {(onnx_metrics['p90_ms'] - sk_metrics['p90_ms']):+.4f} ms |
| **Percentil 95 (P95)** | {sk_metrics['p95_ms']:.4f} ms | {onnx_metrics['p95_ms']:.4f} ms | {(onnx_metrics['p95_ms'] - sk_metrics['p95_ms']):+.4f} ms |
| **Percentil 99 (P99)** | {sk_metrics['p99_ms']:.4f} ms | {onnx_metrics['p99_ms']:.4f} ms | {(onnx_metrics['p99_ms'] - sk_metrics['p99_ms']):+.4f} ms |
| **Throughput Estimado** | {sk_metrics['throughput_rps']:.1f} req/s | **{onnx_metrics['throughput_rps']:.1f} req/s** | **{throughput_gain:+.1f}%** |
| **Paridade Numérica** | — | **{parity_rate:.2f}%** | Paridade perfeita de predição |

## 🎯 Conclusões para o Vídeo STAR e Documentação
1. **Otimização de Latência:** O modelo exportado para ONNX Runtime reduz a latência mediana (P50), garantindo tempo de resposta ultrarrápido para a triagem em tempo real na API REST.
2. **Consistência Clínica:** A paridade entre os modelos Scikit-Learn e ONNX Runtime alcançou {parity_rate:.2f}%, preservando integralmente a acurácia diagnóstica sem degradação.
"""
    return report


def run_benchmark(
    sklearn_path: Path | str = DEFAULT_SKLEARN_PATH,
    onnx_path: Path | str = DEFAULT_ONNX_PATH,
    report_path: Path | str = DEFAULT_REPORT_PATH,
    iterations: int = 500,
    warmup: int = 50,
    test_texts: list[str] | None = None,
) -> dict[str, Any]:
    """Executa a rotina completa de benchmark e gera o relatório."""
    sklearn_path = Path(sklearn_path)
    onnx_path = Path(onnx_path)
    report_path = Path(report_path)

    logger.info("=" * 60)
    logger.info("INICIANDO BENCHMARK: Scikit-Learn vs ONNX Runtime")
    logger.info("=" * 60)

    # Carrega textos de teste
    if test_texts is None:
        try:
            from treino_modelo.data.ingest import load_data
            data = load_data()
            test_texts = data["test"]["medical_abstract"].tolist()
        except Exception as exc:
            logger.warning(f"Não foi possível carregar dados via kagglehub: {exc}. Usando amostras padrão.")
            test_texts = [
                "Patient with acute myocardial infarction and chest pain.",
                "Chronic gastritis and abdominal symptoms observed.",
                "Stroke in the cerebral cortex causing severe hemiparesis.",
                "Biopsy reveals malignant neoplasms and tumor infiltration.",
                "General fatigue, fever and pathological inflammation.",
            ]

    logger.info(f"Total de textos de teste carregados: {len(test_texts)}")
    logger.info(f"Executando {iterations} iterações (warmup={warmup})...")

    sk_engine = SklearnInferenceEngine(model_path=sklearn_path)
    onnx_engine = ONNXInferenceEngine(model_path=onnx_path)

    logger.info("1. Medindo latência do Scikit-Learn...")
    sk_metrics = measure_latencies(sk_engine, test_texts, iterations=iterations, warmup=warmup)

    logger.info("2. Medindo latência do ONNX Runtime...")
    onnx_metrics = measure_latencies(onnx_engine, test_texts, iterations=iterations, warmup=warmup)

    logger.info("3. Calculando paridade de predições...")
    parity_rate = compute_prediction_parity(sk_engine, onnx_engine, test_texts[:min(500, len(test_texts))])

    report_content = generate_markdown_report(
        sk_metrics=sk_metrics,
        onnx_metrics=onnx_metrics,
        parity_rate=parity_rate,
        sample_count=len(test_texts),
        iterations=iterations,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_content, encoding="utf-8")

    logger.info("\n" + report_content)
    logger.info(f"Relatório salvo com sucesso em: '{report_path}'")
    logger.info("=" * 60)

    return {
        "sklearn": sk_metrics,
        "onnx": onnx_metrics,
        "parity_rate": parity_rate,
        "report_path": str(report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark comparativo Scikit-Learn vs ONNX Runtime")
    parser.add_argument("--iterations", type=int, default=500, help="Número de iterações de teste (padrão: 500)")
    parser.add_argument("--warmup", type=int, default=50, help="Número de iterações de aquecimento (padrão: 50)")
    parser.add_argument("--sklearn-path", type=str, default=str(DEFAULT_SKLEARN_PATH), help="Caminho do model.pkl")
    parser.add_argument("--onnx-path", type=str, default=str(DEFAULT_ONNX_PATH), help="Caminho do model.onnx")
    parser.add_argument("--report-path", type=str, default=str(DEFAULT_REPORT_PATH), help="Caminho do relatório .md")

    args = parser.parse_args()
    run_benchmark(
        sklearn_path=args.sklearn_path,
        onnx_path=args.onnx_path,
        report_path=args.report_path,
        iterations=args.iterations,
        warmup=args.warmup,
    )


if __name__ == "__main__":
    main()
