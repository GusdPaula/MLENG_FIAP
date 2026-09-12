# Relatório de Benchmark de Latência: Scikit-Learn vs ONNX Runtime

**Data/Execução:** 2026-08-29 16:19:40
**Iterações Unitárias (batch_size=1):** 500
**Amostras Únicas de Teste:** 2888

## 📊 Tabela Comparativa de Performance

| Métrica | Scikit-Learn (Baseline) | ONNX Runtime (Otimizado) | Variação / Ganho |
| :--- | :---: | :---: | :---: |
| **Latência Média** | 0.1744 ms | 0.0712 ms | -0.1032 ms |
| **Mediana (P50)** | **0.1732 ms** | **0.0702 ms** | **+59.5% de redução** |
| **Percentil 90 (P90)** | 0.1985 ms | 0.0859 ms | -0.1125 ms |
| **Percentil 95 (P95)** | 0.2066 ms | 0.0932 ms | -0.1134 ms |
| **Percentil 99 (P99)** | 0.2278 ms | 0.1062 ms | -0.1217 ms |
| **Throughput Estimado** | 5729.5 req/s | **14023.1 req/s** | **+144.8%** |
| **Paridade Numérica** | — | **100.00%** | Paridade perfeita de predição |

## 🎯 Conclusões para o Vídeo STAR e Documentação
1. **Otimização de Latência:** O modelo exportado para ONNX Runtime reduz a latência mediana (P50), garantindo tempo de resposta ultrarrápido para a triagem em tempo real na API REST.
2. **Consistência Clínica:** A paridade entre os modelos Scikit-Learn e ONNX Runtime alcançou 100.00%, preservando integralmente a acurácia diagnóstica sem degradação.
