# POS TECH FIAP — Tech Challenge (Fase 3)

> **Deploy de Modelo em Produção com Pipeline CI/CD, Monitoramento e Otimização de Latência.**

Este projeto implementa uma solução completa de MLOps para triagem automática de laudos médicos, cobrindo ingestão, validação, treinamento, orquestração com Airflow e otimização de latência com ONNX Runtime.

---

## 🛠️ Gerenciamento de Dependências com Poetry

O projeto utiliza **Poetry** para gerenciar dependências, ambientes virtuais e scripts de execução.

### Instalação

1. **Instalar dependências principais e de desenvolvimento:**
   ```bash
   poetry install
   ```

2. **Instalar com extras (Airflow / API / Notebooks):**
   ```bash
   # Com suporte a Airflow
   poetry install --all-extras

   # Ou instalando grupos específicos (ex: notebooks)
   poetry install --with notebooks
   ```

3. **Ativar o ambiente virtual:**
   ```bash
   poetry shell
   ```

---

## 🚀 Execução da Pipeline e Comandos (CLI Scripts)

O Poetry já possui scripts configurados para facilitar a execução:

### 1. Treinamento Completo da Pipeline
Executa Ingestão $\rightarrow$ Validação $\rightarrow$ Treinamento (Scikit-Learn) $\rightarrow$ Avaliação $\rightarrow$ Exportação ONNX:
```bash
poetry run train-pipeline
```

### 2. Benchmark de Latência (Scikit-Learn vs ONNX Runtime)
Executa a comparação de latência (P50, P90, P95, P99), throughput e paridade de predições:
```bash
poetry run benchmark --iterations 500 --warmup 50
```

#### 📊 Interpretação dos Resultados do Benchmark

O relatório de benchmark é gerado em `treino_modelo/artifacts/benchmark_report.md` e contém:

**Métricas de Latência:**
- **Média**: Tempo médio de inferência em todas as iterações
- **P50 (Mediana)**: Valor central - representa o desempenho típico
- **P90/P95/P99**: Percentis que mostram o desempenho no pior caso (90º, 95º, 99º percentil)

**Throughput**: Estimativa de requisições por segundo que o sistema consegue processar

**Paridade de Predições**: Porcentagem de predições idênticas entre Scikit-Learn e ONNX Runtime (deve ser 100% para segurança clínica)

**Ganhos de Performance:**
- Porcentagem de redução de latência
- Porcentagem de aumento de throughput

#### 🎯 Resultados do Benchmark Atual

Com base no benchmark mais recente (2026-08-29):

| Métrica | Scikit-Learn | ONNX Runtime | Melhoria |
|---------|--------------|--------------|----------|
| Latência Mediana (P50) | 0.1732 ms | 0.0702 ms | **59.5% mais rápido** |
| Throughput | 5.729 req/s | 14.023 req/s | **144.8% de aumento** |
| Paridade de Predições | — | 100% | **Correspondência perfeita** |

**Principais Conclusões:**
- ONNX Runtime proporciona redução significativa de latência para respostas em tempo real na API
- Paridade perfeita de predições garante que a acurácia clínica é mantida
- Maior throughput permite escalabilidade para mais usuários simultâneos

#### 📈 Como Interpretar Seus Resultados

**Resultados Bons:**
- **Paridade de Predições = 100%**: Crítico para segurança clínica - os modelos devem produzir predições idênticas
- **Redução de Latência > 50%**: Melhoria significativa de performance com otimização ONNX
- **Aumento de Throughput > 100%**: Melhor escalabilidade para API em produção

**Se os Resultados Forem Ruins:**
- Verifique se ambos os modelos usam os mesmos dados de treinamento
- Confirme que a exportação ONNX foi concluída com sucesso
- Certifique-se de que os dados de teste são representativos da carga de produção

### 3. Testes Automatizados (Pytest)
```bash
poetry run pytest
```

### 4. Verificação e Formatação de Código (Ruff)
```bash
# Verificar problemas de lint e imports
poetry run ruff check .

# Aplicar correções automáticas
poetry run ruff check . --fix

# Formatar código
poetry run ruff format .
```

---

## 📂 Estrutura do Projeto

```
fase_3/
├── dags/
│   └── medical_training_dag.py    # DAG semanal do Apache Airflow (@weekly)
├── notebooks/
│   └── explorer.ipynb             # Notebook de análise exploratória
├── treino_modelo/
│   ├── data/
│   │   ├── ingest.py              # Ingestão de dados (Kagglehub)
│   │   └── validate.py            # Validação de dados e schema
│   ├── train/
│   │   ├── train.py               # Treinamento TF-IDF + LinearSVC
│   │   └── evaluate.py            # Avaliação e métricas
│   ├── optimize/
│   │   ├── onnx_exporter.py       # Conversor Scikit-Learn -> ONNX
│   │   ├── inference.py           # Abstração de inferência (Sklearn & ONNX)
│   │   └── benchmark.py           # CLI de medição de latência comparativa
│   ├── artifacts/                 # Modelos gerados (.pkl, .onnx, metadados)
│   ├── tests/                     # Suíte de testes com Pytest
│   └── pipeline.py                # Orquestrador local da pipeline
├── pyproject.toml                 # Configuração do Poetry e dependências
└── README.md                      # Documentação do projeto
```
