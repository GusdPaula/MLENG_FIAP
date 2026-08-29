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
