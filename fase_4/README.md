# Sistema de Observabilidade Contínua, Detecção de Drift, Qualidade de Dados e Governança em Produção

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](./pyproject.toml)
[![Pandera](https://img.shields.io/badge/Data_Contracts-Pandera-00A67E?style=for-the-badge)](./src/default_pred/data/contract.py)
[![Evidently AI](https://img.shields.io/badge/Drift_Detection-Evidently_AI-ED1C24?style=for-the-badge)](./src/default_pred/models/drift.py)
[![MLflow](https://img.shields.io/badge/Observability-MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](./src/default_pred/observability/)
[![LGPD & Causalidade](https://img.shields.io/badge/Governan%C3%A7a-LGPD_%26_Causal_DAG-6366F1?style=for-the-badge)](./docs/GOVERNANCA_LGPD.md)
[![Apresentação STAR](https://img.shields.io/badge/Apresenta%C3%A7%C3%A3o_HTML-M%C3%A9todo_STAR-F59E0B?style=for-the-badge)](./docs/presentation/index.html)

---

## 1. Visão Geral do Problema e Objetivo

Uma fintech de concessão de crédito opera um classificador de **Credit Scoring** (predição de inadimplência / *default*) em produção. Com mudanças macroeconômicas recentes — **inflação persistente e elevação da taxa de juros** — a diretoria identificou indícios de degradação silenciosa nas decisões do modelo.

Este projeto implementa a camada completa de **Sustentação, Confiabilidade, Observabilidade e Governança (MLOps)** para modelos de risco de crédito, estruturada em quatro pilares fundamentais:

* **Validação de Dados e Contratos (*Data Contracts*):** Treinamento do classificador baseline (`LogisticRegression`) sobre o dataset *Lending Club* e proteção da ingestão com regras estritas via **Pandera** (`Contract`), bloqueando registros inválidos (*fail-fast*).
* **Simulação e Detecção Estatística de Drift:** Simulação reprodutível de **Data Drift** ($P(X)$) e **Concept Drift** ($P(Y \mid X)$) sob estresse econômico (`drift_simulation.py`) e detecção estatística automatizada com **Evidently AI** (Kolmogorov-Smirnov e Z-test) combinada ao cálculo explícito de **PSI (*Population Stability Index*)**.
* **Observabilidade de Pipelines e Modelos:** Arquitetura Hexagonal (*Ports & Adapters*) integrando **MLflow** (`mlflow.db` com 3 experimentos dedicados) e **Logging Centralizado Rotativo** (`logs/loan_prediction.log`).
* **Governança, LGPD e Análise Causal:** Mapeamento de PII (*Privacy by Design* e *by Default*), enquadramento na Lei nº 13.709/2018 (LGPD), mitigação de vieses (*Fairness*), Grafo Causal (DAG) da degradação e apresentação executiva no método **STAR**.

---

## 2. Arquitetura do Projeto

```
.
├── data/
│   ├── raw/                        # Dataset bruto Lending Club (baixado sob demanda via kagglehub)
│   ├── processed/                  # Datasets de referência escalonados (X_train.pkl, X_test.pkl, ...)
│   └── entries/                    # Payloads JSON de inferência, falha de contrato e lotes com drift
│       ├── loan_data_row.json              # Registro individual válido
│       ├── loan_data_row_fail.json         # Registro inválido (viola as 3 regras do contrato Pandera)
│       ├── loan_data_drifted.json          # Lote de produção com Data Drift (1.000 amostras)
│       └── loan_data_concept_drifted.json  # Lote gerado pelo pipeline de Data + Concept Drift
├── docs/
│   ├── GOVERNANCA_LGPD.md          # Documento completo de Governança, LGPD, Fairness e Análise Causal
│   └── presentation/
│       └── index.html              # Apresentação interativa e responsiva (Método STAR)
├── logs/
│   └── loan_prediction.log         # Log operacional centralizado com rotação automática (1 MB x 3)
├── models/                         # Artefatos serializados (logistic_regression_model.pkl, scaler.pkl)
├── notebooks/                      # Notebooks exploratórios (EDA.ipynb, Drift.ipynb)
├── src/default_pred/
│   ├── config.py                   # Caminhos, constantes, URIs do MLflow e limiares de alerta
│   ├── data/
│   │   ├── contract.py             # Contrato de dados Pandera (Contract)
│   │   └── load.py                 # Download automático (kagglehub) e loaders com validação
│   ├── features/
│   │   ├── engineering.py          # Sanitização de PII, codificações e limpeza de features
│   │   └── preprocessing.py        # Split estratificado e padronização (StandardScaler)
│   ├── models/
│   │   ├── train.py                # Treinamento da Regressão Logística baseline
│   │   ├── evaluate.py             # Cálculo de Accuracy, Precision, Recall, F1 e ROC-AUC
│   │   ├── predict.py              # Alinhamento de schema, checagem de drift e inferência
│   │   ├── drift.py                # Geração do relatório HTML do Evidently (DataDriftPreset)
│   │   └── drift_simulation.py     # Simulação Clean Code de Data Drift, Concept Drift e cálculo de PSI
│   ├── observability/
│   │   ├── tracker.py              # Porta (Protocol) ExperimentTracker e RunStatus
│   │   ├── mlflow_tracker.py       # Adapter concreto para MLflow (SQLite backend)
│   │   ├── null_tracker.py         # Adapter no-op para execução isolada e testes rápidos
│   │   ├── factory.py              # Fábrica com fallback resiliente (MlflowTracker -> NullTracker)
│   │   ├── run.py                  # Context manager tracked_run (ciclo de vida, duração e status)
│   │   ├── timing.py               # Medição de latência em milissegundos (perf_counter)
│   │   ├── metrics.py              # Formatadores puros de parâmetros e métricas de treino/inferência
│   │   ├── drift_summary.py        # Parser agnóstico do snapshot Evidently e regra de alerta
│   │   └── instrumentation.py      # Orquestradores track_training e track_inference
│   ├── pipelines/
│   │   ├── pipeline_train.py           # Pipeline CLI de treinamento com rastreamento MLflow
│   │   ├── pipeline_predict.py         # Pipeline CLI de predição e validação de contrato
│   │   ├── pipeline_simulate_drift.py  # Pipeline CLI de simulação Data/Concept Drift + PSI
│   │   └── pipeline_drift.py           # Pipeline CLI de detecção Evidently e alerta
│   └── utils/
│       └── logging.py              # Configuração idempotente do RotatingFileHandler
├── tests/                          # Suíte de testes unitários e de integração (pytest + coverage)
├── data_drift_report.html          # Relatório interativo HTML gerado pelo Evidently AI
└── pyproject.toml                  # Dependências gerenciadas via uv, configuração Ruff e Pytest
```

---

## 3. Guia de Instalação e Execução Reprodutível

### 3.1 Pré-requisitos e Instalação
O projeto utiliza Python $\ge 3.12$ e o gerenciador de pacotes [**uv**](https://docs.astral.sh/uv/). A partir do diretório raiz:

```bash
uv sync
```

> **Nota sobre o Dataset:** Caso `data/raw/loan_data.csv` não exista localmente, a função [`download_dataset_if_missing()`](./src/default_pred/data/load.py) realiza automaticamente o download do *Lending Club Loan Data* via `kagglehub` e prepara a amostra de 5.000 registros na primeira execução.

### 3.2 Executando os Pipelines

| Caso de Uso | Comando de Execução | Resultado Esperado |
| :--- | :--- | :--- |
| **1. Treinar Modelo Base** | `uv run python src/default_pred/pipelines/pipeline_train.py` | Processa `loan_data.csv`, salva o dataset de referência `data/processed/X_train.pkl`, serializa `logistic_regression_model.pkl` e registra a run no experimento `credit_training` do MLflow. |
| **2. Inferência Válida (Lote/Registro)** | `uv run python src/default_pred/pipelines/pipeline_predict.py --input data/entries/loan_data_row.json` | Valida o registro no contrato Pandera (`Data validation passed!`), executa a predição e registra latência e taxa prevista no experimento `credit_inference`. |
| **3. Demonstração de Falha do Contrato (Bloqueio)** | `uv run python src/default_pred/pipelines/pipeline_predict.py --input data/entries/loan_data_row_fail.json` | **Bloqueia a ingestão** com `pandera.errors.SchemaErrors`, impede a inferência e grava a run com `status=failed` (`run_success=0.0`) no MLflow e em `logs/loan_prediction.log`. |
| **4. Simular Data Drift + Concept Drift + PSI** | `uv run python src/default_pred/pipelines/pipeline_simulate_drift.py` | Simula compressão de renda (`-30%`), aumento de empréstimo (`+30%`) e degradação estrutural $P(Y \mid X)$, calcula o **PSI** (`> 0.2`), gera `data/entries/loan_data_concept_drifted.json` e registra o impacto no MLflow (`credit_drift_monitoring`). |
| **5. Detectar Drift e Gerar Relatório Evidently** | `uv run python src/default_pred/pipelines/pipeline_drift.py` | Compara `X_train.pkl` (Referência) contra o lote degradado, salva `data_drift_report.html`, dispara `DRIFT ALERT` nos logs e anexa o artefato HTML no MLflow (`credit_drift_monitoring`). |

### 3.3 Visualizando o Dashboard MLflow UI e Rodando os Testes

Para inspecionar visualmente os 3 experimentos (`credit_training`, `credit_inference`, `credit_drift_monitoring`), métricas, parâmetros e artefatos:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```
*(Acesse `http://127.0.0.1:5000` no navegador).*

Para executar a suíte completa de testes automatizados (incluindo verificação de cobertura $\ge 90\%$ na camada de observabilidade e pipelines monitorados) e o linter:

```bash
uv run pytest --cov --cov-report=term-missing
uv run ruff check src tests
```

---

## 4. Detalhamento Técnico dos Componentes

### Validação de Dados e Contratos (*Data Contracts*)

O contrato de dados foi implementado em [`src/default_pred/data/contract.py`](./src/default_pred/data/contract.py) com **Pandera** (`pa.DataFrameModel`), aplicando validação *lazy* (`Contract.validate(df, lazy=True)`) em [`load_one_entry()`](./src/default_pred/data/load.py) para capturar todas as violações de schema simultaneamente antes da inferência:

| Coluna Validada | Tipo Esperado | Regra Estrita (Pandera) | Racional de Negócio | Valor no Payload Inválido (`loan_data_row_fail.json`) |
| :--- | :---: | :--- | :--- | :--- |
| **`annual_inc`** | `float` | `gt=20_000`, `nullable=False` | Renda anual mínima exigida pela política de crédito e prevenção de erros de escala/digitação. | `1000.0` ❌ *(Reprovado)* |
| **`purpose`** | `str` | `nullable=False` | Finalidade do empréstimo é obrigatória para enquadramento da operação (Art. 6º, I LGPD). | `null` ❌ *(Reprovado)* |
| **`grade`** | `str` | `isin=["A", "B", "C", "D", "E", "F", "G"]`, `nullable=False` | Faixa de risco interna deve pertencer estritamente ao domínio oficial de *rating*. | `"Z"` ❌ *(Reprovado)* |

Além da execução via CLI (`pipeline_predict.py --input data/entries/loan_data_row_fail.json`), o bloqueio é validado continuamente pelo teste automatizado [`tests/test_predict_single_fail_contract.py`](./tests/test_predict_single_fail_contract.py).

---

### Simulação e Detecção Estatística de Drift

O sistema separa claramente o **Dataset de Referência** (`data/processed/X_train.pkl`, aprendido no treino) do **Dataset de Produção Degradado** (`data/entries/loan_data_drifted.json` e `loan_data_concept_drifted.json`):

1. **Simulação de *Data Drift*, *Prediction Drift* e *Concept Drift* ([`src/default_pred/models/drift_simulation.py`](./src/default_pred/models/drift_simulation.py)):**
   * **Decomposição Conjunta $P(X, Y) = P(X) \cdot P(Y \mid X)$:**
     * **Covariate Shift (*Data Drift* — `simulate_data_drift`):** Altera $P(X)$ mantendo $P(Y \mid X)$ constante — simula o choque inflacionário reduzindo a renda anual em **30%** (`annual_inc * 0.7`) e elevando o montante solicitado em **30%** (`loan_amnt * 1.3`).
     * **Prior Probability Shift:** Deslocamento na prevalência global $P(Y=1)$ (de `4.09%` para `51.07%`), exigindo calibração de limiar de decisão (`concept_drift.decision_threshold`).
     * **Concept Drift Real (*Posterior Shift* — `simulate_concept_drift`):** Altera a própria verdade fundamental $P(Y \mid X)$ (dinâmica combinada **Incremental** pela inflação e **Súbita** pelo choque de juros) — clientes anteriormente adimplentes (`Fully Paid`) com `loan_amnt / annual_inc >= 0.25` ou *grades* `D–G` passam a inadimplir (`Charged Off`), derrubando F1-Score (`-45%` relativo, `delta = -0.0521`), Recall (`-9.56 p.p.`), ROC-AUC (`-0.1616`) e Acurácia (`91.81% -> 47.33%`).
2. **Testes Estatísticos Pré-Rótulo ($t \to t+k$) e Detecção Supervisionada Pós-Rótulo ($t+k$ sob *Feedback Delay*):**
   * **Antes da chegada do rótulo ($t \to t+k$, sem *Ground Truth*):**
     * **Teste de Kolmogorov-Smirnov (KS-test — Evidently `DataDriftPreset`):** $p\text{-value} < 0.05$ para variáveis contínuas.
     * **Z-test de Proporções (Evidently `DataDriftPreset`):** $p\text{-value} < 0.05$ para variáveis binárias (*One-Hot*).
     * **PSI de Features e Prediction Drift (`calculate_psi`):** $\text{PSI} = \sum (\text{Real}_i - \text{Base}_i)\ln(\text{Real}_i / \text{Base}_i) \ge 0.20$, acusando drift severo em `psi.annual_inc = 0.4859`, `psi.loan_amnt = 0.4428`, `psi.installment = 0.4604` (Tabela Price) e `psi.int_rate = 1.4231`.
   * **Após a chegada do rótulo ($t+k$, maturação de 90 dias):**
     * **Algoritmo DDM (*Drift Detection Method* — `detect_concept_drift_ddm`):** Monitora a probabilidade de erro sequencial $p_t$ e o desvio padrão $s_t = \sqrt{p_t(1-p_t)/t}$, disparando **Nível de Aviso** ($p_t + s_t \ge p_{\min} + 2s_{\min}$) e **Nível de Drift** ($p_t + s_t \ge p_{\min} + 3s_{\min}$, acionado no índice `first_drift_idx = 600` com salto de erro de `8.16%` para `37.48%`).
3. **Relatório Visual Interativo:** Gerado em [`data_drift_report.html`](./data_drift_report.html), evidenciando que **84,8% das features monitoradas (84 de 99 colunas ativas)** sofreram drift estatisticamente significativo (`share_drifted = 0.85 >= 0.50`), acionando o alerta operacional (`DRIFT ALERT`).

---

### Observabilidade de Pipelines e Modelos (MLflow + Logs)

A camada de observabilidade em [`src/default_pred/observability/`](./src/default_pred/observability/) segue o padrão **Ports and Adapters**: os pipelines dependem apenas do protocolo [`ExperimentTracker`](./src/default_pred/observability/tracker.py), sendo injetados em produção com [`MlflowTracker`](./src/default_pred/observability/mlflow_tracker.py) (persistindo em `sqlite:///mlflow.db`) ou [`NullTracker`](./src/default_pred/observability/null_tracker.py) caso o backend esteja indisponível.

#### Catálogo Completo de Métricas de Saúde Operacionais, Modelo e Drift

Todas as execuções utilizam o context manager [`tracked_run()`](./src/default_pred/observability/run.py), que garante a coleta automática das métricas de ciclo de vida mesmo em caso de exceção:

| Experimento MLflow | Run Name | Tipo | Nome da Métrica / Parâmetro / Artefato | Descrição e Regra Operacional |
| :--- | :--- | :---: | :--- | :--- |
| **Todos os Experimentos** | *(Todas as runs)* | Métrica | `duration_ms` | Tempo total de execução da tarefa do pipeline em milissegundos. |
| **Todos os Experimentos** | *(Todas as runs)* | Métrica | `rows_processed` | Volume total de registros processados na execução (`0` se falhar antes da carga). |
| **Todos os Experimentos** | *(Todas as runs)* | Métrica | `run_success` | Indicador binário de saúde da tarefa: `1.0` (`FINISHED` / `SKIPPED`) ou `0.0` (`FAILED`, ex.: violação do contrato Pandera). |
| **`credit_training`** | `logistic_regression_training` | Parâmetros | `model`, `max_iter`, `class_weight`, `test_size`, `random_state`, `n_train`, `n_features` | Configuração completa de reprodutibilidade do treinamento. |
| **`credit_training`** | `logistic_regression_training` | Métricas | `accuracy`, `precision`, `recall`, `f1_score`, `roc_auc` | Métricas supervisionadas de qualidade do classificador no conjunto de teste (`X_test`). |
| **`credit_training`** | `logistic_regression_training` | Artefato | `model` (MLflow sklearn flavor) | Modelo `LogisticRegression` versionado no registro de artefatos. |
| **`credit_inference`** | `batch_inference` | Métricas | `rows_in`, `latency_ms`, `default_rate_predicted` | Volume recebido, latência pura de inferência (ms) e proporção de inadimplência prevista pelo modelo. |
| **`credit_drift_monitoring`** | `drift_detection` | Parâmetros | `reference_rows`, `production_rows`, `alert_threshold` | Tamanhos das janelas comparadas e limiar configurado (`0.50`). |
| **`credit_drift_monitoring`** | `drift_detection` | Métricas | `n_features`, `n_drifted_features`, `share_drifted`, `alert_triggered`, `drift_score.<feature>` | Quantidade e fração de variáveis degradadas, flag de alerta (`1.0` se `share_drifted >= 0.50`) e score individual por coluna. |
| **`credit_drift_monitoring`** | `drift_detection` | Artefato | `data_drift_report.html` | Relatório visual completo gerado pelo Evidently AI. |
| **`credit_drift_monitoring`** | `drift_simulation` | Métricas | `psi.annual_inc`, `psi.loan_amnt`, `psi.predicted_prob`, `psi_drifted_count`, `ddm.p_min`, `ddm.s_min`, `ddm.p_current`, `ddm.s_current`, `ddm.warning_triggered`, `ddm.drift_triggered`, `baseline.default_rate`, `concept_drift.default_rate`, `concept_drift.default_rate_delta`, `concept_drift.f1_delta`, `concept_drift.recall_delta`, `concept_drift.roc_auc_delta` | Índices PSI de features e de predição (*Feedback Delay* $t \to t+k$), gatilhos supervisionados do algoritmo **DDM** ($t+k$) e perda de performance do modelo causada pelo *Concept Drift*. |

#### Logging Centralizado Rotativo
* **Arquivo:** `logs/loan_prediction.log` (configurado via [`setup_logging()`](./src/default_pred/utils/logging.py) com `RotatingFileHandler` de `1 MB` e até `3` backups).
* **Eventos Registrados:** Início e fim de cada run (`duration_ms`, `status`, `rows`), bloqueios de contrato (`ERROR | default_pred.data.load | Data validation failed.`), alertas de drift (`WARNING | default_pred.pipelines.pipeline_drift | DRIFT ALERT: 33/66 features drifted`) e alertas de PSI.

---

### Governança, LGPD, Mitigação de Vieses e Análise Causal

A documentação profunda de Governança e Análise Causal encontra-se em **[`docs/GOVERNANCA_LGPD.md`](./docs/GOVERNANCA_LGPD.md)**. Abaixo está a síntese executiva dos pilares implementados:

| Pilar de Governança | Resumo Executivo da Implementação | Referência Detalhada |
| :--- | :--- | :--- |
| **1. *Privacy by Design* & Mapeamento de PII** | Descarte preventivo de identificadores diretos (`id`, `member_id`, `url`) e textos abertos (`title`, `emp_title`, `desc`) em [`engineering.py`](./src/default_pred/features/engineering.py). Pseudonimização na origem de `zip_code` (3 dígitos) e agregação estadual em `addr_state`. | [Seção 1 de `docs/GOVERNANCA_LGPD.md`](./docs/GOVERNANCA_LGPD.md#1-privacy-by-design-e-privacy-by-default--mapeamento-de-pii) |
| **2. Enquadramento Legal (LGPD)** | Fundamentação no **Art. 7º, X (Proteção do Crédito)** e **Art. 7º, V (Execução de Contrato)**. Atendimento ao **Art. 20 (Explicabilidade e Revisão)** via coeficientes transparentes ($\beta_j$) da Regressão Logística. | [Seção 2 de `docs/GOVERNANCA_LGPD.md`](./docs/GOVERNANCA_LGPD.md#2-enquadramento-jurídico-na-lgpd-lei-nº-137092018) |
| **3. Política de Retenção e Descarte (Art. 15 e 16)** | Tabela normativa definindo prazos de guarda e descarte para `data/raw/`, `data/processed/`, `data/entries/`, `models/`, rotação circular de `logs/loan_prediction.log` e *garbage collection* de `mlflow.db`. | [Seção 3 de `docs/GOVERNANCA_LGPD.md`](./docs/GOVERNANCA_LGPD.md#3-política-de-retenção-armazenamento-e-descarte-seguro-art-15-e-art-16-da-lgpd) |
| **4. Equidade Algorítmica (*Fairness*)** | Diagnóstico do risco de *redlining* indireto em `zip_code` e `addr_state`, com prescrição da **Regra dos 80% (*Disparate Impact Ratio*)**, monitoramento de *Equal Opportunity* e remoção programada de `zip_code`. | [Seção 4 de `docs/GOVERNANCA_LGPD.md`](./docs/GOVERNANCA_LGPD.md#4-mitigação-de-vieses-algorítmicos-e-justiça-fairness) |
| **5. Análise Causal (DAG) e Prescrição** | Grafo Causal conectando o choque macroeconômico ($Z$: inflação/juros) ao *Data Drift* (`annual_inc` -30%, `loan_amnt` +30%) e ao *Concept Drift* (aumento de $P(Y=1 \mid X)$), acompanhado de protocolo de resposta em 4 níveis. | [Seção 5 de `docs/GOVERNANCA_LGPD.md`](./docs/GOVERNANCA_LGPD.md#5-análise-causal-da-degradação-do-modelo-em-produção) |

---

## 5. Apresentação Executiva — Método STAR

Para apoiar a demonstração executiva estruturada no método **STAR** (*Situation, Task, Action, Result*), o projeto disponibiliza uma apresentação web interativa, estilizada e responsiva:

* **Apresentação HTML (Slides STAR):** [`docs/presentation/index.html`](./docs/presentation/index.html)
* **Relatório Estatístico do Evidently AI:** [`data_drift_report.html`](./data_drift_report.html)
* **Documento de Governança e LGPD:** [`docs/GOVERNANCA_LGPD.md`](./docs/GOVERNANCA_LGPD.md)
