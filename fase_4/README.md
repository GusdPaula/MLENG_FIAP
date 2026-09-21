# Tech Challenge Fase 4 — Observabilidade, Drift, Qualidade e Governança em Produção

[![CI](https://github.com/seu-org/MLENG_FIAP/actions/workflows/fase_4-ci.yml/badge.svg)](https://github.com/seu-org/MLENG_FIAP/actions/workflows/fase_4-ci.yml)

## Contexto

Uma fintech de concessão de crédito possui um modelo de Credit Scoring em produção. Com recentes mudanças econômicas (inflação, desemprego, novos perfis de clientes), o modelo sofre **degradação silenciosa**. Este projeto implementa a camada de Sustentação e Confiabilidade para detectar essa degradação automaticamente.

---

## Como Rodar

### Pré-requisitos
- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)

### Instalação

```bash
cd fase_4
poetry install
```

### Executar a pipeline completa

```bash
poetry run pipeline
```

A pipeline executa as 4 etapas em sequência e gera:
- `credit_scoring/reports/drift_report.html` — relatório visual de drift
- `credit_scoring/reports/drift_metrics.json` — métricas em JSON
- `mlruns/` — experimentos rastreados no MLflow

### Visualizar MLflow UI

```bash
mlflow ui --port 5000
# Acesse: http://localhost:5000
```

### Testes

```bash
poetry run pytest -v --cov=credit_scoring --cov-report=term-missing
```

### Lint

```bash
poetry run ruff check .
poetry run ruff format --check .
```

---

## Estrutura do Projeto

```
fase_4/
├── credit_scoring/
│   ├── data/
│   │   ├── generate.py          # Dataset sintético German Credit (referência)
│   │   ├── contracts.py         # Contrato Pandera (≥3 regras rígidas)
│   │   └── simulate_drift.py    # Simulação de drift de produção
│   ├── models/
│   │   └── train.py             # LogisticRegression + XGBoost + MLflow
│   ├── monitoring/
│   │   ├── drift_detector.py    # Evidently AI (PSI, KS, HTML report)
│   │   └── mlflow_logger.py     # Centralização de métricas no MLflow
│   └── pipeline.py              # Orquestrador das 4 etapas
├── tests/
│   ├── conftest.py
│   ├── test_contracts.py
│   ├── test_drift.py
│   └── test_pipeline.py
├── pyproject.toml
└── README.md
```

---

## Etapas do Projeto

### Etapa 1 — Validação de Dados e Contratos (Pandera)

O contrato em `credit_scoring/data/contracts.py` define **5 regras rígidas**:

| Regra | Descrição |
|-------|-----------|
| `idade > 18` | Menores não podem solicitar crédito |
| `renda_mensal not null AND > 0` | Renda obrigatória e positiva |
| `valor_emprestimo > 0` | Valor do empréstimo deve ser positivo |
| `score_credito ∈ [300, 900]` | Score dentro da escala válida |
| `sem duplicatas` | Dataset sem linhas repetidas |

**Demonstração de falha:** o dataset de produção é intencionalmente corrompido com nulos em `finalidade` e linhas duplicadas. O contrato detecta e bloqueia a ingestão, logando todas as violações.

### Etapa 2 — Simulação e Detecção de Drift

O dataset de produção simula mudanças econômicas:

| Feature | Alteração | Motivação |
|---------|-----------|-----------|
| `renda_mensal` | +80% média | Inflação / novos perfis premium |
| `score_credito` | -30% média | Crise de crédito |
| `anos_emprego` | -5 anos média | Instabilidade no mercado |
| `inadimplente` | +15 pp taxa | Concept drift (mudança de comportamento) |

**Testes estatísticos aplicados:**
- **PSI (Population Stability Index)**: PSI > 0.2 indica drift significativo
- **Kolmogorov-Smirnov**: p < 0.05 confirma distribuições diferentes

### Etapa 3 — Observabilidade (Evidently + MLflow)

**Evidently AI** compara referência vs produção com:
- `DataDriftPreset` — detecta mudança de distribuição por feature
- `DataQualityPreset` — monitora nulos, outliers, valores inválidos
- `TargetDriftPreset` — detecta concept drift na variável alvo

**MLflow** registra para cada execução:
- Métricas de drift por feature (`psi_renda_mensal`, `psi_score_credito`, ...)
- Performance do modelo (`ref_accuracy`, `prod_accuracy`, `accuracy_delta`)
- Alertas automáticos: tag `drift_alert=True` quando drift detectado
- Artefatos: relatórios HTML anexados ao run

---

## Governança e LGPD (Etapa 4)

### 1. Mapeamento de Dados Pessoais (PII)

Este projeto usa **dataset sintético** sem dados pessoais reais. Em produção, os campos sensíveis seriam:

| Campo | Classificação LGPD | Tratamento |
|-------|-------------------|------------|
| CPF/nome | Dado pessoal identificável | **Não coletado** (Privacy by Design) |
| Idade | Dado pessoal | Pseudonimizado (faixa etária) |
| Renda | Dado financeiro sensível | Criptografia em repouso (AES-256) |
| Score de crédito | Dado derivado | Calculado internamente, não armazenado |

**Privacy by Design:** o modelo opera sobre **variáveis derivadas** (scores, faixas), nunca sobre identificadores diretos. Nenhum dado nominativo transita no pipeline de ML.

### 2. Base Legal (Art. 7º LGPD)

A decisão de crédito se enquadra em:
- **Art. 7º, V** — Execução de contrato (análise de risco na concessão de crédito)
- **Art. 7º, IX** — Legítimo interesse do controlador (prevenção de fraudes e inadimplência)

O titular tem direito à **revisão humana** de decisões automatizadas (Art. 20 LGPD).

### 3. Plano de Retenção de Dados

| Dado | Retenção | Justificativa |
|------|----------|---------------|
| Dados de solicitação | 5 anos | Regulação BACEN / CMN nº 4.656 |
| Logs de decisão | 5 anos | Auditoria e compliance regulatório |
| Modelos treinados | 3 versões anteriores | Rollback e auditoria de vieses |
| Dados de monitoramento (drift) | 2 anos | Análise de tendências e governança |

Após os prazos, os dados são **anonimizados ou deletados** de forma irreversível.

### 4. Mitigação de Vieses

O modelo é avaliado periodicamente sob critérios de equidade:
- **Paridade demográfica**: taxa de aprovação similar entre faixas etárias (18-30, 31-50, 51+)
- **Equidade preditiva**: mesma taxa de falsos positivos entre grupos
- Reavaliação obrigatória sempre que `accuracy_delta > 5%` for detectado

### 5. Análise Causal da Degradação

```
Causa Raiz: Mudança macroeconômica
    │
    ├─► Inflação → renda aumenta → distribuição de renda diverge (Data Drift)
    │                                    └─► Modelo subestima risco de clientes de alta renda
    │
    ├─► Crise → score_credito cai → distribuição de score diverge (Data Drift)
    │                                    └─► Modelo não reconhece novo perfil de risco baixo
    │
    └─► Combinação → taxa de inadimplência muda (Concept Drift)
                          └─► Relação histórica feature→target não se aplica mais
```

**Prescrição:** retreinar o modelo a cada trimestre ou quando `PSI > 0.2` em qualquer feature crítica, usando dados dos últimos 6 meses como nova referência.

### 6. Registro de Tratamento (ROPA)

| Item | Detalhe |
|------|---------|
| Finalidade | Avaliação automatizada de risco de crédito |
| Controlador | Fintech (razão social) |
| Operador | Equipe de ML Engineering |
| Transferência internacional | Não aplicável (dados em nuvem nacional) |
| DPO | Indicar nome/contato do encarregado |

---

## Vídeo STAR

> **Link:** [a ser inserido após gravação]

**Roteiro (5 minutos):**
- **Situation** (1 min): Risco da degradação silenciosa — o modelo toma decisões erradas sem que a equipe perceba
- **Task** (1 min): Construir camada de confiabilidade: contratos de dados, detecção de drift, observabilidade
- **Action** (2 min): Demonstração da pipeline — contrato falhando, Evidently detectando drift, MLflow registrando alertas
- **Result** (1 min): Relatório HTML evidenciando as features degradadas, alertas gerados, lições aprendidas

---

## Bibliotecas Utilizadas

| Biblioteca | Versão | Uso |
|------------|--------|-----|
| scikit-learn | ≥1.3 | Modelo baseline (LogisticRegression) |
| xgboost | ≥2.0 | Modelo campeão |
| pandera | ≥0.18 | Data contracts |
| evidently | ≥0.4 | Detecção de drift e relatórios HTML |
| mlflow | ≥2.10 | Rastreamento de experimentos e observabilidade |
| scipy | — | Testes estatísticos KS e PSI |
