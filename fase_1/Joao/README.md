# Telco Customer Churn

Fase 1 do Tech Challenge focado em tres eixos:
**planejamento do projeto**, **analise exploratoria** e **baselines com metricas de negocio**.

## O que foi feito

### 1. ML Canvas (`docs/ML_CANVAS.md`)

Estruturei o ML Canvas do projeto respondendo as perguntas fundamentais antes de
qualquer linha de modelagem:

- Mapeamento de stakeholders (diretoria, time de retencao, equipe de dados)
- Estimativa de impacto financeiro: ~$1.46M/ano em receita perdida com taxa de
  churn de 26.5%
- Calculo de ROI da solucao: ~$284K/ano de retorno liquido assumindo 20% de
  prevencao de churns
- Definicao de metricas tecnicas (AUC-ROC > 0.80, Recall > 0.70) e SLOs
  (P95 < 200ms, 99.5% uptime)
- Documentacao de restricoes, riscos de vieses (gender, SeniorCitizen) e
  estrategias de mitigacao

### 2. EDA (`notebooks/01_eda.ipynb`)

Analise exploratoria completa usando Pandas sobre o dataset estendido IBM Telco
Customer Churn (33 colunas, xlsx)

Fluxo da analise:

1. **Load & inspect**: schema, dtypes, describe, nulls
2. **Column triage**: classificacao das 33 colunas em KEEP / DROP / EDA-only,
   com justificativa (leakage para Churn Score/CLTV/Churn Reason, zero-variance
   para Count/Country/State, alta cardinalidade para Zip Code/Lat Long)
3. **Data cleaning**: Total Charges nulls (11 para Tenure Months=0), duplicatas
4. **Target**: distribuicao do Churn (73.5% / 26.5%), ratio ~2.8:1
5. **Numerica univariada**: histogramas sobrepostos + boxplots por classe de
   churn + correlacoes de Pearson
6. **Categorica bivariada**: value counts + taxa de churn por categoria com
   graficos de barras horizontais para todas as 16 features categoricas
7. **Analise geografica**: todos os clientes estao na California; churn rate
   por City (1,129 cidades unicas), top 15 cidades
8. **Multivariada**: cruzamentos Contract x Internet Service, Payment Method x
   Contract, buckets de tenure e Monthly Charges, contagem de servicos ativos
9. **Outliers**: analise IQR para numericas
10. **Correlacao**: heatmap incluindo Latitude e Longitude
11. **Data readiness summary**: resumo com contagem de features, decisoes e
    colunas descartadas

### Principais achados

| Insight | Detalhe |
|---------|---------|
| Month-to-month + Fiber optic | Combinacao com maior taxa de churn |
| Tenure curto (0-12 meses) | Faixa com maior propensao a cancelamento |
| Cobranca alta (>$70/mes) | Correlacao positiva com churn |
| Servicos de protecao | Online Security e Tech Support reduzem churn |
| Electronic check | Metodo de pagamento com maior taxa de churn |
| Leakage columns | Churn Score, CLTV, Churn Reason e Churn Value descartados |
| Geografia | Todos os clientes sao da California (State = zero variance) |

### Decisoes de preprocessing documentadas

- Remover colunas de ID, zero-variance e leakage (10 colunas)
- Marcar City, Latitude, Longitude como EDA-only (nao usar na modelagem)
- Fill Total Charges null com 0 (clientes novos com Tenure Months=0)
- OneHotEncoder para categoricas, StandardScaler para numericas
- Considerar `class_weight` ou `pos_weight` para tratar desbalanceamento

### 3. Baselines & Metricas (`notebooks/02_baselines.ipynb`)

Define o framework de avaliacao (metricas tecnicas + metrica de negocio) e treina
dois modelos baseline que estabelecem o piso de performance.

**Metricas tecnicas:**

| Metrica | Por que usar | Ressalva |
|---------|-------------|----------|
| AUC-ROC | Threshold-agnostic, avalia ranking | Pode ser otimista com classes desbalanceadas |
| PR-AUC | Foca na classe positiva (churners) | Melhor que AUC-ROC para desbalanceamento |
| F1-Score | Resumo em um numero (Precision x Recall) | Depende do threshold (default 0.5) |

**Metrica de negocio -- Custo de Churn Evitado:**

Usa o `monthly_charges` real de cada cliente (nao um valor fixo):
- FP (oferta desnecessaria): $50 fixo
- FN (cliente perdido): `monthly_charges * 12 meses` por cliente (range: $219 -- $1,425)
- Escolhemos `monthly_charges * 12` apos testar tambem o CLTV do dataset original.
  O MC*12 e mais transparente, esta presente nos dados limpos e usa uma premissa
  defensavel (1 ano de receita perdida).

**Baselines:**

| Modelo | F1 | Recall | ROC-AUC | PR-AUC | Custo |
|--------|----|--------|---------|--------|-------|
| DummyClassifier | 0.00 | 0.00 | 0.50 | 0.27 | $326,579 |
| LogisticRegression (balanced) | 0.62 | 0.78 | 0.85 | 0.65 | $75,929 |

A LogisticRegression reduz o custo de negocio em **$250,650 (76.8%)** em relacao ao
Dummy. Todos os experimentos sao registrados no MLflow.

## Estrutura

```
Joao/
  README.md
  docs/
    ML_CANVAS.md
    Tech Challenge Fase 01.pdf
  notebooks/
    01_eda.ipynb
    02_baselines.ipynb
  data/
    raw/
      Telco_customer_churn.xlsx   (gitignored)
    processed/
      telco_churn_clean.parquet   (output do 01_eda)
```

## Requisitos

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.9
seaborn>=0.13
openpyxl>=3.1
scikit-learn>=1.3
mlflow>=2.10
pyarrow>=14.0
```
