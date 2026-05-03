# Model Card - Telco Churn Prediction

Última atualização: 2026-05-03
Fonte oficial das métricas e custos: `notebooks/02_experimento_controlado.ipynb`

## 1) Modelo recomendado no notebook

- Modelo: **LogisticRegression-balanced**
- Critério de recomendação: maior **Net Benefit** na análise de trade-off
- Threshold ótimo: **0.15**
- Net Benefit máximo: **$706,000**
- Performance no threshold ótimo:
  - Recall: **0.9840** (98.40%)
  - Precision: **0.3802** (38.02%)
  - TP=**368**, FP=**600**, FN=**6**

## 2) Comparação de métricas (threshold padrão 0.5)

| Modelo | AUC-ROC | PR-AUC | Accuracy | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoostClassifier | 0.852816 | 0.675894 | 0.809084 | 0.662539 | 0.572193 | 0.614060 |
| XGBoostClassifier-tuned | 0.852453 | 0.664232 | 0.801987 | 0.654723 | 0.537433 | 0.590308 |
| MLPWrapper-PyTorch | 0.848952 | 0.654735 | 0.758694 | 0.530686 | 0.786096 | 0.633621 |
| LogisticRegression-balanced | 0.848211 | 0.644412 | 0.743790 | 0.511464 | 0.775401 | 0.616366 |
| LogisticRegression-simples | 0.848066 | 0.644134 | 0.803407 | 0.643068 | 0.582888 | 0.611501 |
| LogisticRegression-SMOTE | 0.846741 | 0.642804 | 0.743080 | 0.510417 | 0.786096 | 0.618947 |
| RandomForestClassifier | 0.833785 | 0.627266 | 0.793471 | 0.640678 | 0.505348 | 0.565022 |
| DummyClassifier-most_frequent | 0.500000 | 0.265436 | 0.734564 | 0.000000 | 0.000000 | 0.000000 |

## 3) Comparação de custo-benefício (threshold ótimo por modelo)

Premissas do notebook:

- custo de falso positivo (`cost_fp`) = **$50**
- custo de falso negativo evitado (`cost_fn`) = **$2000**

| Modelo | Threshold ótimo | Net Benefit | Recall | Precision |
| --- | ---: | ---: | ---: | ---: |
| LogisticRegression-balanced | 0.15 | $706,000 | 98.40% | 38.02% |
| LogisticRegression-SMOTE | 0.15 | $705,500 | 98.13% | 39.17% |
| MLPWrapper-PyTorch | 0.10 | $688,750 | 95.99% | 38.03% |
| LogisticRegression-simples | 0.10 | $685,100 | 94.92% | 41.62% |
| XGBoostClassifier | 0.10 | $676,050 | 93.58% | 42.22% |
| XGBoostClassifier-tuned | 0.10 | $668,600 | 92.51% | 42.51% |
| RandomForestClassifier | 0.10 | $663,950 | 92.25% | 39.84% |
| DummyClassifier-most_frequent | 0.10 | $0 | 0.00% | 0.00% |

## 4) Decisão técnica

Mesmo sem liderar AUC/F1 no threshold padrão 0.5, o **LogisticRegression-balanced** foi escolhido porque maximizou o resultado econômico no cenário de negócio modelado no notebook.

## 5) Limitações

- O Net Benefit depende diretamente das premissas de custo (`$50` e `$2000`).
- Alterações de custo real de campanha ou valor de cliente exigem recalibrar threshold e ranking.
- Recomenda-se reavaliação periódica com dados atualizados e custos reais.

## 6) Nota sobre MLflow e versionamento

Os experimentos que embasaram este Model Card foram salvos no MLflow.
Os artefatos de experimento não foram comitados no Git por falta de necessidade para a execução da aplicação e para manter a limpeza do repositório.
