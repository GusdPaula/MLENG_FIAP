# ML Canvas - Telco Churn (Fase 1)

Última atualização: 2026-05-03

## 1. Problema de negócio

Uma operadora de telecom perde clientes por cancelamento (churn), impactando receita recorrente e margem.
O desafio é antecipar quem tem maior risco de churn para permitir ação proativa de retenção.

## 2. Objetivo de negócio

Priorizar clientes com maior probabilidade de churn para:

- reduzir cancelamentos evitáveis
- aumentar retenção de receita
- orientar campanhas com melhor custo-benefício

## 3. Stakeholders

| Stakeholder | Responsabilidade | Resultado esperado |
| --- | --- | --- |
| Marketing/CRM | Campanhas de retenção segmentadas | Melhor conversão das ofertas |
| Customer Success | Contato e negociação com clientes de risco | Redução de churn efetivo |
| Financeiro | Medir impacto em receita e ROI | Planejamento de receita |
| Time de Dados/ML | Treino, monitoramento e manutenção | Modelo estável e auditável |
| Tecnologia | Operação da API e observabilidade | Baixa latência e disponibilidade |

## 4. Solução de ML escolhida

- Tipo: classificação binária (`churn` vs `não churn`)
- Modelo recomendado: **LogisticRegression-balanced**
- Base da decisão: `notebooks/02_experimento_controlado.ipynb` e [MODEL_CARD.md](MODEL_CARD.md)
- Threshold operacional recomendado: **0.15**
- Justificativa: maior resultado econômico na análise de trade-off, com boa explicabilidade para negócio

## 5. KPI de sucesso

### KPI de negócio

- Reduzir churn no público priorizado por campanha
- Aumentar receita retida por cliente
- Atingir ROI positivo nas ações de retenção
- Maximizar Net Benefit sob premissas de custo da campanha

### KPI técnico

- ROC-AUC de referência (modelo recomendado): **0.8482** (threshold 0.5)
- F1-score de referência (modelo recomendado): **0.6164** (threshold 0.5)
- Recall no threshold operacional (0.15): **98.40%**
- Precision no threshold operacional (0.15): **38.02%**
- Latência da API < 200 ms por requisição

## 5.1 Premissas de custo no experimento

- `cost_fp = $50` (ação de retenção em falso positivo)
- `cost_fn = $2000` (valor perdido por churn não evitado)
- Resultado do modelo recomendado no notebook:
  - Net Benefit máximo: **$706,000**
  - TP=368, FP=600, FN=6

## 6. Dados

- Fonte: Kaggle - Telco Customer Churn
  Link: <https://www.kaggle.com/datasets/blastchar/telco-customer-churn>
- Volume: 7,043 clientes
- Target: `Churn`
- Dicionário de dados: [DICIONARIO_DADOS.md](DICIONARIO_DADOS.md)

## 7. Requisitos funcionais

1. Gerar score de risco de churn por cliente.
2. Expor previsão por API para uso em operação.
3. Permitir rastreabilidade de runs e métricas via MLflow.
4. Suportar análise de explicabilidade para equipes de negócio.

## 8. Requisitos não funcionais

- Reprodutibilidade de treino e avaliação
- Governança de modelo (versionamento e métricas)
- Observabilidade (logs, métricas e health check)
- Confiabilidade operacional em ambiente containerizado

## 9. Riscos e limitações

- Drift de dados ao longo do tempo (novas ofertas e mudança de perfil de clientes)
- Performance sensível a qualidade de dados de entrada
- Necessidade de revisão de threshold conforme custo real de campanha

## 10. Plano de evolução

1. Monitoramento mensal de performance e drift.
2. Recalibração de threshold por capacidade operacional da retenção.
3. Retreinamento periódico com dados novos e validação de impacto financeiro.

## 11. Nota sobre MLflow e versionamento

Os experimentos deste projeto foram registrados no MLflow para rastreabilidade técnica.
Os artefatos e históricos de execução do MLflow não foram comitados no repositório, por não serem necessários ao código-fonte e para preservar a limpeza do repo.
