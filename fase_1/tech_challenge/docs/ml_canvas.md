# ML Canvas - Telco Churn (Fase 1)

Ultima atualizacao: 2026-05-03

## 1. Problema de negocio

Uma operadora de telecom perde clientes por cancelamento (churn), impactando receita recorrente e margem.
O desafio e antecipar quem tem maior risco de churn para permitir acao proativa de retencao.

## 2. Objetivo de negocio

Priorizar clientes com maior probabilidade de churn para:

- reduzir cancelamentos evitaveis
- aumentar retencao de receita
- orientar campanhas com melhor custo-beneficio

## 3. Stakeholders

| Stakeholder | Responsabilidade | Resultado esperado |
| --- | --- | --- |
| Marketing/CRM | Campanhas de retencao segmentadas | Melhor conversao das ofertas |
| Customer Success | Contato e negociacao com clientes de risco | Reducao de churn efetivo |
| Financeiro | Medir impacto em receita e ROI | Planejamento de receita |
| Time de Dados/ML | Treino, monitoramento e manutencao | Modelo estavel e auditavel |
| Tecnologia | Operacao da API e observabilidade | Baixa latencia e disponibilidade |

## 4. Solucao de ML escolhida

- Tipo: classificacao binaria (`churn` vs `nao churn`)
- Modelo recomendado: **LogisticRegression-balanced**
- Base da decisao: `notebooks/02_experimento_controlado.ipynb` e [MODEL_CARD.md](MODEL_CARD.md)
- Threshold operacional recomendado: **0.15**
- Justificativa: maior resultado economico na analise de trade-off, com boa explicabilidade para negocio

## 5. KPI de sucesso

### KPI de negocio

- Reduzir churn no publico priorizado por campanha
- Aumentar receita retida por cliente
- Atingir ROI positivo nas acoes de retencao
- Maximizar Net Benefit sob premissas de custo da campanha

### KPI tecnico

- ROC-AUC de referencia (modelo recomendado): **0.8482** (threshold 0.5)
- F1-score de referencia (modelo recomendado): **0.6164** (threshold 0.5)
- Recall no threshold operacional (0.15): **98.40%**
- Precision no threshold operacional (0.15): **38.02%**
- Latencia da API < 200 ms por requisicao

## 5.1 Premissas de custo no experimento

- `cost_fp = $50` (acao de retencao em falso positivo)
- `cost_fn = $2000` (valor perdido por churn nao evitado)
- Resultado do modelo recomendado no notebook:
  - Net Benefit maximo: **$706,000**
  - TP=368, FP=600, FN=6

## 6. Dados

- Fonte: Kaggle - Telco Customer Churn
  Link: <https://www.kaggle.com/datasets/blastchar/telco-customer-churn>
- Volume: 7,043 clientes
- Target: `Churn`
- Dicionario de dados: [DICIONARIO_DADOS.md](DICIONARIO_DADOS.md)

## 7. Requisitos funcionais

1. Gerar score de risco de churn por cliente.
2. Expor previsao por API para uso em operacao.
3. Permitir rastreabilidade de runs e metricas via MLflow.
4. Suportar analise de explicabilidade para equipes de negocio.

## 8. Requisitos nao funcionais

- Reprodutibilidade de treino e avaliacao
- Governanca de modelo (versionamento e metricas)
- Observabilidade (logs, metricas e health check)
- Confiabilidade operacional em ambiente containerizado

## 9. Riscos e limitacoes

- Drift de dados ao longo do tempo (novas ofertas e mudanca de perfil de clientes)
- Performance sensivel a qualidade de dados de entrada
- Necessidade de revisao de threshold conforme custo real de campanha

## 10. Plano de evolucao

1. Monitoramento mensal de performance e drift.
2. Recalibracao de threshold por capacidade operacional da retencao.
3. Retreinamento periodico com dados novos e validacao de impacto financeiro.

## 11. Nota sobre MLflow e versionamento

Os experimentos deste projeto foram registrados no MLflow para rastreabilidade tecnica.
Os artefatos e historicos de execucao do MLflow nao foram comitados no repositorio, por nao serem necessarios ao codigo-fonte e para preservar a limpeza do repo.
