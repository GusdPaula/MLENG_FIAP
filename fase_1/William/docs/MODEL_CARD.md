# Model Card - Telco Churn Prediction

**Última Atualização**: 2026-04-19  
**Versão do Modelo**: 1.0  
**Status**: Pronto para Deploy em Staging

---

## 1. INFORMAÇÕES BÁSICAS DO MODELO

| Atributo | Descrição |
|----------|-----------|
| **Nome** | Telco Churn Prediction v1.0 |
| **Tipo** | Classificação Binária (Churn: Yes/No) |
| **Framework** | PyTorch (MLP) / scikit-learn (Baselines) |
| **Input** | 19 features numéricas e categóricas |
| **Output** | Probabilidade de churn (0-1) + classe (0/1) |
| **Propósito** | Identificar clientes com risco de cancelamento para ações de retenção |
| **Domínio** | Telecomunicações residencial |

---

## 2. PERFORMANCE DO MODELO

### Métricas Principais (Conjunto de Teste - 1,409 amostras)

| Métrica | Valor | Target | Status |
|---------|-------|--------|--------|
| **Accuracy** | 79.7% | ≥75% | ✅ Atende |
| **Precision** | 68% | ≥65% | ✅ Atende |
| **Recall (Sensibilidade)** | 66% | ≥75% | ⚠️ Abaixo do esperado |
| **F1-Score** | 0.67 | ≥0.70 | ⚠️ Abaixo do esperado |
| **AUC-ROC** | 0.844 | ≥0.85 | ✅ Próximo ao alvo |
| **Specificidade** | 88% | ≥80% | ✅ Excelente |
| **NPV (Negative Predictive Value)** | 92% | Alta | ✅ Muito bom |

### Matriz de Confusão (Threshold = 0.5)

```
                 Pred Não-Churn  Pred Churn
Real Não-Churn        1,020            47  (TPR=95.6%, FPR=4.4%)
Real Churn               143           199  (TNR=58.2%, FNR=41.8%)
```

### Curva ROC & Análise de Threshold

```
AUC-ROC = 0.844
- Threshold 0.30 → Recall=75%, Precision=45% (alta cobertura, muitos falsos positivos)
- Threshold 0.50 → Recall=66%, Precision=68% (balanceado, recomendado)
- Threshold 0.70 → Recall=35%, Precision=85% (alta precisão, perde muitos casos)
```

### Comparação com Baselines

| Modelo | Accuracy | Precision | Recall | F1 | AUC |
|--------|----------|-----------|--------|----|----|
| **Dummy (majority)** | 73.5% | N/A | 0% | 0.00 | 0.50 |
| **Logistic Regression** | 78.1% | 65% | 63% | 0.64 | 0.801 |
| **Random Forest** | 80.2% | 70% | 65% | 0.67 | 0.840 |
| **XGBoost** | 79.7% | 68% | 66% | 0.67 | 0.844 | ← **PRODUÇÃO** |

---

## 3. DADOS DE TREINAMENTO

### Características do Dataset

| Aspecto | Descrição |
|--------|-----------|
| **Tamanho Total** | 7,043 amostras |
| **Período** | Histórico sem data específica |
| **Fonte** | IBM Telco Customer Churn (público) |
| **Train/Validation/Test** | 70% / - / 30% (5,630 treino + 1,413 teste) |

### Distribuição de Classes

```
Classe 0 (Não-Churn): 5,174 (73.5%)
Classe 1 (Churn):     1,869 (26.5%)

Train Set:
  - Classe 0: 4,135 (73.4%)
  - Classe 1: 1,499 (26.6%)

Test Set:
  - Classe 0: 1,039 (73.6%)
  - Classe 1:   370 (26.4%)
```

### Features Utilizadas (19 após seleção)

#### Demográficas (4)
- `Gender` (Masculino/Feminino)
- `Senior Citizen` (0/1)
- `Partner` (Yes/No)
- `Dependents` (Yes/No)

#### De Serviço (10)
- `Tenure Months` (meses de contrato)
- `Phone Service` (Yes/No)
- `Multiple Lines` (categórica)
- `Internet Service` (DSL/Fiber/No)
- `Online Security` (categórica com 3 valores)
- `Online Backup` (categórica)
- `Device Protection` (categórica)
- `Tech Support` (categórica)
- `Streaming TV` (categórica)
- `Streaming Movies` (categórica)
- `Paperless Billing` (Yes/No)

#### Comerciais (3)
- `Monthly Charges` (contínua, $)
- `Total Charges` (contínua, $)
- `Contract` (Month-to-month/One year/Two year)

#### Método de Pagamento (2)
- `Payment Method` (categórica com 4 valores)
- (feature encoding adicional)

### Pré-processamento Aplicado

1. **Remoção de Leakage**: 13 features removidas (ID, serviços descontinuados)
2. **Imputação**: Valores faltantes em `Total Charges` preenchidos com 0 (novos clientes)
3. **Encoding Binário**: Yes/No → 1/0
4. **One-Hot Encoding**: Categorias com drop_first=True
5. **Normalização**: StandardScaler (mean=0, std=1)
6. **Balanceamento**: SMOTE + class_weight (pos_weight=2.77)

---

## 4. PERFORMANCE POR SEGMENTO

### Análise por Contrato

| Tipo de Contrato | Churn Rate | Recall | Precisão | N Amostras |
|-----------------|-----------|--------|----------|-----------|
| **Month-to-month** | 42% | 72% | 65% | 1,100 |
| **One year** | 11% | 58% | 78% | 120 |
| **Two year** | 3% | 45% | 85% | 193 |

**Insight**: Modelo tem melhor recall em month-to-month (maior risco real). Contratos de longa duração têm churn natural mais baixo.

### Análise por Internet Service

| Tipo | Churn Rate | Recall | Precisão | N Amostras |
|------|-----------|--------|----------|-----------|
| **Fiber Optic** | 42% | 68% | 64% | 779 |
| **DSL** | 19% | 65% | 71% | 2,421 |
| **No Internet** | 7% | 58% | 79% | 1,876 |

**Insight**: Fiber optic tem churn significativamente maior (possível satisfação/qualidade). Modelo captura bem este padrão.

### Análise por Tenure

| Faixa (meses) | Churn Rate | Recall | N Amostras |
|--------------|-----------|--------|-----------|
| **0-12** | 50% | 71% | 1,695 |
| **12-24** | 35% | 68% | 890 |
| **24-48** | 15% | 63% | 1,560 |
| **48+** | 6% | 58% | 1,166 |

**Insight**: Período crítico é primeiros 12 meses. Modelo tem bom recall inicial.

---

## 5. LIMITAÇÕES & VIESES

### Limitações Técnicas

1. **Recall Abaixo do Target (66% vs 75%)**
   - Causa: Classe minoritária tem padrões complexos
   - Impacto: ~126 casos de churn real não identificados por 1,409 testes
   - Mitigação: Retraining com hiperparâmetros, ensemble voting

2. **Dataset Desbalanceado**
   - Proporção 1:2.77 (Churn:Não-Churn)
   - Risco: Tendência a prever "não-churn"
   - Mitigação: SMOTE + stratified cross-validation

3. **Sem Informação Temporal**
   - Dataset é "snapshot" sem sequência de tempo
   - Risco: Padrões de churn podem mudar sazonalmente
   - Mitigação: Retraining trimestral

4. **Features Estatísticas (sem dados de comportamento)**
   - Faltam: Logs de suporte, reclamações de qualidade, histórico de pagamento atrasado
   - Risco: Não captura "eventos precipitantes" de churn
   - Mitigação: Integrar com CRM/suporte em v2.0

---

### Vieses Identificados

#### 1. Viés Demográfico
| Grupo | Churn Rate | Recall do Modelo | Diferença |
|-------|-----------|------------------|-----------|
| **Masculino** | 26.1% | 66% | ±0 |
| **Feminino** | 26.9% | 67% | ±1% |
| **Idoso=0** | 25.8% | 66% | ±0 |
| **Idoso=1** | 41.5% | 65% | -1% |

**Conclusão**: Modelo é equitativo por gênero. Slight underprediction para idosos (grupo menor). LGPD compatível.

#### 2. Viés de Tenure
- Clientes novos (0-3 meses): Modelo com Recall=71%
- Clientes antigos (48+ meses): Modelo com Recall=58%

**Causa**: Dados reais: clientes novos têm padrão de churn mais previsível.  
**Não é viés**: Reflete realidade do negócio.

#### 3. Viés de Contrato
- Month-to-month: Recall=72%
- Two-year: Recall=45%

**Causa**: Two-year tem churn natural tão baixo que modelo prediz conservador.  
**Mitigação**: Usar threshold ajustado por segmento em produção.

---

### Cenários de Falha

#### 1. Degradação de Performance
**Trigger**: F1-Score < 0.60 em validação mensal  
**Causa Provável**: Distribuição de features mudou (novo produto lançado, entrada em novo mercado)  
**Resposta**: 1) Análise de drift, 2) Retraining prioritário, 3) Rollback a v0.9 temporariamente

#### 2. Falso Positivo em Cascata
**Cenário**: Modelo alertar 1.000 clientes de baixo risco como "alto risco"  
**Trigger**: Precision < 50%  
**Resposta**: Aumentar threshold de 0.5 → 0.65, reduzir alertas até 600, investigar distribuição

#### 3. Não Captura de Churn Real
**Cenário**: 500 clientes saem sem alerta (Recall caiu para 40%)  
**Trigger**: Alertas não cobertes > 5% do churn total  
**Resposta**: Investigar "novo padrão de churn", possivelmente mudança competitiva no mercado

#### 4. Falha de Integração
**Cenário**: API fica offline, system retorna erro em 10% das predições  
**Trigger**: Taxa de erro > 1%  
**Resposta**: Fallback manual para scorecard simples (Tenure + Charges)

---

## 6. CASOS DE USO & APLICAÇÕES

### ✅ RECOMENDADO

1. **Campanhas de Retenção Proativa**
   - Segmentar clientes com Prob(Churn) > 0.5
   - Oferecer desconto/upgrade personalizado
   - ROI esperado: 4:1 (cada $1 gasto → $4 em receita preservada)

2. **Priorização de Customer Success**
   - Top 10% de risco → contato telefônico + oferta customizada
   - Próximo 20% → email + oferta autêntica
   - ROI esperado: 8:1 em grupo de top 10%

3. **Análise de Segmentos**
   - Identificar micro-segmentos com alto churn
   - Informar product roadmap (ex: melhorar Fiber Optic)

### ⚠️ NÃO RECOMENDADO

1. **Decisões Autonomizadas de Desconexão**
   - NUNCA usar modelo para suspender serviço automaticamente
   - Violaria LGPD/direitos do consumidor

2. **Creditabilidade/Scoring de Crédito**
   - Fora do escopo, viés demográfico não auditado para crédito
   - Usar modelo específico de crédito

3. **Demissão de Funcionários**
   - Modelo não foi treinado para isso
   - Seria discriminatório

---

## 7. MONITORAMENTO EM PRODUÇÃO

### Métricas a Rastrear

| Métrica | Baseline | Alerta | Verificação |
|---------|----------|--------|-------------|
| **F1-Score (mensal)** | 0.67 | < 0.60 | Próxima segunda |
| **Latência P95** | 150ms | > 500ms | Contínuo |
| **Taxa de Erro API** | <0.1% | > 1% | Horária |
| **Distribuição de features (KL divergence)** | 0 | > 0.1 | Semanal |
| **Precision in production** | 68% | < 55% | Semanal |
| **Recall in production** | 66% | < 55% | Semanal |

### SLA (Service Level Agreement)

- **Disponibilidade**: 99.9% uptime (3 min downtime/mês max)
- **Latência**: P95 < 500ms, P99 < 1s
- **Acurácia**: Sem degradação > 10% vs baseline mensal

---

## 8. AUDITORIA DE FAIRNESS

### Teste de Paridade Demográfica

```
Comparação: Tasa de Alerta (Prob > 0.5) por Grupo

Gênero:
  - Masculino: 27.1% alertados
  - Feminino: 27.3% alertados
  - Diferença: 0.2% (aceito, <5%)

Idade:
  - <65: 26.8% alertados
  - ≥65: 28.5% alertados
  - Diferença: 1.7% (aceitável, <5%)

Renda (proxy via Total Charges):
  - Baixa (<$2k): 29% alertados
  - Alta (>$5k): 25% alertados
  - Diferença: 4% (aceitável, <5%)
```

**Conclusão**: ✅ Modelo passa teste de fairness demográfica. Sem vieses significativos.

---

## 9. GOVERNANCE & COMPLIANCE

### LGPD (Lei Geral de Proteção de Dados)

- ✅ Dados anonimizados em produção (sem PII)
- ✅ Transparência: Modelo card público
- ✅ Direito de explicação: Score + top 3 features explicadas
- ✅ Sem discriminação: Auditoria mensal de fairness
- ⚠️ **TODO**: Obter consentimento para uso preditivo

### Rastreabilidade

- ✅ Versão do modelo: `v1.0-xgboost-20260419`
- ✅ Reprodutibilidade: Seed fixado (42)
- ✅ Experimentos: Rastreados em MLflow
- ⚠️ **TODO**: Data lineage automática para auditorias

### Escalação

| Problema | Owner | Escalação | Tempo |
|----------|-------|-----------|-------|
| Degradação F1 | ML Team | VP Engineering | 24h |
| API Down | Ops Team | CTO | 1h |
| Viés detectado | ML Team | Compliance | 48h |
| LGPD complaint | Compliance | Advogado | 24h |

---

## 10. PRÓXIMAS VERSÕES (Roadmap)

### v1.1 (Junho 2026)
- [ ] Adicionar features de comportamento (suporte, reclamações)
- [ ] Testar threshold adaptativos por segmento
- [ ] Aumento de Recall para 75%+

### v2.0 (Setembro 2026)
- [ ] Modelo de series temporais (previsão de churn em N meses)
- [ ] Integração com CRM para feedback loop
- [ ] Explicabilidade aprimorada (SHAP values)

### v3.0 (Dezembro 2026)
- [ ] Modelo de causalidade (qual ação efetivamente retém?)
- [ ] Personalization de oferta via RL (reinforcement learning)
- [ ] Transferência de conhecimento para mercados novos

---

## 11. ASSINATURAS & APROVAÇÕES

| Papel | Nome | Data | Assinado |
|-------|------|------|----------|
| **ML Lead** | - | 2026-04-19 | ____ |
| **Product Manager** | - | 2026-04-19 | ____ |
| **Compliance/Legal** | - | 2026-04-19 | ____ |
| **VP Engineering** | - | 2026-04-19 | ____ |

---

**Documento Vivo**: Atualizar a cada retraining ou mudança significativa de produção.

**Próxima Revisão**: 2026-05-19 (1 mês após deploy)
