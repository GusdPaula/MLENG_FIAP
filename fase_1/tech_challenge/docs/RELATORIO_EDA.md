# 📊 Etapa 1 & 2: ML Canvas + Exploração de Dados (EDA)
## Previsão de Churn - Telecomunicações

**Data**: Abril 2026
**Status**: ✅ Completo - Etapas 1 e 2 Executadas

---

## 1️⃣ Entendimento de Negócio (ML Canvas)

### Problema de Negócio
Uma operadora de telecomunicações está perdendo clientes em ritmo acelerado, impactando significativamente a receita mensal (MRR - Monthly Recurring Revenue).

### Proposta de Valor
Implementar um **sistema de alerta precoce** que identifique clientes com **alta propensão ao cancelamento**, permitindo que o time de Customer Success atue de forma personalizada e preventiva antes do encerramento do contrato.

### Benefício Direto
- **Redução de 10%** na taxa de churn trimestral
- **Aumento de 15%** no Lifetime Value (LTV) médio
- **ROI de retenção > 400%** (custo de retenção < 25% do revenue)

### Atores de Negócio (Stakeholders)

| Ator | Responsabilidade | Métrica | Impacto |
|------|-------------------|---------|--------|
| **Marketing** | Campanhas de retenção segmentadas | Taxa de conversão em retenção | Direcionar orçamento |
| **Financeiro** | Controle do impacto no MRR | Revenue retido / Customer | Prever receita |
| **Customer Success** | Ações personalizadas de retenção | % de clientes em risco contatados | Aumentar LTV |
| **Engenharia de ML** | Manutenção do modelo e dados | F1-Score, AUC-ROC | Garantir precisão |

### Métricas de Sucesso (KPIs)

#### Técnicas
- **F1-Score**: > 0.70 (importante pelo desbalanceamento)
- **AUC-ROC**: > 0.85 (boa discriminação)
- **Recall**: ≥ 0.80 (capturar máximo de churn)
- **Precision**: ≥ 0.65 (reduzir falsos positivos)
- **Latência da API**: < 200ms por requisição

---

## 2️⃣ Exploração de Dados (EDA) e Qualidade

### Dataset
- **Fonte**: IBM Telco Customer Churn
- **Tamanho**: **7,043 clientes** × **33 variáveis**
- **Período**: Dados históricos de customerbase de telecomunicações
- **Variável Target**: `Churn Label` (Yes/No)

### Inspeção Inicial

#### Tipos de Dados
- **Numéricas**: 10 colunas (tenure_months, monthly_charges, total_charges, etc.)
- **Categóricas**: 23 colunas (contract, internet_service, payment_method, etc.)
- **Valores Faltantes**: Apenas 11 em `total_charges` (clientes novos com tenure=0)

#### Tratamento de Dados
✅ **Coluna 'total_charges'**: Originalmente em string
→ Convertida para float
→ 11 valores NaN identificados (clientes novos)
→ Preenchidos com 0 (clientes ainda não geraram cobranças)

**Resultado**: Dataset limpo e pronto para modelagem

---

## 3️⃣ Padrões Críticos de Churn Identificados

### 📊 1. Distribuição de Churn (Target)

| Status | Quantidade | Percentual | Insight |
|--------|-----------|-----------|---------|
| **Não-Churn (No)** | 5,174 | 73.5% | Maioria retém |
| **Churn (Yes)** | 1,869 | 26.5% | ~1 em 4 clientes sai |

**Desbalanceamento**: Razão 1:2.77 (importante indicador para seleção de métricas)

---

### 🔴 2. Churn por TIPO DE CONTRATO

```
Month-to-month    →  42.71% CHURN  🔴 ALTO
One year          →  11.27% CHURN  🟡 MÉDIO
Two year          →   2.83% CHURN  🟢 BAIXO
```

**Insight**: Clientes com contratos flexíveis (mês-a-mês) têm **15x MAIS churn** que os dois anos!

**Implicação de Negócio**: Foco em retenção deve ser em clientes mês-a-mês.

---

### 🔴 3. Churn por TIPO de SERVIÇO DE INTERNET

```
Fiber optic    →  41.89% CHURN  🔴 ALTO
DSL            →  18.96% CHURN  🟡 MÉDIO
No Internet    →   7.40% CHURN  🟢 BAIXO
```

**Insight**: Clientes com Fibra Óptica têm **2.2x MAIS churn** que DSL!

**Hipótese**: Possível insatisfação com custo, velocidade ou estabilidade técnica de Fibra.

---

### 🔴 4. Churn por MÉTODO DE PAGAMENTO

```
Electronic check           →  45.29% CHURN  🔴 ALTO
Mailed check              →  19.11% CHURN  🟡 MÉDIO
Credit card (automatic)   →  15.24% CHURN  🟢 BAIXO
Bank transfer (automatic) →  16.71% CHURN  🟢 BAIXO
```

**Insight**: "Electronic check" tem **3x MAIS churn** que pagamentos automáticos!

**Padrão**: Automação de pagamento reduz churn significativamente.

---

## 4️⃣ Análise de Variáveis Numéricas

### tenure_months (Meses de Permanência)

```
Média:     32.37 meses
Mediana:   29.00 meses
Min:        0.00 (clientes novos)
Max:       72.00 (6 anos)
Std Dev:   24.56 (alta variabilidade)
```

**Padrão**: Distribuição indica muitos clientes novos com alto churn imediato.

### monthly_charges ($)

```
Média:     $64.76
Mediana:   $70.35
Min:       $18.25
Max:      $118.75
Std Dev:   $30.09
```

**Padrão**: Clientes mais caros tendem a ter maior churn (verificar correlação).

### total_charges ($)

```
Média:     $2,279.73
Mediana:   $1,394.55
Min:       $0.00 (clientes novos)
Max:       $8,684.80
Std Dev:   $2,266.79 (muito variável)
```

---

## 5️⃣ Checklist de Data Readiness

✅ **Volume**: 7,043 registros (suficiente para ML)
✅ **Limpeza**: Valores faltantes tratados
✅ **Tipos de Dados**: Corrigidos (string → float)
✅ **Distribuição**: Padrões claros de churn identificados
✅ **Integridade**: Nenhuma inconsistência lógica
✅ **Balance**: Desbalanceamento documentado (73.5% vs 26.5%)

**Status**: 🟢 **PRONTO PARA MODELAGEM**

---

## 6️⃣ Recomendações para Modelagem

### Tratamento de Desbalanceamento
- Usar **F1-Score** e **AUC-ROC** como métricas principais
- Implementar **weighted loss functions** (pytorch) para dar mais peso ao churn
- Considerar SMOTE ou undersampling se necessário
- Usar validação cruzada **estratificada** (StratifiedKFold)

### Seleção de Features
**Numéricas**:
- tenure_months, monthly_charges, total_charges

**Categóricas Críticas** (por correlação com churn):
- contract (muito importante!)
- internet_service (muito importante!)
- payment_method (importante!)
- senior_citizen, tech_support, online_security, etc.

### Estratégia de Baseline
1. **DummyClassifier** (baseline simples)
2. **Regressão Logística** (baseline linear)
3. **Random Forest / XGBoost** (baseline tree-based)
4. **Rede Neural MLP (PyTorch)** (modelo principal)

---

## 7️⃣ Próximas Etapas (Roadmap)

### 📋 Etapa 2 - Modelagem com Redes Neurais
1. ✅ Treinar baselines (DummyClassifier, LogReg)
2. ✅ Construir MLP (3 camadas, ReLU, Dropout)
3. ✅ Implementar early stopping e learning rate scheduler
4. ✅ Comparar todos os modelos

### 📋 Etapa 3 - API e Engenharia
1. ✅ Refatorar em modelos reutilizáveis
2. ✅ Criar pipeline sklearn
3. ✅ Escrever testes (pytest)
4. ✅ Construir API FastAPI

### 📋 Etapa 4 - Documentação
1. ✅ Model Card completo
2. ✅ Análise de Fairness
3. ✅ README + instruções
4. ✅ Vídeo STAR (5 min)

---

## 📂 Arquivos Gerados

- ✅ `ML_CANVAS.md` - ML Canvas detalhado
- ✅ `RELATORIO_EDA.md` - Este arquivo
- ✅ `01_eda_analysis.py` - Script de análise
- ✅ `data/processed/telco_churn_processed.csv` - Dataset limpo
- ✅ `notebooks/01_eda_and_ml_canvas.ipynb` - Jupyter notebook

---

## 🎯 Conclusão

Os dados revelam **oportunidades claras de segmentação** para retenção:
- Target: Clientes com **contratos mês-a-mês** e **Fibra Óptica**
- Estratégia: Oferecer incentivos para upgrade a contrato anual
- Automatizar: Pagamento automático (reduz churn em 65%)


---

**Versão**: 1.0
**Data Conclusão**: Abril 2026
