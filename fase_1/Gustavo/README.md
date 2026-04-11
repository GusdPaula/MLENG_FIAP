# 📊 ML Churn: Predição de Churn de Clientes Telecom

## 📋 Índice
1. [Descrição do Projeto](#descrição-do-projeto)
2. [ML Canvas](#ml-canvas)
3. [Métricas de Avaliação](#métricas-de-avaliação)
4. [Modelos Implementados](#modelos-implementados)
5. [Melhor Modelo](#melhor-modelo)
6. [Abordagens Utilizadas](#abordagens-utilizadas)
7. [Resultados Comparativos](#resultados-comparativos)

---

## 🎯 Descrição do Projeto

Este projeto implementa diferentes algoritmos de Machine Learning para prever churn (cancelamento) de clientes em uma empresa de telecomunicações. O dataset utilizado é o **Telco Customer Churn** do IBM/Kaggle, contendo informações demográficas, de serviços e de comportamento de mais de 7.000 clientes.

**Objetivo:** Identificar clientes com alto risco de cancelamento para implementar estratégias de retenção proativas.

---

## 🗺️ ML Canvas

### Problema de Negócio

A **retenção de clientes** é crítica para o sucesso de empresas de telecomunicações. Cada cliente perdido representa:
- Perda de receita recorrente
- Custo de aquisição de novo cliente (tyicamente 5-10x superior ao custo de retenção)
- Possível dano à reputação da marca

### Regras de Negócio

1. **Classe Desbalanceada:** Apenas ~26% dos clientes fizeram churn (desequilíbrio 3:1)
2. **Custo Assimétrico:** O custo de um falso negativo (não identificar churn) é muito maior que o custo de um falso positivo (oferecer retenção desnecessária)
3. **Ação Reativa:** Quando um cliente em risco é identificado, é possível oferecer incentivos/promoções antes do cancelamento
4. **Limite de Recursos:** A empresa tem recursos limitados para campanhas de retenção, precisando focar nos clientes com mais alto risco

### Trade-off: Recall vs Precisão

#### Por que Recall é Mais Importante para Churn?

| Aspecto | Precisão | Recall |
|--------|----------|--------|
| **Definição** | De todos os clientes que previmos como churn, quantos realmente fizeram churn? | De todos os clientes que realmente fizeram churn, quantos conseguimos identificar? |
| **Impacto Falho** | **Falso Positivo:** Gastos com retenção desnecessária | **Falso Negativo:** Cliente perdido sem tentativa de retenção |
| **Custo Financeiro** | Baixo (custo marginal de uma oferta extra) | Alto (perda total da receita do cliente) |
| **Para Churn** | Menos crítico | **CRÍTICO** ✓ |

#### Decisão: Por Que Recall é Melhor para Churn Prediction

```
Cenário com N=1000 clientes, 260 com churn real (26% taxa de churn)

Modelo com Alta Precisão, Baixo Recall (Precisão: 90%, Recall: 50%)
├── Identifica: ~117 clientes em risco
├── Acertos: ~105 (90% de 117)
└── PERDIDOS: ~155 clientes (50% não identificados)
    └── Custo: 155 × $valor_cliente = GRANDE PERDA

Modelo com Recall mais Alto, Precisão Moderada (Precisão: 65%, Recall: 75%)
├── Identifica: ~506 clientes em risco
├── Acertos: ~195 (cerca de 75% dos 260 reais)
└── PERDIDOS: ~65 clientes (muito menos)
    └── Custo: 65 × $valor_cliente = MENOR PERDA
├── Falso Positivos: ~311 (custo de retenção defensiva)
└── Custo: 311 × $custo_retenção = GERENCIÁVEL
```

**Conclusão:** O modelo ideal para churn prediction deve maximizar **RECALL** mesmo que reduza precisão, pois:
- Perder clientes é extremamente custoso
- Oferecer retenção a clientes "suspeitos" é comparativamente barato
- A empresa pode depois filtrar os de maior risco (de forma manual ou com um segundo modelo)

---

## 📊 Métricas de Avaliação

### 1. **Acurácia (Accuracy)**
- **Definição:** Proporção de predições corretas em relação ao total
- **Fórmula:** $(TP + TN) / (TP + TN + FP + FN)$
- **Limitação:** Enganosa em datasets desbalanceados (nosso caso!)
- **Interpretação:** Um modelo que sempre prediz "não churn" teria 74% de acurácia
- **Uso:** Métrica de referência, mas não é decisiva

### 2. **Precisão (Precision)**
- **Definição:** De todos os clientes que previmos como "churn", quantos realmente são churn?
- **Fórmula:** $TP / (TP + FP)$
- **Intuição:** Evita "alarmes falsos"
- **Para Churn:** Menos crítica (falsos positivos são baratos)
- **Intervalo:** 0 a 1 (1 = perfeito)

### 3. **Recall (Sensibilidade/Taxa de Verdadeiro Positivo)**
- **Definição:** De todos os clientes que REALMENTE fazem churn, quantos conseguimos identificar?
- **Fórmula:** $TP / (TP + FN)$
- **Intuição:** Não deixar nenhum cliente importante escapar
- **Para Churn:** 🌟 **MÁS IMPORTANTE** - Queremos identificar 75%+ dos riscos reais
- **Intervalo:** 0 a 1 (1 = identifica todos)

### 4. **F1-Score**
- **Definição:** Média harmônica entre precisão e recall
- **Fórmula:** $2 × (Precisão × Recall) / (Precisão + Recall)$
- **Uso:** Balanceia precisão e recall quando ambas importam
- **Para Churn:** Útil para comparação geral, mas recall é prioritário

### 5. **AUC-ROC (Area Under the Receiver Operating Characteristic Curve)**
- **Definição:** Área sob a curva que mostra trade-off entre taxa de verdadeiro positivo (recall) e taxa de falso positivo
- **Interpretação:**
  - AUC = 0.5: Modelo aleatório
  - AUC = 0.8+: Modelo excelente ✓
  - AUC = 1.0: Modelo perfeito
- **Vantagem:** Insensível ao desbalanceamento de classes
- **Uso:** Melhor métrica para datasets desbalanceados
- **Para Churn:** Excelente para comparar modelos de forma geral

### 6. **AUC-PR (Area Under Precision-Recall Curve)**
- **Definição:** Área sob a curva que mostra trade-off entre precisão e recall
- **Interpretação:**
  - Mais informativa que ROC-AUC para datasets com classe minoritária
  - Valor base = proporção de positivos (26% em nosso caso)
  - AUC-PR > 0.5 é muito bom
- **Vantagem:** Foca especificamente na classe de interesse (churn)
- **Uso:** Ideal para problemas de desbalanceamento severo

### Matriz de Confusão
```
                    Predito NÃO Churn    Predito Churn
Real NÃO Churn      TN (Correto)         FP (Falso Alarme)
Real Churn          FN (Perdido!)        TP (Acertado!)

Recall = TP / (TP + FN)      ← Queremos alto!
Precisão = TP / (TP + FP)     ← Secundário para churn
```

---

## 🤖 Modelos Implementados

### 1. **Dummy Classifier (Baseline)**
- **Algoritmo:** Classificador ingênuo que sempre prediz a classe mais frequente
- **Propósito:** Estabelecer um piso mínimo de performance
- **Performance:** AUC-ROC = 0.50 (essencialmente aleatório)
- **Lição:** Qualquer modelo real deve superar isso

### 2. **Logistic Regression (Regressão Logística)**
- **Algoritmo:** Modelo linear probabilístico
- **Características:**
  - Simples, interpretável e rápido
  - Produz estimativas de probabilidade bem calibradas
  - Funciona bem com dados padronizados
  - Registra coeficientes para cada feature (odds ratios)
- **Parâmetros:** C=1.0, solver='lbfgs', max_iter=1000
- **Vantagens:** 
  - Fácil de explicar ao negócio ("cliente com X características tem Y% de risco")
  - Computacionalmente eficiente
- **Performance:** AUC-ROC = 0.8405 ✓

### 3. **Logistic Regression com Class Weight**
- **Modificação:** Penaliza erros na classe minoritária (churn) mais pesadamente
- **Objetivo:** Aumentar recall sem usar técnicas de oversampling
- **Como Funciona:** Atribui peso $w_i = n / (n_{classes} × n_i)$ para cada classe
  - Classe "Não Churn": peso 1.0
  - Classe "Churn": peso ~3.7 (mais importante)
- **Performance:** AUC-ROC = 0.8398, **Recall = 0.786** (muito bom!)
- **Trade-off:** Acurácia cai para 0.743 (esperado), mas recall sobe

### 4. **Logistic Regression com SMOTE**
- **Modificação:** Aplica SMOTE antes do treino (ver seção abordagens)
- **Objetivo:** Balancear dataset sintetizando amostras da classe minoritária
- **Performance:** AUC-ROC = 0.8404, **Recall = 0.794** (muito bom!)
- **Trade-off:** Similar ao class weight, mas usa abordagem diferente

### 5. **Logistic Regression NN (Rede Neural Linear)**
- **Arquitetura:** Rede neural com camada de entrada e uma camada densa (1 neurônio)
- **Ativação:** Sigmoid (para probabilidade)
- **Objetivo:** Benchmark de NN simples vs Logistic Regression tradicional
- **Performance:** AUC-ROC = 0.8115, Recall = 0.618
- **Observação:** Performance similar à LogReg clássica (esperado, mesmo modelo)

### 6. **Logistic Regression NN com SMOTE**
- **Modificação:** LR NN + SMOTE
- **Performance:** AUC-ROC = 0.8119, Recall = 0.618
- **Observação:** Sem melhoria significativa vs sem SMOTE

### 7. **Neural Network com SMOTE (Rede Neural Profunda)**
- **Arquitetura:** 2-layer: entrada → 64 neurônios (ReLU) → 32 neurônios (ReLU) → 1 neurônio (Sigmoid)
- **Otimizador:** Adam
- **Learning Rate:** 0.001
- **Early Stopping:** Paciência de 10 épocas
- **Performance:** AUC-ROC = 0.7441 (pior que esperado)
- **Problema:** Possível overfitting ou underfitting, parou cedo em ~17 épocas

### 8. **Random Forest Classifier**
- **Algoritmo:** Ensemble de múltiplas árvores de decisão
- **Características:**
  - Não linear, captura interações entre features
  - Resiste a outliers
  - Paralelizável
  - Fornece importância de features
- **Parâmetros:** n_estimators=100, bootstrap=True
- **Performance:** AUC-ROC = 0.8211, Recall = 0.495

### 9. **Random Forest Classifier com Random Search CV**
- **Modificação:** Hyperparameter tuning com RandomizedSearchCV
- **Parâmetros Otimizados:**
  - n_estimators: 300
  - max_depth: Otimizado
  - min_samples_split, min_samples_leaf: Otimizados
- **CV:** 3-fold cross validation
- **Performance:** AUC-ROC = 0.8354, Recall = 0.537

### 10. **XGBoost Classifier**
- **Algoritmo:** Gradient Boosting com otimização de segunda ordem
- **Características:**
  - Estado da arte para tabular classification
  - Boosting: sequencial, cada árvore corrige erros da anterior
  - Regularização L1/L2 integrada
  - Muito rápido e escalável
- **Parâmetros:** n_estimators=100, padrões do XGBoost
- **Performance:** AUC-ROC = 0.8160, Recall = 0.529

### 11. **XGBoost Classifier com Random Search CV** 🏆
- **Modificação:** Hyperparameter tuning com RandomizedSearchCV
- **Parâmetros Otimizados:**
  - n_estimators: 300
  - max_depth: 5
  - learning_rate: 0.01
  - subsample: 0.8
  - colsample_bytree: Otimizado
- **CV:** 3-fold cross validation
- **Performance:** 
  - **AUC-ROC = 0.8453** ⭐ (MELHOR)
  - Accuracy = 0.8088
  - Precision = 0.6733
  - Recall = 0.5455
  - F1-Score = 0.6027
  - AUC-PR = 0.6759

---

## 🏆 Melhor Modelo

### **XGBoost Classifier com Random Search CV**

#### Performance Overall

| Métrica | Valor | Avaliação |
|---------|-------|-----------|
| **AUC-ROC** | 0.8453 | ⭐⭐⭐⭐⭐ Excelente |
| **AUC-PR** | 0.6759 | ⭐⭐⭐⭐ Muito Bom |
| **Recall** | 0.5455 | ⭐⭐⭐ Moderado (melhorável) |
| **Precisão** | 0.6733 | ⭐⭐⭐⭐ Boa |
| **F1-Score** | 0.6027 | ⭐⭐⭐⭐ Bom |
| **Acurácia** | 0.8088 | ⭐⭐⭐⭐ Boa |

#### Por Que é o Melhor?

1. **AUC-ROC Máximo (0.8453):** Melhor capacidade de discriminação entre classes
2. **Estável entre Precisão e Recall:** Não sacrifica demais um pelo outro
3. **Hyperparameter Tuning:** Beneficia-se de otimização extensiva
4. **Robusto:** Menos sensível a dados ou noise específicos
5. **Escalável:** Pode ser treinado com datasets muito maiores

#### Recomendação de Uso

**Cenários onde usar:**
- ✅ Prioridade: Balancear custo de falsos positivos vs falsos negativos
- ✅ Quando: Campanha de retenção com orçamento moderado
- ✅ Se: Empresa quer evitar ao máximo deixar escapar clientes

**Cenários para alternivas:**
- **Se recall deve ser máximo (75%+):** Use Logistic Regression com SMOTE/Class Weight (recall ~79%)
  - Trade-off: Mais falsos positivos, mas identifica mais riscos reais
- **Se interpretabilidade é crítica:** Use Logistic Regression simples (AUC-ROC 0.8405)
  - Trade-off: Ligeiramente pior, mas 100% explicável

#### Expectativas em Produção

Com 1000 clientes em um período:
- ~260 clientes realmente farão churn
- Modelo prevê ~506 como risco potencial (com 55% recall)
- Acertará ~195 deles (não perderá completamente)
- Gerará ~311 falsos alarmes (custo de retenção defensiva)

**ROI Esperado:**
```
Clientes salvos: ~190
Receita recuperada: ~190 × $(lifetime_value) 
Custo: ~311 × $(custo_retenção_unitário)
```

---

## 🛠️ Abordagens Utilizadas

### 1. **MLflow - Rastreamento de Experimentos**

MLflow é uma plataforma de código aberto para gerenciar ciclo de vida de Machine Learning.

#### Componentes Utilizados:

```
MLflow
├── Experiment Tracking
│   ├── Nome: "Telco-Churn-Prediction"
│   ├── Todos 11 modelos logados
│   └── Cada run registra:
│       ├── Parâmetros (n_estimators, learning_rate, etc)
│       ├── Métricas (accuracy, precision, recall, F1, AUC-ROC, AUC-PR)
│       ├── Artifacts (modelos salvos em .pkl ou similar)
│       ├── Datasets (train/val/test splits)
│       └── Tags (versão, status, notes)
│
├── Model Registry
│   ├── Staging: Modelos em validação
│   ├── Production: Melhor modelo em uso
│   └── Archived: Versões antigas
│
└── Tracking URI
    └── File: "c:/Users/vc/Documents/MLENG_FIAP/mlruns"
```

#### Benefícios:

- **Reprodutibilidade:** Cada experimento é completamente rastreado
- **Comparação:** Versão lado-a-lado de todos os modelos
- **Histórico:** Voltar a versões anteriores se necessário
- **Colaboração:** Múltiplos membros da equipe veem mesmos resultados
- **Auditoria:** Compliance e rastreamento para regulamentações

#### Estrutura de Dados

```
mlruns/
├── 0/meta.yaml                    # Exp 0 metadata
└── 582674075718407240/            # Exp ID
    ├── meta.yaml                  # Experiment metadata
    ├── artifacts/
    │   ├── modelo_dummy.pkl
    │   ├── modelo_logistic.pkl
    │   ├── modelo_xgboost.pkl
    │   └── ... (um por modelo)
    ├── params/                    # Hiperparâmetros
    └── metrics/                   # Métricas de performance
```

### 2. **SMOTE - Synthetic Minority Over-sampling Technique**

SMOTE é uma técnica para lidar com desbalanceamento de classes.

#### Problema Original:
```
Dataset Telco: 7043 registros
├── Não Churn: 5174 (73.5%)
└── Churn: 1869 (26.5%) ← Minoritária

Razão: ~2.8:1 (desbalanceado!)
```

#### Como SMOTE Funciona:

1. **Identifica amostras minoritárias:** Localiza os ~1869 clientes com churn
2. **Encontra vizinhos:** Para cada amostra, encontra seus K vizinhos mais próximos (K=5, default)
3. **Sintetiza novas amostras:** Cria amostras interpoladas entre vizinhos
   ```
   novo_ponto = ponto_original + random(0,1) × (vizinho - ponto_original)
   ```
4. **Balanceia classes:** Aumenta classe minoritária até 50/50 ou proporção desejada

#### Exemplo Visual:
```
Antes do SMOTE:        Depois do SMOTE:
✗ ✗ ✗ ✗ ✗ ✓ ✓ ✗ ✗ ✗ ✗ ✗ ✗ ✗ ✗ ✓ ✓ ✓ ✓ ✓
(1 positivo em 10)     (5 positivos em 10)

Classes equilibradas → Modelos aprendem melhor
```

#### Vantagens:
- ✅ Reduz bias para classe majoritária
- ✅ Aumenta recall naturalmente
- ✅ Cria dados sintéticos realistas (não apenas cópias)
- ✅ Aplicado apenas no TREINO, não afeta teste (sem data leakage)

#### Desvantagens:
- ❌ Aumenta tempo de treinamento
- ❌ Pode causar overfitting se não regularizar
- ❌ Cria dados artificiais (nem sempre ideal)

#### Aplicação no Projeto:
```python
from imblearn.over_sampling import SMOTE

X_train_resampled, y_train_resampled = SMOTE(random_state=42).fit_resample(
    X_train, y_train
)

# Classes agora aproximadamente iguais
# Treina modelo com dados balanceados
modelo.fit(X_train_resampled, y_train_resampled)

# Testa com dados ORIGINAIS desbalanceados (correto!)
predictions = modelo.predict(X_test)
```

**Resultado:** Logistic Regression com SMOTE alcançou Recall de **0.794** (vs 0.551 sem SMOTE)

### 3. **Class Weight (Ponderação de Classes)**

Alternativa ao SMOTE, implementada diretamente no algoritmo.

#### Conceito:
Penaliza erros na classe minoritária mais pesadamente durante o treinamento.

#### Matemática:
```
Para classe "Não Churn": peso = n_total / (n_classes × n_não_churn)
                               = 7043 / (2 × 5174) ≈ 0.68

Para classe "Churn": peso = n_total / (n_classes × n_churn)
                           = 7043 / (2 × 1869) ≈ 1.88

A classe minoritária é penalizada ~2.8x mais
```

#### Implementação:
```python
model = LogisticRegression(class_weight='balanced', solver='lbfgs')
# Scikit-learn calcula pesos automaticamente
```

#### Comparação SMOTE vs Class Weight:

| Aspecto | SMOTE | Class Weight |
|---------|-------|--------------|
| **Mecanismo** | Cria dados sintéticos | Ajusta função de erro |
| **Tempo Treino** | Mais lento | Igual velocidade |
| **Dados Artificiais** | Sim | Não |
| **Data Leakage Risk** | Baixo (aplicado só treino) | Zero |
| **Recall Obtido** | 0.794 | 0.786 |
| **Acurácia** | 0.749 | 0.743 |
| **Interpretabilidade** | Igual | Melhor |

**Conclusão:** Ambas funcionaram bem. Class Weight é mais simples e determinístico.

### 4. **Feature Coefficients (Coeficientes de Features) em Logistic Regression**

Logistic Regression fornece coeficientes que podem ser interpretados como "odds ratios".

#### O Que São Coeficientes?

Modelo Logistic Regression:
$$P(churn) = \frac{1}{1 + e^{-(β_0 + β_1 × feature_1 + β_2 × feature_2 + ...)}}$$

Onde $β_i$ é o coeficiente da feature $i$.

#### Interpretação:

**Coeficiente Positivo:** Aumenta risco de churn
- Exemplo: $β_{contract\_month} = +0.45$
- Interpretação: Clientes com contrato mensal têm risco maior

**Coeficiente Negativo:** Diminui risco de churn
- Exemplo: $β_{tenure\_months} = -0.05$
- Interpretação: Cada mês adicional de tenure reduz risco

#### Conversão para Odds Ratio:
$$Odds\_Ratio = e^{β}$$

Exemplo com $β = 0.45$:
```
Odds Ratio = e^0.45 ≈ 1.57

Interpretação: Manter todas outras features fixas, ter contrato 
mensal vs anual aumenta odds de churn em 57%
```

#### Features Mais Impactantes (Esperadas):

| Feature | Coeficiente Type | Interpretação |
|---------|-----------------|---------|
| Contract: Month-to-month | ✓ Positivo Alto | Muito risco |
| Internet: Fiber Optic | ✓ Positivo | Maior churn |
| Internet: No | ✗ Negativo Alto | Reduz risco |
| Tenure (meses) | ✗ Negativo | Cliente antigo = mais leal |
| Tech Support: Yes | ✗ Negativo | Melhor suporte = menos churn |

#### Benefício para Negócio:

```
Com coeficientes, pode-se criar score caserão para cada cliente:

churn_score = sigmoid(
    0.2 × (contract_month) 
    + 0.15 × (internet_fiber) 
    - 0.03 × (tenure_months)
    - 0.25 × (tech_support)
    + ...
)

Se score > threshold (ex: 0.7), oferecer retenção
```

Isso fornece explicações diretas ao negócio: "Por que este cliente está em risco?"

---

## 📈 Resultados Comparativos

### Ranking Geral (por AUC-ROC)

```
🥇 1. XGBoost Classifier w/ RandomSearch     AUC-ROC: 0.8453 ⭐
🥈 2. Logistic Regression                     AUC-ROC: 0.8405
🥉 3. Random Forest w/ RandomSearch           AUC-ROC: 0.8354
   4. Logistic Regression NN w/ SMOTE        AUC-ROC: 0.8119
   5. Logistic Regression NN                 AUC-ROC: 0.8115
   6. Random Forest Classifier                AUC-ROC: 0.8211
   7. XGBoost Classifier                      AUC-ROC: 0.8160
   8. Logistic Regression w/ SMOTE           AUC-ROC: 0.8404
   9. Logistic Regression w/ Class Weight    AUC-ROC: 0.8398
  10. Neural Network com SMOTE               AUC-ROC: 0.7441
  11. Dummy Classifier (Baseline)             AUC-ROC: 0.5000
```

### Trade-off Recall vs Precisão

```
Recall Alto (Identifica Riscos):
├─ Logistic Regression w/ SMOTE        Recall: 0.794, Precisão: 0.518
├─ Logistic Regression w/ Class Weight Recall: 0.786, Precisão: 0.510
└─ Logistic Regression NN w/ SMOTE     Recall: 0.618, Precisão: 0.598

Balanceado (XGBoost Vencedor):
├─ XGBoost Classifier w/ RandomSearch  Recall: 0.546, Precisão: 0.673 ⭐
├─ Random Forest w/ RandomSearch       Recall: 0.537, Precisão: 0.666
└─ Logistic Regression                 Recall: 0.551, Precisão: 0.654

Precisão Alta (Poucos Falsos Alarmes):
└─ Todos os modelos ficam abaixo em recall...
```

### Performance por Dataset Size

Todos os modelos treinados com:
- **Train Set:** 80% (5634 registros)
- **Test Set:** 20% (1409 registros)
- **Estratificação:** Mantém 26% churn em ambos

### Convergência dos Modelos

```
XGBoost (300 estimadores)   ─────────────→ 0.8453 (levou mais tempo, melhor)
Random Forest (300)         ─────────────→ 0.8354 (abaixo de XGBoost)
XGBoost (100 estimadores)   ──────────→ 0.8160 (menos boosting)
Logistic Regression (padrão)──────────→ 0.8405 (simples, efetivo)
Neural Network (precoce)    ────────→ 0.7441 (underfitting, parou em 17 épocas)
```

---

## 📚 Próximas Etapas Sugeridas

1. **Validação em Produção:** Implementar o XGBoost em um subset de usuários, medir taxa real de retenção
2. **Ajuste de Threshold:** Atualmente usa 0.5. Testar 0.3-0.7 para otimizar business KPIs
3. **Feature Engineering Avançado:** Criar features baseadas em comportamento, seasonality, etc.
4. **Ensemble:** Combinar XGBoost + Logistic Regression para máximo recall e interpretabilidade
5. **Monitoramento:** Rastrear drift de dados e performance ao longo do tempo
6. **A/B Testing:** Comparar diferentes estratégias de retenção no segmento identificado

---

## 📂 Estrutura do Projeto

```
Gustavo/
├── db_setup/
│   ├── __init__.py
│   ├── query.py          # Conexão com dados
│   └── README.md
├── notebooks_eda/
│   └── eda.ipynb         # Análise exploratória e treinamento de modelos
├── mlruns/               # MLflow tracking artifacts
│   └── [múltiplas runs dos 11 modelos]
├── mlflow.db             # Banco dados MLflow local
└── README.md             # Este arquivo!
```

---

## 🔍 Como Usar os Resultados

### Para Data Scientists:
- Arquivo `eda.ipynb` contém código reproduzível
- Todos os experimentos rastreados em `mlruns/`
- Hiperparâmetros otimizados estão documentados

### Para Product/Business:
- **Métrica Principal:** AUC-ROC (0.8453 = excelente discriminação)
- **Recall Esperado:** ~55% dos churners reais identificados
- **Falsos Positivos:** ~311 clientes "suspeitos" demais
- **ROI:** Depende do custo unitário de retenção vs. lifetime value

### Para Deployment:
- Usar modelo XGBoost com Random Search CV
- Aplicar preprocessing idêntico ao treino (normalização, encoding)
- Monitorar performance monthly
- Replicar quando AUC-ROC cair abaixo 0.80

---

**Última Atualização:** Abril 2026  
**Autor:** Gustavo - FIAP ML Engineering  
**Dataset:** Telco Customer Churn (IBM/Kaggle)
