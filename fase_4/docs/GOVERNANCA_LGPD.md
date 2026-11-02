# Governança de Dados, Conformidade LGPD, Equidade Algorítmica e Análise Causal

Este documento consolida a arquitetura de **Governança, Privacidade (*Privacy by Design* e *Privacy by Default*), Mitigação de Vieses (*Fairness*) e Análise Causal** do sistema de *Credit Scoring* (`src/default_pred/`). Todas as diretrizes aqui descritas refletem fielmente as transformações aplicadas pelo código de engenharia de atributos, contratos de dados e observabilidade em produção.

---

## 1. *Privacy by Design* e *Privacy by Default* — Mapeamento de PII

O modelo baseline foi construído sobre o dataset **Lending Club Loan Data** (`data/raw/loan_data.csv`, amostrado em 5.000 registros representativos com 151 colunas na origem). Seguindo os princípios de **Minimização de Dados** e **Privacidade por Padrão** (*Privacy by Default*), o pipeline [`feature_engineering()`](../src/default_pred/features/engineering.py) atua como uma barreira sanitizadora antes que qualquer dado chegue ao treinamento ([`train.py`](../src/default_pred/models/train.py)) ou à inferência ([`predict.py`](../src/default_pred/models/predict.py)).

### 1.1 Matriz de Tratamento de Identificadores e Quase-Identificadores

| Atributo no Dataset Bruto | Categoria de Privacidade (LGPD) | Risco de Reidentificação / Exposição | Regra Implementada em `src/default_pred/features/engineering.py` | Destino no Modelo (`feature_columns.pkl`) |
| :--- | :--- | :--- | :--- | :--- |
| **`id`** | Identificador Direto (ID do empréstimo) | Alto (vinculação direta ao contrato do titular) | Descarte explícito em `df.drop(columns=["id", "title"])` (linha 37) | **Removido** |
| **`member_id`** | Identificador Direto (ID do cliente) | Crítico (identificação individual direta) | Descartado automaticamente pelo filtro de colunas com $>20\%$ de valores ausentes (`df.isna().mean() * 100 > 20`, linha 19) | **Removido** |
| **`url`** | Identificador Indireto (Link da proposta) | Alto (contém o identificador único na URL da plataforma) | Descartado pelo filtro de alta cardinalidade categórica ($>20\%$ de valores únicos em colunas `object`, linhas 40–46) | **Removido** |
| **`title`** | Texto Livre informacional | Médio (pode conter nomes próprios ou detalhes pessoais no título do empréstimo) | Descarte explícito em `df.drop(columns=["id", "title"])` (linha 37) | **Removido** |
| **`emp_title`** | Quase-Identificador / Texto Livre (Cargo/Empregador) | Alto (combinação de cargo específico + empregador + CEP permite singularização do indivíduo) | Descartado pelo filtro de alta cardinalidade categórica ($>20\%$ de valores únicos em colunas `object`, linhas 40–46) | **Removido** |
| **`desc`** | Texto Livre (Descrição detalhada do pedido) | Crítico (textos abertos frequentemente contêm PII, dados familiares ou de saúde) | Descartado pelo filtro de $>20\%$ de valores ausentes (linha 19) e alta cardinalidade textual (linhas 40–46) | **Removido** |
| **`zip_code`** | Quase-Identificador Geográfico (CEP parcial) | Médio (pseudonimizado na fonte no formato `XXXxx`, mantendo apenas os 3 primeiros dígitos) | Extração apenas do prefixo numérico de 3 dígitos via regex `r"(\d+)"` e conversão para `float` (linhas 118–123) | **Mantido (Pseudonimizado 3 dígitos)** — monitorado sob controle de *Fairness* |
| **`addr_state`** | Quase-Identificador Geográfico (UF/Estado) | Baixo isoladamente; Médio quando combinado a renda e CEP | Codificado via *One-Hot Encoding* (`pd.get_dummies`, linhas 131 e 139) em variáveis binárias (`addr_state_CA`, `addr_state_NY`, etc.) | **Mantido (Agregado Estadual)** — monitorado sob controle de *Fairness* |

### 1.2 Controles Técnicos de *Privacy by Default* nos Logs e Experimentos
1. **Desacoplamento de Payload nos Logs Operacionais:** O arquivo rotativo `logs/loan_prediction.log` ([`utils/logging.py`](../src/default_pred/utils/logging.py)) registra exclusivamente metadados de ciclo de vida (`experiment`, `run_name`, `status`, `duration_ms`, `rows`) e alertas agregados de drift. Nenhum valor individual de renda, CEP ou documento de clientes é impresso nos arquivos de log.
2. **Rastreamento Agregado no MLflow:** O adapter [`MlflowTracker`](../src/default_pred/observability/mlflow_tracker.py) persiste hiperparâmetros, métricas estatísticas globais (`share_drifted`, `psi.<feature>`, `drift_score.<feature>`, `default_rate_predicted`) e artefatos de modelo, evitando o armazenamento de registros nominais de clientes no banco `mlflow.db`.

---

## 2. Enquadramento Jurídico na LGPD (Lei nº 13.709/2018)

### 2.1 Princípios Atendidos (Art. 6º da LGPD)
* **Finalidade e Adequação (Art. 6º, I e II):** Os dados financeiros e cadastrais são processados com o propósito específico, legítimo e explícito de estimar a probabilidade de inadimplência (*default*) na concessão de crédito.
* **Necessidade / Minimização (Art. 6º, III):** Das 151 colunas brutas, identificadores diretos (`id`, `member_id`, `url`) e campos de texto livre (`title`, `emp_title`, `desc`) são eliminados no pré-processamento, retendo-se apenas variáveis estritamente correlacionadas à capacidade de pagamento e histórico creditício.
* **Qualidade dos Dados (Art. 6º, V):** Materializado tecnicamente na camada de ingestão por meio do contrato de dados Pandera ([`Contract`](../src/default_pred/data/contract.py)), que impede que registros corrompidos, nulos em campos críticos (`purpose`, `grade`) ou com valores fora de domínio (`annual_inc <= 20000`) gerem decisões automatizadas incorretas.
* **Transparência e Prestação de Contas (Art. 6º, VI e X):** Materializado na camada de monitoramento e observabilidade via relatórios auditáveis de drift (`data_drift_report.html`) e rastreabilidade completa de execuções no MLflow (`credit_training`, `credit_inference`, `credit_drift_monitoring`).
* **Prevenção e Segurança (Art. 6º, VII e VIII):** Bloqueio *fail-fast* na ingestão e monitoramento contínuo de degradação estatística (*Data Drift* e *Concept Drift*).

### 2.2 Bases Legais para o Tratamento em Concessão de Crédito (Art. 7º da LGPD)
O tratamento automatizado realizado pelo modelo fundamenta-se em duas bases legais primárias da Lei nº 13.709/2018:
1. **Proteção do Crédito (Art. 7º, inciso X):** Base legal específica para modelos de *Credit Scoring* e análise de risco de inadimplência no mercado financeiro e em fintechs, autorizando o uso de histórico de crédito, comprometimento de renda e pontuações (*grades*) para avaliação de solvência.
2. **Execução de Contrato ou Procedimentos Preliminares (Art. 7º, inciso V):** Aplicável no fluxo de solicitação de empréstimo pelo titular, no qual a validação cadastral e a inferência do modelo constituem procedimentos preliminares indispensáveis à celebração do contrato de mútuo.

### 2.3 Direitos do Titular e Explicabilidade de Decisões Automatizadas (Art. 18 e Art. 20)
* **Direito de Acesso, Correção e Eliminação (Art. 18):** O titular pode requisitar a confirmação da existência de tratamento e a retificação de dados cadastrais desatualizados (como atualização de `annual_inc` ou `verification_status`), sendo bloqueada qualquer decisão baseada em payload que viole o contrato de qualidade.
* **Direito à Revisão e Explicação de Decisões Automatizadas (Art. 20, caput e § 1º):**
  * A escolha arquitetural pela **Regressão Logística** (`LogisticRegression` em [`train.py`](../src/default_pred/models/train.py)) com padronização explícita (`StandardScaler` salvo em `models/scaler.pkl`) oferece **interpretabilidade intrínseca (*Glass-Box Model*)**.
  * Para qualquer predição individual $P(Y=1 \mid x) = \sigma(\beta_0 + \sum_j \beta_j x_j)$, a contribuição marginal exata de cada variável padronizada $x_j$ é dada pelo produto $\beta_j \cdot x_j$ (log-odds).
  * Isso permite fornecer ao cliente e aos órgãos reguladores (BACEN / ANPD) os critérios objetivos que motivaram uma eventual recusa (ex.: elevada razão `loan_amnt / annual_inc`, histórico de atrasos ou faixa de risco `grade`), sem violar segredos comerciais.

---

## 3. Política de Retenção, Armazenamento e Descarte Seguro (Art. 15 e Art. 16 da LGPD)

Em conformidade com os **Artigos 15 e 16 da LGPD** (término do tratamento e eliminação de dados após cumprida a finalidade ou encerrado o prazo legal/regulatório), a tabela abaixo define o ciclo de vida de cada ativo de dados e artefato gerado pelos pipelines:

| Ativo / Caminho no Repositório | Natureza do Conteúdo | Finalidade Operacional | Prazo de Retenção em Produção | Controle de Acesso / Versionamento | Critério de Expurgo / Descarte Seguro |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`data/raw/loan_data.csv`** | Base histórica bruta (5.000 linhas, 151 colunas) | Treinamento inicial e referência para regeneração de cache | **Até 5 anos** após encerramento da obrigação contratual (prazo prescricional civil/BACEN) | Ignorado no Git (`data/raw/*` no `.gitignore`); baixado sob demanda via `kagglehub` | Exclusão segura (*shredding*) ou truncamento irreversível de identificadores após geração do conjunto processado |
| **`data/processed/*.pkl`** (`X_train.pkl`, `X_test.pkl`, `y_train.pkl`, `y_test.pkl`, `loan_data_processed.pkl`) | Matrizes numéricas pré-processadas e escalonadas (sem PII direto) | Dataset de Referência para detecção de drift e avaliação do modelo | **Ciclo de vida da versão ativa do modelo + 12 meses** para auditoria estatística | Ignorado no Git (`data/processed/*` no `.gitignore`) | Sobrescrita e descarte automático quando uma nova baseline de modelo é promovida e arquivada |
| **`data/entries/*.json`** (`loan_data_row.json`, `loan_data_row_fail.json`, `loan_data_drifted.json`, `loan_data_concept_drifted.json`) | Payloads sintéticos/amostrais de teste de contrato e simulação de drift | Testes automatizados (`pytest`), validação de contratos e simulação de estresse | **Permanente para fixtures sintéticas**; **30 dias** para lotes reais de inferência em produção | Fixtures de teste versionadas; lotes produtivos devem residir em bucket com *Object Lifecycle Management* | Expurgo automatizado (TTL de 30 dias) após consolidação das métricas agregadas no MLflow |
| **`models/*.pkl`** (`logistic_regression_model.pkl`, `scaler.pkl`, `feature_columns.pkl`) | Pesos agregados do classificador linear, médias/desvios do scaler e lista de colunas | Execução de inferência e alinhamento de schema em produção | **24 meses** após descontinuação da versão do modelo (para auditoria de decisões passadas — Art. 20 LGPD) | Versionado por release ou registrado no Model Registry do MLflow | Arquivamento frio (*cold storage*) e remoção após expiração do prazo regulatório de contestação |
| **`logs/loan_prediction.log`** | Logs operacionais estruturados de execução dos pipelines | Monitoramento de saúde operacional, auditoria de falhas de contrato e alertas de drift | **90 dias** em armazenamento quente; rotação automática configurada em `setup_logging()` | Rotação automática por tamanho (`RotatingFileHandler`: máx. `1 MB` por arquivo, `backupCount=3`) | Sobrescrita circular automática pelo `RotatingFileHandler` ao atingir o limite de backups |
| **`mlflow.db` e `mlruns/`** | Banco SQLite de metadados de experimentos e artefatos HTML de drift (`data_drift_report.html`) | Governança de MLOps, rastreabilidade histórica de métricas e evidência de monitoramento | **18 meses** para runs de auditoria de drift e treino | Listados no `.gitignore` para ambientes produtivos (`mlflow.db`, `mlruns/`) | Comando `mlflow gc` (Garbage Collection) após exclusão lógica de experimentos expirados |

---

## 4. Mitigação de Vieses Algorítmicos e Justiça (*Fairness*)

### 4.1 Diagnóstico de Variáveis *Proxy* e Risco de *Redlining* Indireto
Mesmo após a remoção de identificadores pessoais diretos e de atributos protegidos explícitos (como gênero, raça ou estado civil, que não integram o vetor de features), um modelo de crédito pode sofrer de **discriminação indireta (*Disparate Impact*)** caso utilize variáveis altamente correlacionadas com recortes demográficos vulneráveis (Art. 6º, IX da LGPD — Princípio da Não Discriminação):

1. **`zip_code` (3 primeiros dígitos do CEP) e `addr_state` (Unidade Federativa):**
   * No pipeline atual ([`engineering.py`](../src/default_pred/features/engineering.py#L118-L139)), o prefixo de 3 dígitos do CEP é mantido como variável numérica contínua e o estado (`addr_state`) é expandido via *One-Hot Encoding*.
   * **Risco Técnico e Ético:** Regiões postais (3-digit ZIP) historicamente apresentam forte segregação socioeconômica e demográfica. Tratar `zip_code` como variável numérica contínua também introduz uma ordenação artificial entre regiões geográficas. Isso pode configurar *redlining* algorítmico (penalização de solicitantes solventes apenas pelo código postal de residência).
2. **`annual_inc` (Renda Anual) e Regra Rígida do Contrato (`gt=20_000`):**
   * O contrato Pandera ([`Contract`](../src/default_pred/data/contract.py#L10-L13)) bloqueia entradas com `annual_inc <= 20000`. Em cenários de compressão salarial inflacionária (-30% na renda real), esse corte estático pode excluir desproporcionalmente trabalhadores jovens ou de regiões com menor custo de vida.

### 4.2 Controles Prescritivos de Governança e *Fairness*
Para mitigar esses riscos no ciclo de vida do modelo, ficam estabelecidos três controles prescritivos:
* **Controle F1 — Auditoria de Razão de Impacto Desproporcional (*Disparate Impact Ratio — Regra dos 80%*):**
  Antes de promover qualquer retreinamento em `credit_training`, calcular a taxa de aprovação ($P(\hat{Y}=0)$) segmentada por macro-região (`addr_state` / faixas de `zip_code`):
  $$\text{DI} = \frac{P(\hat{Y} = 0 \mid \text{Grupo Regional Protegido})}{P(\hat{Y} = 0 \mid \text{Grupo Regional de Referência})} \ge 0.80$$
* **Controle F2 — Igualdade de Oportunidade (*Equal Opportunity Difference*):**
  Monitorar a diferença na Taxa de Verdadeiros Positivos (e Falsos Positivos — clientes bons classificados erroneamente como inadimplentes) entre estados para garantir que o erro do modelo não recaia sistematicamente sobre uma mesma região.
* **Controle F3 — Remoção Programada de `zip_code` na Próxima Revisão do Classificador:**
  Em conformidade com a minimização de efeito discriminatório, prescreve-se a exclusão da coluna `zip_code` do vetor de treinamento na revisão `v2` do pipeline, substituindo-a exclusivamente por indicadores diretos de capacidade financeira individual (`dti`, `installment / annual_inc`, `revol_util`).

---

## 5. Análise Causal da Degradação do Modelo em Produção

### 5.1 Contexto Macroeconômico e Grafo Causal (DAG)
A suspeita da diretoria da fintech — de que o modelo de *Credit Scoring* está tomando decisões erradas de forma silenciosa — decorre de um **choque macroeconômico externo** (inflação persistente combinada com aperto monetário e elevação da taxa básica de juros). Esse choque atua como uma **variável confundidora não observada pelo modelo ($Z$)** que afeta simultaneamente as variáveis de entrada ($X$) e a variável resposta ($Y = \text{loan\_status}$).

```
+-----------------------------------------------------------------------------------+
|                   GRAFO CAUSAL (DAG) DA DEGRADACAO EM PRODUCAO                    |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|            +-------------------------------------------------------+              |
|            |   Z: Choque Macroeconomico (Inflacao + Alta de Juros) |              |
|            +-------------------------------------------------------+              |
|                 |                        |                        |               |
|    Compressao   |                        | Aumento da necessidade | Corrosao da   |
|    da renda     |                        | de liquidez/credito    | capacidade de |
|    real (-30%)  |                        | (+30% no valor pedido) | pagamento     |
|                 v                        v                        |               |
|     +----------------------+  +----------------------+            |               |
|     | X1: annual_inc (x0.7)|  | X2: loan_amnt (x1.3) |            |               |
|     +----------------------+  +----------------------+            |               |
|                 \                        /                        |               |
|                  \                      /                         |               |
|                   v                    v                          v               |
|            +----------------------------------+        +--------------------+     |
|            | Razao Comprometimento (X2 / X1)  | -----> | Y: Inadimplencia   |     |
|            | sobe de ~24% para ~45% (+85.7%)  |        | Real (loan_status) |     |
|            +----------------------------------+        +--------------------+     |
|                             |                                     ^               |
|                             | Fronteira P(Y|X) aprendida          |               |
|                             | antes do choque fica obsoleta       |               |
|                             v                                     |               |
|            +----------------------------------+                   |               |
|            | Y_hat: Predicao do Modelo Base   | ------------------+               |
|            | (Subestima o risco real P(Y|X))  |   Erro Silencioso (Falsos Neg.)   |
|            +----------------------------------+                                   |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

### 5.2 Formalização Matemática, Taxonomia Temporal e *Feedback Delay* ($P(X,Y) = P(X)\cdot P(Y \mid X)$)

Sob a regra da probabilidade conjunta $P(X, Y) = P(X) \cdot P(Y \mid X)$, o módulo [`src/default_pred/models/drift_simulation.py`](../src/default_pred/models/drift_simulation.py) e o pipeline [`src/default_pred/pipelines/pipeline_simulate_drift.py`](../src/default_pred/pipelines/pipeline_simulate_drift.py) (em conjunto com o detector Evidently em [`src/default_pred/models/drift.py`](../src/default_pred/models/drift.py)) decompõem e monitoram as três variações de distribuição e o dilema operacional do **Feedback Delay ($t \to t + k$)**:

| Dimensão | 1. *Covariate Shift* (*Data Drift*) | 2. *Prior Probability Shift* | 3. *Concept Drift Real* (*Posterior Shift*) |
| :--- | :--- | :--- | :--- |
| **Definição Matemática** | $P_{\text{prod}}(X) \neq P_{\text{ref}}(X)$, mantendo $P(Y \mid X)$ constante | $P_{\text{prod}}(Y) \neq P_{\text{ref}}(Y)$, mantendo $P(X \mid Y)$ constante | $P_{\text{prod}}(Y \mid X) \neq P_{\text{ref}}(Y \mid X)$ (a verdade fundamental / fronteira de decisão muda) |
| **Mecanismo Econômico Simulado** | A inflação comprime a renda disponível em **-30%** (`annual_inc * 0.7`) e encarece o crédito solicitado em **+30%** (`loan_amnt * 1.3`). | Desbalanceamento repentino na proporção global de inadimplentes da carteira (de ~4,1% para ~51,1%), exigindo calibração de *threshold*. | Com o custo de vida e juros elevados, clientes que antes pagavam em dia (`Fully Paid = 0`) sob `loan_amnt / annual_inc >= 0.25` (ou *grades* `D–G`) entram em colapso financeiro e passam a `Charged Off = 1`. |
| **Taxonomia Temporal** | **Incremental** (corrosão inflacionária contínua mês a mês) | **Gradual / Súbito** | **Combinado (Incremental + Súbito)**: inflação contínua somada a choque abrupto na taxa básica de juros. |
| **Implementação no Código** | [`simulate_data_drift()`](../src/default_pred/models/drift_simulation.py) e lote `data/entries/loan_data_drifted.json`. | Calibração de limiar em [`_calibrate_predictions()`](../src/default_pred/models/drift_simulation.py) e métrica `concept_drift.default_rate_delta`. | [`simulate_concept_drift()`](../src/default_pred/models/drift_simulation.py), [`detect_concept_drift_ddm()`](../src/default_pred/models/drift_simulation.py) e [`evaluate_concept_drift_impact()`](../src/default_pred/models/drift_simulation.py). |
| **Detecção Pré-Rótulo ($t \to t+k$, sem *Ground Truth*)** | • **KS-test (Evidently):** $p\text{-value} < 0.05$ para numéricas.<br/>• **Z-test (Evidently):** para binárias.<br/>• **PSI de Features:** $\text{PSI} = \sum (\text{Real}_i - \text{Base}_i)\ln(\text{Real}_i/\text{Base}_i) \ge 0.20$ (`psi.annual_inc`, `psi.loan_amnt`). | • **Prediction Drift PSI:** monitoramento de `psi.predicted_prob` e `default_rate_predicted` sobre as saídas $\hat{P}(Y=1 \mid X)$. | Indicado indiretamente pela combinação de *Data Drift* severo nas features de maior peso causal com *Prediction Drift*. |
| **Detecção Pós-Rótulo ($t + k$, após maturação de 90 dias)** | N/A (independente de $Y$) | Auditoria da prevalência real observada `concept_drift.default_rate`. | • **Algoritmo DDM (*Drift Detection Method*):** monitora a taxa de erro $p_t$ e $s_t = \sqrt{p_t(1-p_t)/t}$, disparando **Warning** ($p_t + s_t \ge p_{\min} + 2s_{\min}$) e **Drift** ($p_t + s_t \ge p_{\min} + 3s_{\min}$, `ddm.drift_triggered = 1.0`).<br/>• Queda supervisionada em **ROC-AUC**, **Recall**, **F1-Score** e **Accuracy**. |

### 5.3 Protocolo Prescritivo de Resposta Operacional e Adaptação em MLOps

A camada de observabilidade ([`pipeline_drift.py`](../src/default_pred/pipelines/pipeline_drift.py) e [`pipeline_simulate_drift.py`](../src/default_pred/pipelines/pipeline_simulate_drift.py)) registra automaticamente no experimento `credit_drift_monitoring` do MLflow as métricas pré-rótulo (`share_drifted`, `alert_triggered`, `psi.<feature>`, `psi.predicted_prob`) e pós-rótulo (`ddm.warning_triggered`, `ddm.drift_triggered`, `concept_drift.roc_auc_delta`). Quando um alerta é disparado, a equipe de MLOps e Risco executa o protocolo em 4 níveis:

1. **Nível 1 — Triagem Imediata Pré-Rótulo ($t \to t+k$: `alert_triggered == 1.0` ou `PSI >= 0.20`):**
   * Verificar nos logs (`logs/loan_prediction.log`) e no experimento `credit_inference` se houve aumento de falhas do contrato Pandera (`run_success == 0.0`).
   * Se o contrato Pandera passou mas `psi.annual_inc >= 0.20` e `psi.loan_amnt >= 0.20` (com deslocamento em `psi.predicted_prob`), confirma-se mudança genuína no perfil econômico dos solicitantes (*Covariate Shift*), antecipando o risco antes do *Feedback Delay* de 90 dias.
2. **Nível 2 — Contenção de Curto Prazo (*Threshold Tuning* para *Prior Shift* e *Concept Drift*):**
   * Recalibrar preventivamente o limiar de decisão da Regressão Logística (`concept_drift.decision_threshold`) para conter a aprovação de falsos negativos enquanto os rótulos reais da nova safra amadurecem.
3. **Nível 3 — Janelas Temporais Ponderadas (*Time-Decay Sliding Windows*) e Enriquecimento Causal ($t + k$ com `ddm.drift_triggered == 1.0`):**
   * Quando o detector **DDM** confirma a quebra da fronteira $P(Y \mid X)$ ($p_t + s_t \ge p_{\min} + 3s_{\min}$), descartar ou subponderar amostras antigas usando decaimento exponencial temporal $w_i = e^{-\lambda(t_{\text{atual}} - t_i)}$ e incorporar a feature causal explícita `loan_to_income = loan_amnt / annual_inc`.
4. **Nível 4 — Retreinamento Governado e Deploy *Shadow / Champion-Challenger* (`credit_training`):**
   * Treinar o modelo desafiante (*Challenger*) sobre a janela recente rotulada com `class_weight="balanced"`, validar a **Regra dos 80% (*Disparate Impact*)**, executá-lo em **Modo Shadow** em paralelo ao modelo *Champion* e promovê-lo gradualmente (*Canary Rollout*: 5% $\to$ 20% $\to$ 100%) no MLflow, atualizando a referência `data/processed/X_train.pkl`.
