# Changelog

## [2026-10-25] - Fase 4 - Primeira etapa concluída

### Resumo

A primeira etapa da fase 4 foi concluída com a implementação da base do projeto de machine learning para predição de default de crédito, incluindo organização de estrutura de código, validação de dados, engenharia de features, treinamento do modelo e pipeline de predição.

### O que foi realizado

#### 1. Estrutura inicial do projeto
- Organização do repositório em módulos lógicos:
  - `src/default_pred/data`
  - `src/default_pred/features`
  - `src/default_pred/models`
  - `src/default_pred/pipelines`
- Criação da configuração central em `src/default_pred/config.py`.
- Definição do diretório de dados, modelos e colunas alvo.

#### 2. Carregamento e validação de dados
- Implementação do carregamento de datasets a partir de CSV.
- Criação da função `load_one_entry()` para validar uma única linha de entrada.
- Definição do contrato de dados com Pandera em `src/default_pred/data/contract.py`.
- Garantia de rejeição de entradas inválidas antes da predição.

#### 3. Engenharia de features
- Desenvolvimento do processo de engenharia em `src/default_pred/features/engineering.py`.
- Ajuste de target para classificação binária.
- Remoção de colunas irrelevantes ou com alta taxa de missing values.
- Encoding de variáveis categóricas e binárias.
- Transformação de datas para diferença em dias a partir de hoje.
- One-hot encoding para variáveis categóricas com baixa cardinalidade.

#### 4. Pré-processamento e treinamento
- Implementação de `split_and_scale()` em `src/default_pred/features/preprocessing.py`.
- Uso de `StandardScaler` para normalizar variáveis numéricas.
- Salva do scaler e das colunas de features para uso futuro.
- Treinamento do modelo de regressão logística em `src/default_pred/models/train.py`.
- Avaliação do modelo com métricas de classificação, como acurácia, precisão, recall, F1-score e ROC AUC.

#### 5. Pipeline de predição
- Criação do módulo `src/default_pred/models/predict.py`.
- Reindexação do DataFrame para respeitar a ordem exata das colunas usadas no treinamento.
- Aplicação do scaler no fluxo de inferência.
- Validação do fluxo para predição de um único registro.

#### 6. Pipelines executáveis
- Criação dos scripts em `src/default_pred/pipelines/` para:
  - treinamento (`pipeline_train.py`)
  - predição (`pipeline_predict.py`)
- Armazenamento dos artefatos gerados em `models/`.

#### 7. Testes automatizados
- Criação de testes para validar:
  - predição com entrada válida;
  - rejeição de entrada inválida pelo contrato.
- Arquivos criados em `tests/`:
  - `test_predict_single_entry.py`
  - `test_predict_single_fail_contract.py`

#### 8. Configuração do ambiente e qualidade de código
- Configuração do projeto com `uv` para gerenciamento do ambiente.
- Criação do arquivo `pyproject.toml` com configuração do projeto e do Ruff.
- Geração do `uv.lock` para lock de dependências.
- Aplicação da padronização de código com Ruff.

### Validação executada

Os testes relevantes foram executados com sucesso:

- predição com entrada válida: OK
- rejeição de entrada inválida pelo contrato: OK
- checagem de lint/format com Ruff: OK

### Observações

- O projeto já está em uma base funcional para continuar para as próximas etapas da fase 4.
- Há uma advertência de depreciação do Pandas no módulo de feature engineering, porém ela não bloqueia o funcionamento atual.
- O resultado atual demonstra que a proposta foi implementada até a etapa de treinamento e inferência de um registro isolado.

### Status

Status geral: concluída a primeira etapa da fase 4.

---

## [2026-10-25] - Fase 4 - Etapa 2 concluída (cumulativa)

### Resumo

A etapa 2 da fase 4 foi concluída como evolução cumulativa da etapa 1. O projeto passou a suportar entradas com múltiplos registros, normalização de payloads driftados, validação e pré-processamento batch, além de detecção de drift com Evidently e ajuste do ambiente para manter a execução estável em conjunto com o restante do fluxo de ML.

### O que foi realizado

#### 1. Suporte a múltiplas entradas
- Implementação da função `load_entries()` para carregar payloads em formato de dicionário com vários registros.
- Ajuste do pipeline de predição para distinguir entre entrada única e lote de registros.
- Garantia de que arquivos como `loan_data_drifted.json` sejam tratados como dataset completo e não como uma única linha.

#### 2. Normalização de entradas driftadas
- Criação de lógica para transformar registros com estrutura driftada em formato compatível com o DataFrame esperado pelo modelo.
- Ajuste do tratamento de valores aninhados e campos com chaves heterogêneas.
- Correção de conversão de payloads mal formatados para a estrutura de dados da aplicação.

#### 3. Detecção de drift com Evidently
- Integração do módulo `src/default_pred/models/drift.py` ao fluxo de predição.
- Ajuste da referência de treinamento para comparação com dados atuais.
- Reparo do problema de carregamento de artefatos em tempo de importação.
- Sanidade da comparação de colunas para que o DataFrame atual fique alinhado ao schema do treino.
- Eliminação de colunas com variância nula ou quase nula antes do cálculo de correlação do Evidently para evitar erros em divisão por zero.
- Geração do relatório HTML de drift em `data_drift_report.html`.

#### 4. Compatibilidade e estabilidade do ambiente
- Ajuste do `pyproject.toml` para manter versões compatíveis de pandas, numpy e Evidently.
- Re-solução do ambiente com `uv` para estabilizar dependências e evitar quebra por incompatibilidade de pacote.
- Prevenção de falhas no fluxo principal quando a detecção de drift não puder executar corretamente para uma amostra insuficiente.

#### 5. Validação de regressão
- Criação e evolução de testes para validar:
  - entrada single-row válida;
  - rejeição de entrada inválida pelo contrato;
  - processamento de payload driftado com múltiplas linhas;
  - comportamento seguro da detecção de drift em amostras pequenas.

#### 6. Qualidade e execução
- Ajustes de Ruff e limpeza final de alguns problemas de lint.
- Validação do pipeline principal com execução real do fluxo de predição sobre entrada batch.

### Validação executada

Os testes e execuções relevantes foram confirmados com sucesso:

- predição de entrada válida: OK
- rejeição por contrato: OK
- carregamento de lote driftado: OK
- detecção de drift em dados batch: OK
- execução real do pipeline com predição em lote: OK

### Observações

- Esta etapa foi incorporada de forma cumulativa sobre a etapa 1, sem substituir o que já havia sido implementado.
- O projeto segue funcional e preparado para continuar o desenvolvimento das próximas etapas da fase 4.

### Status

Status geral: concluída a etapa 2 da fase 4, acumulada sobre a etapa 1.
