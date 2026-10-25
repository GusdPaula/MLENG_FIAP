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
