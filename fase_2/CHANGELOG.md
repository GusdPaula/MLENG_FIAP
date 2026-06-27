# Changelog

Todas as alterações relevantes deste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## Não Lançado

### Adicionado

- Novo pacote `recommender.mlflow_toolkit` com `MLflowToolkit` para configuração de experimentos, registro de dataset, registro de modelo e registro.
- Dependência MLflow adicionada ao `requirements.txt`.
- Suporte a **early stopping** na classe `Trainer` através de um novo auxiliar `EarlyStopping`. O pipeline de treinamento agora suporta early stopping quando configurado em `configs/model.yaml`.
- Suporte a **processamento em lote** em `RecommenderDataset` com um novo parâmetro `streaming` que permite amostragem negativa eficiente em memória.
- Novos utilitários `BatchCollator` e `make_batches` para manuseio explícito de lotes.
- Nova dataclass `EpochResult` para rastrear métricas de treinamento por época.
- Projeto Poetry inicializado com dependências de runtime/dev
- Ambiente virtual local `.venv` criado
- `pyproject.toml` e `poetry.lock` adicionados
- Pacote `data_pipeline` adicionado com suporte a download do Kaggle e upload para BigQuery
- `run_pipeline.py` adicionado como ponto de entrada do módulo e README
- `bigquery_query.py` adicionado para extração do BigQuery e versionamento/exportação DVC
- `ecommerce_recommender/notebooks/data_pipeline_eda.ipynb` adicionado para análise exploratória de dados
- `ecommerce_recommender/README.md` adicionado com visão geral do pacote e documentação do data_pipeline
- Configuração dedicada `fase_2/.pre-commit-config.yaml` adicionada para executar `ruff` e `pytest` no contexto fase_2
- `fase_2/ruff.toml` ajustado para excluir notebooks e usar configuração de lint correta
- Documentação abrangente de métricas (`METRICS_DOCUMENTATION.md`) explicando interpretação de métricas e por que múltiplas métricas são necessárias

### Alterado

- README raiz atualizado para descrever a estrutura do projeto, modelos, fluxo de treinamento e suporte MLflow.
- `src/README.md` atualizado para refletir o layout atual do pacote e nomes dos módulos de treinamento.
- A classe `Trainer` agora suporta tanto treinamento simples época por época quanto early stopping com `fit_with_early_stopping`.
- O `RecommenderDataset` agora suporta um modo de streaming para processamento em lote eficiente em memória.
- O pipeline de treinamento agora suporta early stopping através de configuração em `configs/model.yaml`.

### Corrigido

- Import do pipeline de treinamento agora aponta para `recommender.training.trainer`.
- **ModelFactory** agora filtra hiperparâmetros por tipo de modelo através de `MODEL_PARAM_MAP` para evitar passar parâmetros inválidos para modelos (ex: GMF agora rejeita `hidden_layers`).
- Métrica de **early stopping** alterada de AUC para NDCG@10 para melhor avaliação de classificação. Adicionado cálculo leve de NDCG@K ao método evaluate do treinador com amostragem para eficiência.
- **.gitignore** atualizado para manipular adequadamente artefatos de modelo com hierarquia de unignore para mlflow_experiments enquanto mantém modelos .pt específicos no Git conforme necessário.
- **.pre-commit-config.yaml** corrigido para garantir que hooks sejam acionados em commits executando do diretório correto onde pyproject.toml está localizado.
- **Integração DVC** melhorada para versionamento e rastreamento de dados. Adicionados datasets brutos e processados ao cache DVC, atualizado .gitignore para rastrear dvc.lock, criada configuração básica dvc.yaml.
- **Cobertura de testes** melhorada de 87% para 91% adicionando testes unitários abrangentes para metrics.py (100% de cobertura), evaluator.py (100% de cobertura), checkpoint.py (81% de cobertura) e callbacks.py (100% de cobertura).
- **Documentação e estilo** melhorados convertendo docstrings em português para estilo Google em inglês em todos os módulos e adicionando dicas de tipo de retorno ausentes.
- **Avaliação de modelo** verificou que todos os experimentos rastreiam AUC-ROC, Average Precision, HitRate@K e NDCG@K.
- `BigQueryQuery` corrigido para aceitar `dvc_repo_path` explícito e permitir versionamento local com DVC
- **Arquitetura** melhorada extraindo padrões comuns de inicialização de pesos para BaseRecommenderModel (_init_embeddings, _init_linear_layers) para reduzir duplicação de código através dos modelos GMF, NCF e MatrixFactorization.
- **Arquitetura** melhorada extraindo lógica comum de filtragem min_interactions para a classe base DataProcessor (_filter_by_min_interactions) para reduzir duplicação de código através de WeightedEventProcessor, BinaryInteractionProcessor e ImplicitFeedbackProcessor.
