# Registro de alterações

Todas as alterações relevantes deste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased version]
- Inicializar o projeto Poetry e adicionar dependências de runtime/dev
- Criar ambiente virtual local `.venv`
- Adicionar `pyproject.toml` e `poetry.lock`
- Adicionar pacote `data_pipeline` com suporte a download do Kaggle e upload para BigQuery
- Adicionar `run_pipeline.py` como ponto de entrada e README do módulo
- Adicionar `bigquery_query.py` para extração de BigQuery e exportação/versionamento com DVC
- Adicionar notebook `ecommerce_recommender/notebooks/data_pipeline_eda.ipynb` para análise exploratória
- Atualizar `.gitignore` para permitir rastrear `poetry.lock` e ignorar arquivos `.env` locais
- Adicionar `ecommerce_recommender/README.md` com visão geral do pacote e documentação do data_pipeline
- Adicionar configuração dedicada `fase_2/.pre-commit-config.yaml` para executar `ruff` e `pytest` no contexto fase_2
- Ajustar `fase_2/ruff.toml` para excluir notebooks e usar a configuração de lint correta
- Corrigir `BigQueryQuery` para aceitar `dvc_repo_path` explícito e permitir versionamento local com DVC
