# ecommerce_recommender

Estrutura base para o Tech Challenge Fase 02 (Etapa 1: Clean Code e Estrutura).

## Estrutura de pastas

```text
.
├── configs/
│   ├── base.yaml
│   ├── mlflow.yaml
│   └── model.yaml
├── data/
│   ├── external/
│   ├── interim/
│   ├── processed/
│   └── raw/
├── models/
├── src/
│   └── recommender/
│       ├── data/
│       ├── features/
│       ├── models/
│       ├── pipelines/
│       ├── training/
│       └── utils/
└── tests/
	├── integration/
	└── unit/
```

## Responsabilidade de cada pasta

- `src/`: código-fonte do projeto (lógica de negócio e pipeline).
- `tests/`: testes unitários e de integração.
- `data/`: dados versionados e intermediários do pipeline.
- `models/`: artefatos de modelo treinado (pesos, serializações, exportações).
- `configs/`: configurações declarativas (app, modelo, MLflow, paths).

## Convenções recomendadas para a Etapa 1

- `src/recommender/pipelines/`: orquestração de pré-processamento, treino e avaliação.
- `src/recommender/models/`: definição de arquitetura e factory para criar modelos.
- `src/recommender/features/`: transformações de features e estratégias de pré-processamento.
- `src/recommender/training/`: loops de treino, validação e métricas.
- `src/recommender/utils/`: funções utilitárias pequenas e coesas.

## Data Pipeline Module

O diretório `data_pipeline/` contém a implementação de ingestão e exportação de dados:

- `kaggle_data_loader.py`: baixa e prepara os dados do Kaggle.
- `bigquery_uploader.py`: carrega CSVs para BigQuery.
- `bigquery_query.py`: extrai dados de BigQuery para CSV local e versiona com DVC.
- `pipeline.py`: orquestra a execução completa de download, combinação e upload.
- `run_pipeline.py`: entrypoint para execução com variáveis de ambiente ou parâmetros de linha de comando.

Use este módulo para centralizar extração de dados antes de construir o modelo e os artefatos de rastreamento.

Essa estrutura atende ao requisito de diretórios base e facilita evoluir para as próximas etapas (Poetry, Docker, DVC e MLflow).
