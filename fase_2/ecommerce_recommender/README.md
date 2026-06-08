# ecommerce_recommender

Sistema de recomendação para e-commerce usando Neural Collaborative Filtering (NCF) com PyTorch, treinado no dataset [RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) do Kaggle.

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

## Componentes implementados

| Módulo | Arquivo | Responsabilidade |
|--------|---------|-----------------|
| Modelo NCF | `src/recommender/models/ncf.py` | Embeddings de user/item + MLP com dropout e inicialização Xavier/Kaiming |
| Dataset | `src/recommender/data/dataset.py` | Carga de eventos, mapeamento de IDs, negative sampling |
| Trainer | `src/recommender/training/trainer.py` | Loop de treino com Adam + BCELoss, avaliação com AUC-ROC e Average Precision |
| Métricas | `src/recommender/training/metrics.py` | Hit Rate@K e NDCG@K |
| Pipeline | `src/recommender/pipelines/train_pipeline.py` | Orquestração end-to-end (load → filtro cold-start → treino → avaliação → salva modelo) |
| Testes | `tests/unit/` | Validação do modelo e dataset |

## Resultados do treino

| Métrica | Valor |
|---------|-------|
| AUC-ROC | 0.9078 |
| Average Precision | 0.8196 |
| Hit Rate@10 | 0.1330 |
| NDCG@10 | 0.0764 |

<<<<<<< HEAD:fase_2/ecommerce-recommender/README.md
## Como rodar

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Baixar dataset do Kaggle
python scripts/download_data.py

# 3. Treinar o modelo
PYTHONPATH=src python -m recommender.pipelines.train_pipeline

# 4. Rodar testes
PYTHONPATH=src python -m pytest tests/unit/ -v
```

## Hiperparâmetros

Configuráveis em `configs/model.yaml`:

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `embedding_dim` | 64 | Dimensão dos vetores de representação de user/item |
| `hidden_layers` | [128, 64, 32] | Camadas ocultas da MLP |
| `dropout` | 0.2 | Taxa de dropout para regularização |
| `learning_rate` | 0.001 | Taxa de aprendizado do Adam |
| `epochs` | 10 | Número de épocas de treino |
| `batch_size` | 256 | Tamanho do batch |
| `num_negatives` | 4 | Exemplos negativos por interação positiva |
| `min_interactions` | 5 | Mínimo de interações para filtro cold-start |
=======
## Data Pipeline Module

O diretório `data_pipeline/` contém a implementação de ingestão e exportação de dados:

- `kaggle_data_loader.py`: baixa e prepara os dados do Kaggle.
- `bigquery_uploader.py`: carrega CSVs para BigQuery.
- `bigquery_query.py`: extrai dados de BigQuery para CSV local e versiona com DVC.
- `pipeline.py`: orquestra a execução completa de download, combinação e upload.
- `run_pipeline.py`: entrypoint para execução com variáveis de ambiente ou parâmetros de linha de comando.

Use este módulo para centralizar extração de dados antes de construir o modelo e os artefatos de rastreamento.

Essa estrutura atende ao requisito de diretórios base e facilita evoluir para as próximas etapas (Poetry, Docker, DVC e MLflow).
>>>>>>> main:fase_2/ecommerce_recommender/README.md
