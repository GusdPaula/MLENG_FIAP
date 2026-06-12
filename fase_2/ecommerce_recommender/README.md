# ecommerce_recommender

Sistema de recomendação para e-commerce usando PyTorch, com treinamento configurável, múltiplos modelos e suporte a MLflow para rastreamento de experimentos, datasets e modelos.

O projeto usa o dataset [RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) do Kaggle.

## Estrutura

```text
.
├── configs/
│   ├── base.yaml
│   ├── mlflow.yaml
│   └── model.yaml
├── data/
├── models/
├── src/
│   └── recommender/
│       ├── data/
│       ├── mlflow_toolkit/
│       ├── models/
│       ├── pipelines/
│       ├── training/
│       └── utils/
└── tests/
    ├── integration/
    └── unit/
```

## Componentes principais

| Módulo | Arquivo | Responsabilidade |
|--------|---------|------------------|
| Dataset | `src/recommender/data/dataset.py` | Carrega eventos, cria mappings de IDs e faz negative sampling |
| Processadores | `src/recommender/data/processors.py` | Estratégias para transformar eventos em interações |
| Modelos | `src/recommender/models/` | `NCF`, `GMF` e `MatrixFactorization` |
| Factory | `src/recommender/models/factory.py` | Cria modelos por nome via configuração |
| Trainer | `src/recommender/training/trainer.py` | Loop de treino com `BCELoss`, Adam e métricas de classificação |
| Métricas | `src/recommender/training/metrics.py` | `Hit Rate@K` e `NDCG@K` |
| Pipeline | `src/recommender/pipelines/train_pipeline.py` | Orquestra o fluxo completo de treino |
| MLflow Toolkit | `src/recommender/mlflow_toolkit/toolkit.py` | Centraliza setup, logging e registro no MLflow |

## Modelos implementados

### `matrix_factorization`

Baseline clássico de collaborative filtering.

- aprende embeddings de usuário e item
- inclui biases global, de usuário e de item
- usa o produto escalar entre embeddings para produzir o score

### `gmf`

Generalized Matrix Factorization.

- aprende embeddings de usuário e item
- combina os vetores com multiplicação elemento a elemento
- opcionalmente projeta a representação antes da predição final

### `ncf`

Neural Collaborative Filtering.

- aprende embeddings de usuário e item
- concatena os vetores
- passa por uma MLP com camadas configuráveis
- produz um score final com sigmoid

## Fluxo de treino

O pipeline principal está em [`src/recommender/pipelines/train_pipeline.py`](src/recommender/pipelines/train_pipeline.py).

Ele executa:

1. leitura do YAML de configuração
2. carregamento dos eventos brutos
3. aplicação do processador de dados escolhido
4. geração do dataset com negative sampling
5. split de treino e validação
6. criação do modelo via factory
7. treino e avaliação
8. cálculo de métricas de ranking
9. salvamento do artefato final

## MLflow

O projeto possui um toolkit dedicado em [`src/recommender/mlflow_toolkit/toolkit.py`](src/recommender/mlflow_toolkit/toolkit.py).

Ele serve para:

- configurar tracking URI e experiment
- iniciar runs
- logar parâmetros e métricas
- logar datasets
- registrar modelos no Model Registry

O comportamento é configurado por [`configs/mlflow.yaml`](configs/mlflow.yaml).

## Como rodar

### Com `pip`

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m recommender.pipelines.train_pipeline
PYTHONPATH=src python -m pytest tests/unit -v
```

### Com Poetry

Se o projeto estiver gerenciado por Poetry, o fluxo fica assim:

```bash
poetry install
poetry run python -m recommender.pipelines.train_pipeline
poetry run pytest tests/unit -v
```

Se o `PYTHONPATH` não estiver configurado no ambiente, use:

```bash
PYTHONPATH=src poetry run python -m recommender.pipelines.train_pipeline
PYTHONPATH=src poetry run pytest tests/unit -v
```

## Configuração do modelo

O arquivo [`configs/model.yaml`](configs/model.yaml) controla:

- tipo do modelo
- seed
- batch size
- learning rate
- epochs
- number of negatives
- mínimo de interações
- processador de dados
- hiperparâmetros do modelo

## Saída do treino

Ao final do processo, o pipeline salva um arquivo `.pt` contendo:

- tipo do modelo
- estado dos pesos
- `user2idx`
- `item2idx`
- configuração usada
- métricas finais

Isso permite reconstruir o modelo depois com o mesmo mapeamento de usuários e itens.
