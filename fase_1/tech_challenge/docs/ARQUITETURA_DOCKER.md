# Arquitetura Docker Implementada

## Visao Geral

A stack local do projeto roda com `docker-compose` e quatro servicos:

- `db` (PostgreSQL) para backend store do MLflow
- `mlflow` para tracking, registry e artifacts
- `api` (FastAPI) para inferencia
- `training` (on-demand) para treino e registro de modelos

## Diagrama da Arquitetura Docker

```mermaid
flowchart TB
    USER["Usuario / Cliente HTTP"] --> API["FastAPI API :8000"]
    USER --> MLFLOWUI["MLflow UI :5000"]

    subgraph COMPOSE["docker-compose / mlflow_network"]
        DB["db<br/>PostgreSQL :5432"]
        MLFLOW["mlflow<br/>MLflow Server :5000"]
        API["api<br/>FastAPI Inference :8000"]
        TRAIN["training<br/>Job on-demand"]
        ART["./mlartifacts:/mlartifacts"]
    end

    DB --> MLFLOW
    TRAIN --> MLFLOW
    TRAIN --> ART
    API --> MLFLOW
    MLFLOW --> ART
```

## Fluxo Final de Predicao

```mermaid
sequenceDiagram
    participant U as Usuario
    participant A as FastAPI
    participant M as ModelManager
    participant F as MLflow
    participant R as Model Registry
    participant P as Pipeline

    U->>A: POST /api/predict
    A->>M: valida payload e monta DataFrame
    M->>F: verifica /health
    F-->>M: servidor disponivel
    M->>R: carrega models:/TelcoChurnPipeline@champion
    R-->>M: retorna pipeline sklearn
    M->>P: predict / predict_proba
    P-->>A: classe e probabilidade
    A-->>U: resposta JSON
```

## Comandos Essenciais

```bash
docker-compose up -d
docker-compose run --rm training
docker-compose logs -f api
```

## Nota

Os experimentos foram salvos no MLflow, mas os artefatos de experimento nao foram comitados no Git para manter o repositorio limpo.
