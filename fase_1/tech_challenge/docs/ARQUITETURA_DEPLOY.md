# Arquitetura de Deploy - Telco Churn Prediction

Data: 2026-05-03
Versao: 2.0
Status: Deploy em nuvem pendente

## Resumo Executivo

- O projeto possui deploy **local funcional** via Docker Compose (API + MLflow + PostgreSQL + training job).

## Estado Atual (Implementado)

```mermaid
flowchart LR
    USER["Usuario"] --> API["FastAPI :8000"]
    API --> MLFLOW["MLflow :5000"]
    TRAIN["Training on-demand"] --> MLFLOW
    MLFLOW --> DB["PostgreSQL :5432"]
    MLFLOW --> ART["./mlartifacts"]
```

## Observacao de Governanca

Os experimentos foram salvos no MLflow durante os estudos.
Os artefatos de experimento nao foram comitados no Git para manter o repositorio limpo.
