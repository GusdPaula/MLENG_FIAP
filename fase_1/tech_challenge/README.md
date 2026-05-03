# Tech Challenge - Telco Churn Prediction

Este projeto implementa uma solucao de Machine Learning Engineering para prever churn de clientes de telecomunicacoes.

## Linguagens e Tecnologias

| Categoria | Stack |
| --- | --- |
| Linguagem principal | Python |
| API | FastAPI |
| Machine Learning | scikit-learn, XGBoost, MLP |
| Experiment tracking | MLflow |
| Testes | pytest |
| Containers | Docker, Docker Compose |
| Banco para tracking | PostgreSQL |

## Problema de Negocio

Empresas de telecomunicacoes sofrem com churn (cancelamento de clientes), o que impacta receita recorrente e aumenta o custo de aquisicao de novos clientes.
O desafio de negocio deste projeto e identificar clientes com maior risco de cancelamento para priorizar acoes de retencao mais eficientes.

## Dados

Os dados utilizados neste projeto foram retirados do Kaggle, no dataset **Telco Customer Churn**:

- Kaggle: [blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Referencia do dicionario de dados: [docs/DICIONARIO_DADOS.md](docs/DICIONARIO_DADOS.md)

## Objetivo do Projeto

Construir uma solucao fim a fim para prever churn e apoiar decisoes de negocio orientadas por dados.

## Resumo do Modelo e Impacto Financeiro

Modelo recomendado no experimento controlado: **LogisticRegression-balanced**.

Metricas do modelo recomendado (threshold padrao 0.5):

- ROC-AUC: `0.8482`
- PR-AUC: `0.6444`
- F1-score: `0.6164`
- Recall: `0.7754`
- Precision: `0.5115`

Resultado de negocio no threshold otimizado:

- Threshold recomendado: `0.15`
- Net Benefit maximo: **$706,000**
- Recall no threshold recomendado: **98.40%**
- Precision no threshold recomendado: **38.02%**
- Confusao no threshold recomendado: `TP=368`, `FP=600`, `FN=6`

Premissas de custo usadas no notebook `02_experimento_controlado.ipynb`:

- `cost_fp = $50`
- `cost_fn = $2000`

Top 3 modelos por Net Benefit no notebook:

1. LogisticRegression-balanced: **$706,000** (threshold 0.15)
2. LogisticRegression-SMOTE: **$705,500** (threshold 0.15)
3. MLPWrapper-PyTorch: **$688,750** (threshold 0.10)

## Entregaveis

| Entregavel | Local | Descricao |
| --- | --- | --- |
| ML Canvas | [docs/ML_CANVAS.md](docs/ML_CANVAS.md) | Contexto de negocio, stakeholders, proposta de valor e metricas. |
| EDA | [notebooks/01_eda_and_ml_canvas.ipynb](notebooks/01_eda_and_ml_canvas.ipynb), [docs/RELATORIO_EDA.md](docs/RELATORIO_EDA.md) | Analise exploratoria, qualidade dos dados e principais insights de churn. |
| Dataset processado | [data/processed/telco_churn_processed.csv](data/processed/telco_churn_processed.csv) | Base limpa para treino dos modelos. |
| Experimento controlado | [notebooks/02_experimento_controlado.ipynb](notebooks/02_experimento_controlado.ipynb) | Comparacao entre baselines, Logistic Regression, Random Forest, XGBoost e MLP. |
| Modelo treinado | [models/](models/) | Modelos treinados (*Também foram adicionados ao MLFlow). |
| Model Card | [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | Descricao tecnica do modelo, metricas, limitacoes e vieses. |
| API de inferencia | [src/api/](src/api/) | FastAPI com predicao individual, predicao em lote, health check e model info. |
| Testes | [tests/](tests/) | Testes automatizados da API, dados, metricas e treinamento. |
| Docker | [docker-compose.yml](docker-compose.yml), [Dockerfile.api](Dockerfile.api), [Dockerfile.training](Dockerfile.training), [Dockerfile.mlflow](Dockerfile.mlflow) | Orquestracao da API, treinamento, MLflow e PostgreSQL. |
| Documentacao operacional | [docs/ARQUITETURA_DEPLOY.md](docs/ARQUITETURA_DEPLOY.md), [docs/PLANO_MONITORAMENTO.md](docs/PLANO_MONITORAMENTO.md), [docs/TERRAFORM_AWS_PLAN.md](docs/TERRAFORM_AWS_PLAN.md) | Deploy, monitoramento e plano de infraestrutura. |
| Video STAR | A definir | Placeholder para o link do video de apresentacao no formato STAR. |
| Deploy em AWS | A definir | Placeholder para a URL do deploy em nuvem quando o ambiente estiver publicado. |

## Arquitetura do Projeto

```mermaid
flowchart LR
    A["Dados brutos<br/>Telco_customer_churn.xlsx"] --> B["EDA e limpeza<br/>notebooks"]
    B --> C["Dataset processado<br/>data/processed"]
    C --> D["Treinamento e experimentos<br/>src/models"]
    D --> E["MLflow<br/>metricas e artefatos"]
    E --> F["Modelo versionado"]
    F --> G["FastAPI<br/>src/api"]
    G --> H["Predicoes<br/>/api/predict"]
    G --> I["Predicoes em lote<br/>/api/predict-batch"]
```

## Arquitetura Docker

```mermaid
flowchart TB
    USER["Usuario / Cliente HTTP"] --> API["FastAPI API<br/>:8000"]
    USER --> MLFLOWUI["MLflow UI<br/>:5000"]

    subgraph Compose["docker-compose / mlflow_network"]
        DB["PostgreSQL<br/>service: db<br/>:5432"]
        MLFLOW["MLflow Server<br/>service: mlflow<br/>:5000"]
        API["FastAPI Inference<br/>service: api<br/>:8000"]
        TRAIN["Training Job (on-demand)<br/>service: training"]
        ART["Volume bind<br/>./mlartifacts:/mlartifacts"]
    end

    DB --> MLFLOW
    TRAIN --> MLFLOW
    TRAIN --> ART
    API --> MLFLOW
    MLFLOW --> ART
```

## Fluxo de Predicao

```mermaid
sequenceDiagram
    participant User as Usuario
    participant API as FastAPI
    participant Manager as ModelManager
    participant MLflow as MLflow
    participant Registry as Model Registry
    participant Model as Pipeline

    User->>API: POST /api/predict
    API->>Manager: valida payload e monta DataFrame
    Manager->>MLflow: GET /health
    MLflow-->>Manager: status ok
    Manager->>Registry: load models:/TelcoChurnPipeline@champion
    Registry-->>Manager: pipeline sklearn
    Manager->>Model: predict / predict_proba
    Model-->>API: classe, probabilidade e confianca
    API-->>User: resposta JSON
```

## Estrutura

```text
tech_challenge/
  data/          dados brutos e processados
  docs/          documentacao de negocio, modelo, deploy e monitoramento
  notebooks/     EDA, ML Canvas e experimentos
  src/api/       API FastAPI
  src/data/      carga e preparacao de dados
  src/evaluation/ metricas
  src/models/    baselines, treinamento e artefatos
  tests/         testes automatizados
```

## API

| Metodo | Endpoint | Uso |
| --- | --- | --- |
| GET | `/api/health` | Verifica saude da API e carregamento do modelo. |
| POST | `/api/predict` | Predicao para um cliente. |
| POST | `/api/predict-batch` | Predicoes em lote. |
| GET | `/api/model-info` | Informacoes do modelo carregado. |
| POST | `/api/schedule-update` | Agenda atualizacao do modelo. |
| GET | `/api/docs` | Swagger UI. |

## Como Executar

```bash
docker-compose up -d
docker-compose run --rm training
pytest -q
```

Servicos locais:

- [API Swagger](http://localhost:8000/api/docs)
- [Health check](http://localhost:8000/api/health)
- [MLflow](http://localhost:5000)
- PostgreSQL: `localhost:5432`

## Entrega Final

| Item | Status | Link |
| --- | --- | --- |
| Video STAR | Pendente | TODO: adicionar link do video STAR |
| Deploy em AWS | Pendente | TODO: adicionar URL publica da API na AWS |

Atalhos uteis:

```bash
make docker-compose-up
make docker-train
make test-cov
make mlflow-ui
```

## Documentacao Complementar

- [docs/RELATORIO_EDA.md](docs/RELATORIO_EDA.md): relatorio da exploracao de dados.
- [docs/MODEL_CARD.md](docs/MODEL_CARD.md): detalhes do modelo selecionado.
- [docs/DICIONARIO_DADOS.md](docs/DICIONARIO_DADOS.md): descricao das variaveis.
- [docs/ARQUITETURA_DEPLOY.md](docs/ARQUITETURA_DEPLOY.md): arquitetura proposta de deploy.
- [docs/PLANO_MONITORAMENTO.md](docs/PLANO_MONITORAMENTO.md): monitoramento e alertas.
- [docs/TERRAFORM_AWS_PLAN.md](docs/TERRAFORM_AWS_PLAN.md): plano de infraestrutura AWS.
- [DOCKER_GUIA_EXECUCAO.md](DOCKER_GUIA_EXECUCAO.md): guia para execucao com Docker.

## Nota sobre MLflow e versionamento

Os experimentos foram salvos no MLflow durante a execucao dos estudos e comparacoes.
Os artefatos de experimento nao foram comitados no Git por falta de necessidade funcional e para manter a limpeza do repositorio.
