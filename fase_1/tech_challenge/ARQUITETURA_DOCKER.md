# 🏗️ Arquitetura Docker Implementada

## 📊 Diagrama da Solução

```
┌─────────────────────────────────────────────────────────────────┐
│                      Docker Network                            │
│                    (mlflow_network)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────┐     ┌──────────────────────┐         │
│  │   mlflow_db          │     │    mlflow_server     │         │
│  │   (PostgreSQL)       │────▶│   (MLflow UI)        │         │
│  │                      │     │   Port 5000          │         │
│  │  • Port 5432         │     │                      │         │
│  │  • username: mlflow  │     │  Backend:            │         │
│  │  • password: mlflow  │     │  PostgreSQL          │         │
│  │  • DB: mlflow_db     │     │  Artifacts: /mlflow  │         │
│  └──────────────────────┘     └──────────────────────┘         │
│         △                             △  │                      │
│         │                             │  │                      │
│         │                    ┌────────┘  │                      │
│         │                    │           │                      │
│         │                    ▼           ▼                      │
│         │          ┌──────────────────────────┐                │
│         │          │   fastapi_inference      │                │
│         │          │   (FastAPI API)          │                │
│         │          │   Port 8000              │                │
│         └─────────▶│                          │                │
│                    │ • Loads model from MLflow│                │
│                    │ • Serves predictions     │                │
│                    │ • Health: /api/health    │                │
│                    └──────────────────────────┘                │
│                             ▲                                   │
│                             │                                   │
│                    (MLFLOW_TRACKING_URI)                       │
│                   http://mlflow:5000                           │
│                             │                                   │
│         ┌───────────────────┘                                  │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────┐                                      │
│  │   ml_training_service│                                      │
│  │   (Training Job)     │                                      │
│  │   (On-demand)        │                                      │
│  │                      │                                      │
│  │ • Runs:              │                                      │
│  │   python src/models/ │                                      │
│  │   base_pipeline.py   │                                      │
│  │ • Logs artefatos     │                                      │
│  │ • Registra modelo    │                                      │
│  └──────────────────────┘                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

External Access:
┌─────────────────────────────────────────────────────────────────┐
│                      Localhost (Host)                           │
├─────────────────────────────────────────────────────────────────┤
│  🌐 http://localhost:8000      →  fastapi_inference            │
│  📊 http://localhost:5000      →  mlflow_server                 │
│  🐘 localhost:5432             →  mlflow_db                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Estrutura de Arquivos Modificada/Criada

```
tech_challenge/
├── docker-compose.yml              ✏️  MODIFICADO
│   ├─ Removido version obsoleto
│   ├─ API usa Dockerfile.api (novo)
│   └─ NOVO: serviço training
│
├── Dockerfile.api                  ✨ NOVO
│   ├─ Base: python:3.12-slim
│   ├─ Pip install (não uv)
│   ├─ Expõe: 8000
│   └─ Health check via curl
│
├── Dockerfile.training             ✨ NOVO
│   ├─ Base: python:3.12-slim
│   ├─ Pip install (não uv)
│   ├─ Executa: base_pipeline.py
│   └─ Volumes: /mlartifacts, /models
│
├── Dockerfile                      ⚠️  OBSOLETO
│   └─ Não usado mais (era com uv)
│
├── Dockerfile.mlflow               
│   └─ Não modificado (funciona)
│
├── Makefile                        ✏️  MODIFICADO
│   ├─ docker-train
│   ├─ docker-train-baseline
│   ├─ docker-train-raw
│   ├─ docker-train-shell
│   └─ docker-train-logs
│
│   └─ Resumo desta implementação
│
└── src/
    ├── models/
    │   ├── base_pipeline.py        ✅ JÁ EXISTIA
    │   ├── model_traning.py        ✅ MANTIDO
    │   └── baseline.py             ✅ MANTIDO
    │
    └── api/
        └── main.py                 ✅ PRONTO para MLflow
```

---

## 🔄 Fluxo: Do Treino ao Deploy

### 1️⃣ **Treino (Training Container)**

```bash
docker-compose run --rm training
# ↓ Executa: python src/models/base_pipeline.py
# ↓ Conecta a: MLFLOW_TRACKING_URI=http://mlflow:5000
# ↓ Faz login:
#   - Cria experiment: Telco_Churn_Production
#   - Log params: max_iter, class_weight, random_state
#   - Log metrics: AUC, Precision, Recall, F1, PR-AUC
#   - Log artifacts: gráficos, pkl model
#   - Registra: TelcoChurnLogisticRegression v1
```

### 2️⃣ **Monitoramento (MLflow UI)**

```
http://localhost:5000
├─ Experiments
│  └─ Telco_Churn_Production
│     └─ Run #1
│        ├─ Params
│        ├─ Metrics
│        └─ Artifacts
│
└─ Model Registry
   └─ TelcoChurnLogisticRegression
      └─ Version 1
         ├─ Staging
         └─ Production ← clique aqui
```

### 3️⃣ **Promoção (Mudança de Stage)**

```
MLflow UI:
1. Model Registry → TelcoChurnLogisticRegression
2. Version 1 → Change Stage → Production

Resultado:
- API carrega automaticamente na próxima requisição
- Nenhum redeployment necessário
```

### 4️⃣ **Predição (FastAPI)**

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "monthly_charges": 75.5,
    ...
  }'

# ↓ API carrega modelo de:
#   models://TelcoChurnLogisticRegression/Production
# ↓ Retorna:
#   {"churn_probability": 0.35, "churn_risk": "low"}
```

---

## 🚀 Comandos Principais

### Setup Inicial
```bash
docker-compose up -d           # Inicia db + mlflow + api
docker-compose ps              # Verifica status
docker-compose logs -f api     # Logs da API
```

### Treinamento
```bash
docker-compose run --rm training                      # Treina com MLflow
docker-compose run --rm training python src/models/baseline.py  # Baseline
docker-compose run --rm training bash                 # Shell interativo
```

### Monitoramento
```bash
http://localhost:5000          # MLflow UI
http://localhost:8000/docs     # Swagger API
docker-compose logs training   # Logs do treino
```

### Shutdown
```bash
docker-compose down            # Para tudo
docker-compose down -v         # Para + remove volumes (limpa tudo)
```

---

**Arquitetura finalizada:** 27/04/2026 ✅
