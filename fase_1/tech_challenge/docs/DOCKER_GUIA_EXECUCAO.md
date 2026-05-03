# 🐳 Guia de Execução com Docker Compose

Documentação completa para rodar a aplicação **Telco Churn Prediction** com Docker Compose.

---

## 📋 Requisitos

### Sistema Operacional
- **Windows 10/11**, **macOS**, ou **Linux** com Docker instalado
- Docker Desktop (recomendado para Windows/macOS) ou Docker Engine

### Dependências Instaladas
- **Docker**: v20.10+
- **Docker Compose**: v2.0+
- **Git** (opcional, para clonar repositório)

### Verificar Instalação
```bash
docker --version
docker-compose --version
# ou
docker compose version
```

### Espaço em Disco
- Mínimo **5GB** de espaço livre
- Recomendado **10GB** para armazenar dados, modelos e artifacts do MLflow

### Portas Disponíveis
Certifique-se de que as seguintes portas estão livres:
- **5000** - MLflow Server
- **5432** - PostgreSQL Database
- **8000** - FastAPI REST API

---

## 🚀 Como Iniciar Todos os Containers

### Opção 1: Iniciar Todos os Serviços (Recomendado)

```bash
# Navegar até o diretório do projeto
cd /path/to/tech_challenge

# Iniciar todos os containers (db, mlflow, api, training)
docker-compose up
```

**Saída esperada:**
```
Creating mlflow_db ... done
Creating mlflow_server ... done
Creating fastapi_inference ... done
Creating ml_training_service ... done
```

### Opção 2: Iniciar em Background (Detached Mode)

```bash
docker-compose up -d
```

Para visualizar logs:
```bash
docker-compose logs -f
```

Para logs de um serviço específico:
```bash
docker-compose logs -f api
docker-compose logs -f mlflow
docker-compose logs -f training
```

### Opção 3: Reconstruir Imagens (First Time / Após Mudanças)

```bash
# Forçar rebuild das imagens
docker-compose up --build

# Ou
docker-compose up --build -d
```

### Verificar Status dos Containers

```bash
# Ver status de todos os containers
docker-compose ps

# Exemplo de saída:
# NAME                  STATUS          PORTS
# mlflow_db             Up 2 minutes     5432/tcp
# mlflow_server         Up 1 minute      5000/tcp
# fastapi_inference     Up 1 minute      8000/tcp
# ml_training_service   Created
```

### Parar/Remover Containers

```bash
# Parar containers (mantém volumes)
docker-compose stop

# Remover containers (mantém volumes e dados)
docker-compose down

# Remover TUDO (containers, volumes, networks)
docker-compose down -v
```

---

## 🤖 Treinar Modelo via Docker Compose

### Executar Treinamento Uma Única Vez

```bash
# Executar o container de treinamento
docker-compose run training

# Ou explicitamente:
docker-compose run training python src/models/model_training_pipeline.py
```

**Saída esperada:**
```
Loading training data...
Training model...
Evaluating model...
Registering model to MLflow...
✓ Model registered as 'TelcoChurnPipeline' (version 1)
```

### Opções de Execução

```bash
# Treinar com output interativo
docker-compose run -it training

# Treinar mantendo containers rodando (detached)
docker-compose run -d training

# Treinar com variáveis de ambiente customizadas
docker-compose run -e EXPERIMENT_NAME=custom_exp training
```

### Verificar Arquivos Gerados

Os modelos treinados serão registrados no **MLflow Model Registry**.

Para verificar:
1. Acesse MLflow UI em: http://localhost:5000
2. Navegue até **Models** → **TelcoChurnPipeline**
3. Verifique as versões registradas

---

## 📊 MLflow - Acessar Interface e APIs

### URL Principal - MLflow UI

```
http://localhost:5000
```

**O que você encontrará:**
- Experiments: Experimentos de treinamento
- Runs: Execuções individuais com métricas e parâmetros
- Models: Modelos registrados com versionamento
- Artifacts: Artefatos (modelos, dados, plots)

### APIs MLflow Úteis

#### 1. Listar Modelos Registrados
```bash
curl -X GET http://localhost:5000/api/2.0/mlflow/registered-models/list
```

#### 2. Obter Versão Mais Recente de um Modelo
```bash
curl -X GET "http://localhost:5000/api/2.0/mlflow/model-versions/search?filter=name='TelcoChurnPipeline' and stage='production'"
```

#### 3. Obter Informações de um Experimento
```bash
curl -X GET http://localhost:5000/api/2.0/mlflow/experiments/search
```

#### 4. Listar Runs de um Experimento
```bash
# Substitua {experiment_id} pelo ID do experimento
curl -X GET "http://localhost:5000/api/2.0/mlflow/runs/search" \
  -H "Content-Type: application/json" \
  -d '{"experiment_ids": ["{experiment_id}"]}'
```

### URL de Download de Artefatos

```
http://localhost:5000/api/2.0/mlflow/artifacts/get?path=MODEL_PATH&run_id=RUN_ID
```

---

## 🔗 REST API - Cenários de Uso

### Base URL
```
http://localhost:8000
```

### Documentação Interativa
- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

---

## 1️⃣ Health Check

**Verificar se API está funcionando:**

```bash
curl -X GET http://localhost:8000/api/health
```

**Resposta:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "model_loaded": true
}
```

**Ou (sem prefixo /api):**
```bash
curl -X GET http://localhost:8000/health
```

---

## 2️⃣ Predição Simples (Single Prediction)

**Endpoint:**
```
POST /api/predict
```

**Request (JSON):**
```json
{
  "features": {
    "gender": "Male",
    "senior_citizen": "No",
    "partner": "Yes",
    "dependents": "No",
    "tenure_months": 24,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 65.5,
    "total_charges": 1575.00
  },
  "return_probability": true
}
```

**cURL:**
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "gender": "Male",
      "senior_citizen": "No",
      "partner": "Yes",
      "dependents": "No",
      "tenure_months": 24,
      "phone_service": "Yes",
      "multiple_lines": "No",
      "internet_service": "Fiber optic",
      "online_security": "No",
      "online_backup": "No",
      "device_protection": "No",
      "tech_support": "No",
      "streaming_tv": "No",
      "streaming_movies": "No",
      "contract": "Month-to-month",
      "paperless_billing": "Yes",
      "payment_method": "Electronic check",
      "monthly_charges": 65.5,
      "total_charges": 1575.00
    },
    "return_probability": true
  }'
```

**Resposta:**
```json
{
  "prediction": 1,
  "probability": 0.7234,
  "confidence": 0.7234,
  "processing_time_ms": 12.345
}
```

---

## 3️⃣ Predição em Lote (Batch Prediction)

**Endpoint:**
```
POST /api/predict-batch
```

**Request (JSON com múltiplos clientes):**
```json
{
  "samples": [
    {
      "gender": "Male",
      "senior_citizen": "No",
      "partner": "Yes",
      "dependents": "No",
      "tenure_months": 24,
      "phone_service": "Yes",
      "multiple_lines": "No",
      "internet_service": "Fiber optic",
      "online_security": "No",
      "online_backup": "No",
      "device_protection": "No",
      "tech_support": "No",
      "streaming_tv": "No",
      "streaming_movies": "No",
      "contract": "Month-to-month",
      "paperless_billing": "Yes",
      "payment_method": "Electronic check",
      "monthly_charges": 65.5,
      "total_charges": 1575.00
    },
    {
      "gender": "Female",
      "senior_citizen": "Yes",
      "partner": "No",
      "dependents": "Yes",
      "tenure_months": 12,
      "phone_service": "No",
      "multiple_lines": "No",
      "internet_service": "DSL",
      "online_security": "Yes",
      "online_backup": "Yes",
      "device_protection": "Yes",
      "tech_support": "Yes",
      "streaming_tv": "Yes",
      "streaming_movies": "Yes",
      "contract": "Two year",
      "paperless_billing": "No",
      "payment_method": "Bank transfer",
      "monthly_charges": 85.0,
      "total_charges": 1020.00
    }
  ],
  "return_probabilities": true
}
```

**cURL:**
```bash
curl -X POST http://localhost:8000/api/predict-batch \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {
        "gender": "Male",
        "senior_citizen": "No",
        "partner": "Yes",
        "dependents": "No",
        "tenure_months": 24,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 65.5,
        "total_charges": 1575.00
      },
      {
        "gender": "Female",
        "senior_citizen": "Yes",
        "partner": "No",
        "dependents": "Yes",
        "tenure_months": 12,
        "phone_service": "No",
        "multiple_lines": "No",
        "internet_service": "DSL",
        "online_security": "Yes",
        "online_backup": "Yes",
        "device_protection": "Yes",
        "tech_support": "Yes",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "contract": "Two year",
        "paperless_billing": "No",
        "payment_method": "Bank transfer",
        "monthly_charges": 85.0,
        "total_charges": 1020.00
      }
    ],
    "return_probabilities": true
  }'
```

**Resposta:**
```json
{
  "predictions": [1, 0],
  "probabilities": [0.7234, 0.2145],
  "batch_size": 2,
  "processing_time_ms": 18.756
}
```

---

## 4️⃣ Obter Informações do Modelo

**Endpoint:**
```
GET /api/model-info
```

**cURL:**
```bash
curl -X GET http://localhost:8000/api/model-info
```

**Resposta:**
```json
{
  "model_type": "LogisticRegression (com ColumnTransformer)",
  "model_version": "production",
  "n_features": 19,
  "features_used": [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "tenure_months",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
    "monthly_charges",
    "total_charges"
  ]
}
```

---

## 5️⃣ Agendar Atualização de Modelo (Hot-Swap)

**Endpoint:**
```
POST /api/schedule-update
```

**Descrição:** Agenda a atualização automática do modelo para uma data/hora específica.

**Request (ISO 8601 DateTime):**
```json
{
  "target_datetime": "2026-04-30T14:30:00Z"
}
```

**cURL:**
```bash
curl -X POST http://localhost:8000/api/schedule-update \
  -H "Content-Type: application/json" \
  -d '{
    "target_datetime": "2026-04-30T14:30:00Z"
  }'
```

**Resposta (202 Accepted):**
```json
{
  "status": "accepted",
  "message": "Atualização do modelo agendada para 2026-04-30T14:30:00+00:00"
}
```

---

## 6️⃣ Root Endpoint

**Endpoint:**
```
GET /
```

**cURL:**
```bash
curl -X GET http://localhost:8000/
```

**Resposta:**
```json
{
  "message": "Telco Churn Prediction API",
  "version": "0.1.0",
  "docs": "/api/docs",
  "health": "/health"
}
```

---

## 🧪 Testes Completos (Script Bash)

Crie um arquivo `test_api.sh`:

```bash
#!/bin/bash

# Cores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

API_URL="http://localhost:8000"
MLFLOW_URL="http://localhost:5000"

echo -e "${BLUE}=== Testing Telco Churn Prediction API ===${NC}\n"

# 1. Health Check
echo -e "${BLUE}1. Testing Health Check...${NC}"
curl -X GET $API_URL/api/health
echo -e "\n${GREEN}✓ Health Check OK${NC}\n"

# 2. Root Endpoint
echo -e "${BLUE}2. Testing Root Endpoint...${NC}"
curl -X GET $API_URL/
echo -e "\n${GREEN}✓ Root Endpoint OK${NC}\n"

# 3. Model Info
echo -e "${BLUE}3. Testing Model Info...${NC}"
curl -X GET $API_URL/api/model-info
echo -e "\n${GREEN}✓ Model Info OK${NC}\n"

# 4. Single Prediction
echo -e "${BLUE}4. Testing Single Prediction...${NC}"
curl -X POST $API_URL/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "gender": "Male",
      "senior_citizen": "No",
      "partner": "Yes",
      "dependents": "No",
      "tenure_months": 24,
      "phone_service": "Yes",
      "multiple_lines": "No",
      "internet_service": "Fiber optic",
      "online_security": "No",
      "online_backup": "No",
      "device_protection": "No",
      "tech_support": "No",
      "streaming_tv": "No",
      "streaming_movies": "No",
      "contract": "Month-to-month",
      "paperless_billing": "Yes",
      "payment_method": "Electronic check",
      "monthly_charges": 65.5,
      "total_charges": 1575.00
    },
    "return_probability": true
  }'
echo -e "\n${GREEN}✓ Single Prediction OK${NC}\n"

# 5. Batch Prediction
echo -e "${BLUE}5. Testing Batch Prediction...${NC}"
curl -X POST $API_URL/api/predict-batch \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {
        "gender": "Male",
        "senior_citizen": "No",
        "partner": "Yes",
        "dependents": "No",
        "tenure_months": 24,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 65.5,
        "total_charges": 1575.00
      }
    ],
    "return_probabilities": true
  }'
echo -e "\n${GREEN}✓ Batch Prediction OK${NC}\n"

# 6. MLflow Health
echo -e "${BLUE}6. Testing MLflow Server...${NC}"
curl -X GET $MLFLOW_URL/health
echo -e "\n${GREEN}✓ MLflow OK${NC}\n"

echo -e "${GREEN}=== All Tests Completed ===${NC}"
```

**Executar:**
```bash
bash test_api.sh
```

---

## 🛠️ Troubleshooting

### Problema: Porta já em uso
```bash
# Verificar processo usando porta 8000
lsof -i :8000  # Linux/macOS
netstat -ano | findstr :8000  # Windows

# Liberar porta (alternativa: mudar porta em docker-compose.yml)
kill -9 <PID>  # Linux/macOS
taskkill /PID <PID> /F  # Windows
```

### Problema: MLflow não consegue conectar ao PostgreSQL
```bash
# Verificar logs
docker-compose logs db
docker-compose logs mlflow

# Verificar status do banco
docker-compose exec db pg_isready -U mlflow_user -d mlflow_db
```

### Problema: API não consegue conectar ao MLflow
```bash
# Verificar se MLflow está saudável
docker-compose exec mlflow python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

# Ver logs da API
docker-compose logs api
```

### Problema: Modelo não carrega na API
```bash
# 1. Verificar se modelo foi registrado no MLflow
curl http://localhost:5000/api/2.0/mlflow/registered-models/list

# 2. Verificar se modelo está em stage 'production'
curl http://localhost:5000/api/2.0/mlflow/model-versions/search \
  -H "Content-Type: application/json" \
  -d '{"filter": "name='\''TelcoChurnPipeline'\'' and stage='\''production'\''"}'

# 3. Treinar modelo primeiro
docker-compose run training
```

---

## 📱 Exemplo de Integração Python

```python
import requests
import json

API_URL = "http://localhost:8000"

# Cliente para fazer predições
class ChurnPredictor:
    def __init__(self, api_url):
        self.api_url = api_url

    def predict_single(self, features_dict):
        """Fazer predição simples"""
        payload = {
            "features": features_dict,
            "return_probability": True
        }
        response = requests.post(
            f"{self.api_url}/api/predict",
            json=payload
        )
        return response.json()

    def predict_batch(self, features_list):
        """Fazer predição em lote"""
        payload = {
            "samples": features_list,
            "return_probabilities": True
        }
        response = requests.post(
            f"{self.api_url}/api/predict-batch",
            json=payload
        )
        return response.json()

    def get_model_info(self):
        """Obter informações do modelo"""
        response = requests.get(f"{self.api_url}/api/model-info")
        return response.json()

# Uso
predictor = ChurnPredictor(API_URL)

# Fazer predição
customer = {
    "gender": "Male",
    "senior_citizen": "No",
    "partner": "Yes",
    "dependents": "No",
    "tenure_months": 24,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 65.5,
    "total_charges": 1575.00
}

result = predictor.predict_single(customer)
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']}")
```

---

## 📚 Estrutura de Containers

```
┌─────────────────────────────────────┐
│       Docker Compose Network        │
│      (mlflow_network bridge)        │
└─────────────────────────────────────┘
           │          │          │
      ┌────▼──┐  ┌────▼──┐  ┌──▼────┐
      │  DB   │  │MLflow │  │  API  │
      │ :5432 │  │ :5000 │  │ :8000 │
      └────┬──┘  └────┬──┘  └──┬────┘
           │          │        │
    PostgreSQL    MLflow   FastAPI
      (Volumes:         (Endpoints:
       postgres_data)   /api/predict
                        /api/predict-batch
                        /api/model-info
                        /api/health)
           │
      Training Container
      (docker-compose run training)
      └─→ Registra modelos no MLflow
      └─→ Armazena artifacts em mlartifacts/
```

---

## 📝 Resumo de Comandos Principais

```bash
# Iniciar
docker-compose up -d

# Ver status
docker-compose ps

# Ver logs
docker-compose logs -f

# Treinar modelo
docker-compose run training

# Parar
docker-compose stop

# Limpar
docker-compose down -v

# Testar API
curl http://localhost:8000/api/health
curl http://localhost:5000/health

# Acessar MLflow
# http://localhost:5000

# Documentação interativa
# http://localhost:8000/api/docs
```

---

## 🎯 Próximas Etapas

1. **Treinar Modelo:** `docker-compose run training`
2. **Verificar MLflow:** http://localhost:5000
3. **Testar API:** `curl http://localhost:8000/api/health`
4. **Fazer Predições:** Use os exemplos acima com cURL ou Python
5. **Monitorar:** Verifique logs e métricas no MLflow

---

**Última atualização:** Abril 2026
**Versão da Documentação:** 1.0
**Autor:** MLOps Team
