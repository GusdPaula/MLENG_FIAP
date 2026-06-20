# Documentação de Testes da API

Este documento fornece comandos cURL para testar todos os endpoints da API de recomendação.

## Como Executar a API

### Com Poetry (Recomendado)

```bash
# Definir a API key
export API_KEY="default-api-key-change-in-production"

# Definir o caminho do modelo (opcional, usa gmf_binary por padrão)
export MODEL_PATH="ecommerce_recommender/models/mlflow_experiments/gmf_binary.pt"

# Executar a API com Poetry (necessário PYTHONPATH)
# PYTHONPATH precisa incluir src (para recommender) e o diretório raiz (para api)
PYTHONPATH=ecommerce_recommender/src:ecommerce_recommender poetry run python -m api.main

# Ou com uvicorn
PYTHONPATH=ecommerce_recommender/src:ecommerce_recommender poetry run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Gerar Chave Segura com Poetry

```bash
# Gerar chave segura
export API_KEY=$(poetry run python -c 'import secrets; print(secrets.token_hex(32))')
```

### Sem Poetry

```bash
# Definir a API key
export API_KEY="default-api-key-change-in-production"

# Navegue para o diretório ecommerce_recommender primeiro
cd ecommerce_recommender

# Executar a API
PYTHONPATH=src python3 -m api.main
```

## Configuração

### Definir a API Key

```bash
export API_KEY="default-api-key-change-in-production"
```

Ou inclua diretamente nos comandos:

```bash
curl -H "X-API-Key: default-api-key-change-in-production" ...
```

## Endpoints

### 1. Health Check

Verifica se o serviço está saudável.

```bash
curl -X GET "http://localhost:8000/health" \
  -H "X-API-Key: "default-api-key-change-in-production"
```

**Resposta esperada:**
```json
{
  "status": "healthy",
  "service": "prediction_api"
}
```

### 2. Informações do Modelo

Obtém informações sobre o modelo carregado.

```bash
curl -X GET "http://localhost:8000/model/info" \
  -H "X-API-Key: default-api-key-change-in-production"
```

**Resposta esperada:**
```json
{
  "model_path": "models/model.pt",
  "predictor_type": "single_user",
  "device": "cpu",
  "metadata": {
    "model_type": "ncf",
    "num_users": 1000,
    "num_items": 5000
  }
}
```

### 3. Predição Única

Gera predições para um único usuário contra itens específicos.

**Nota:** Os IDs de usuários e itens são específicos do modelo carregado. Os exemplos abaixo usam IDs do modelo `gmf_binary`. Se estiver usando `ncf_weighted`, use IDs como: usuário `794181`, itens `[439202, 388242, 43485]`.

```bash
# Para modelo gmf_binary
curl -X POST "http://localhost:8000/predict" \
  -H "X-API-Key: default-api-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 138131,
    "item_ids": [430292, 277119, 183411, 457231, 259078]
  }'

# Para modelo ncf_weighted
curl -X POST "http://localhost:8000/predict" \
  -H "X-API-Key: default-api-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 794181,
    "item_ids": [439202, 388242, 43485, 422768, 140848]
  }'
```

**Resposta esperada:**
```json
{
  "user_id": 123,
  "item_scores": {
    "1": 0.95,
    "2": 0.87,
    "3": 0.72,
    "4": 0.65,
    "5": 0.58
  },
  "metadata": {
    "predictor": "single_user"
  }
}
```

### 4. Predição com Top-K

Gera recomendações top-k para um usuário.

```bash
curl -X GET "http://localhost:8000/recommend/138131?k=10" \
  -H "X-API-Key: default-api-key-change-in-production"
```

**Resposta esperada:**
```json
{
  "user_id": 123,
  "recommendations": [
    [1, 0.95],
    [2, 0.87],
    [3, 0.72],
    [4, 0.65],
    [5, 0.58]
  ],
  "metadata": {
    "predictor": "top_k",
    "k": 10
  }
}
```

### 5. Predição em Lote

Gera predições para múltiplos usuários.

```bash
# Para modelo gmf_binary
curl -X POST "http://localhost:8000/predict/batch" \
  -H "X-API-Key: default-api-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{
    "user_item_pairs": [
      [138131, [430292, 277119, 183411]],
      [911093, [457231, 259078, 183087]]
    ],
    "k": null
  }'

# Para modelo ncf_weighted
curl -X POST "http://localhost:8000/predict/batch" \
  -H "X-API-Key: default-api-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{
    "user_item_pairs": [
      [794181, [439202, 388242, 43485]],
      [1153198, [422768, 140848]]
    ],
    "k": null
  }'
```

**Resposta esperada:**
```json
{
  "predictions": [
    {
      "user_id": 123,
      "item_scores": {
        "1": 0.95,
        "2": 0.87,
        "3": 0.72
      },
      "metadata": {}
    },
    {
      "user_id": 456,
      "item_scores": {
        "4": 0.91,
        "5": 0.83,
        "6": 0.76
      },
      "metadata": {}
    }
  ],
  "metadata": {
    "model_type": "ncf",
    "predictor_type": "batch",
    "num_requests": 2
  }
}
```

### 6. Definir Baselines de Monitoramento

Define os baselines de monitoramento após coletar dados iniciais de predição.

```bash
curl -X POST "http://localhost:8000/monitoring/baselines" \
  -H "X-API-Key: default-api-key-change-in-production"
```

**Resposta esperada:**
```json
{
  "status": "baselines set successfully"
}
```

### 7. Verificar Desvios

Verifica se há desvios de modelo ou dados.

```bash
curl -X GET "http://localhost:8000/monitoring/check" \
  -H "X-API-Key: default-api-key-change-in-production"
```

**Resposta esperada:**
```json
{
  "data_shift": {
    "has_shift": false,
    "shift_type": "data_shift",
    "p_value": 0.8234,
    "test_statistic": 0.1234,
    "threshold": 0.05,
    "message": "Data shift not detected: KS statistic=0.1234, p-value=0.8234, threshold=0.05"
  },
  "performance_drift": {
    "has_shift": false,
    "shift_type": "model_drift",
    "p_value": 0.4567,
    "test_statistic": 0.7890,
    "threshold": 2.0,
    "message": "Performance drift not detected: z-score=0.7890, threshold=2.0"
  }
}
```

### 8. Resumo de Monitoramento

Obtém o resumo do status de monitoramento.

```bash
curl -X GET "http://localhost:8000/monitoring/summary" \
  -H "X-API-Key: default-api-key-change-in-production"
```

**Resposta esperada:**
```json
{
  "performance_stats": {
    "mean": 0.75,
    "std": 0.12,
    "min": 0.45,
    "max": 0.98,
    "count": 500
  },
  "has_baseline": true,
  "window_size": 1000,
  "history_size": 500,
  "metrics_history_size": 10
}
```

## Cenários de Teste

### Cenário 1: Fluxo Completo de Predição

1. Verificar saúde do serviço
2. Obter informações do modelo
3. Fazer predição única
4. Obter recomendações top-k
5. Verificar desvios

```bash
# 1. Health check
curl -X GET "http://localhost:8000/health" \
  -H "X-API-Key: default-api-key-change-in-production"

# 2. Model info
curl -X GET "http://localhost:8000/model/info" \
  -H "X-API-Key: default-api-key-change-in-production"

# 3. Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "X-API-Key: default-api-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 138131, "item_ids": [430292, 277119, 183411, 457231, 259078]}'

# 4. Top-k recommendations
curl -X GET "http://localhost:8000/recommend/138131?k=10" \
  -H "X-API-Key: default-api-key-change-in-production"

# 5. Check shifts
curl -X GET "http://localhost:8000/monitoring/check" \
  -H "X-API-Key: default-api-key-change-in-production"
```

### Cenário 2: Teste de Monitoramento

1. Fazer várias predições
2. Definir baselines
3. Fazer mais predições
4. Verificar desvios
5. Obter resumo

```bash
# 1. Make multiple predictions
for i in {1..10}; do
  curl -X POST "http://localhost:8000/predict" \
    -H "X-API-Key: default-api-key-change-in-production" \
    -H "Content-Type: application/json" \
    -d "{\"user_id\": 138131, \"item_ids\": [430292, 277119, 183411]}"
done

# 2. Set baselines
curl -X POST "http://localhost:8000/monitoring/baselines" \
  -H "X-API-Key: default-api-key-change-in-production"

# 3. Make more predictions
for i in {1..10}; do
  curl -X POST "http://localhost:8000/predict" \
    -H "X-API-Key: default-api-key-change-in-production" \
    -H "Content-Type: application/json" \
    -d "{\"user_id\": 911093, \"item_ids\": [457231, 259078, 183087]}"
done

# 4. Check shifts
curl -X GET "http://localhost:8000/monitoring/check" \
  -H "X-API-Key: default-api-key-change-in-production"

# 5. Get summary
curl -X GET "http://localhost:8000/monitoring/summary" \
  -H "X-API-Key: default-api-key-change-in-production"
```

### Cenário 3: Teste de Erros

1. API key ausente
2. API key inválida
3. Usuário/item não encontrado
4. Entrada inválida

```bash
# 1. Missing API key
curl -X GET "http://localhost:8000/health"

# Expected: 401 Unauthorized

# 2. Invalid API key
curl -X GET "http://localhost:8000/health" \
  -H "X-API-Key: wrong-key"

# Expected: 403 Forbidden

# 3. Invalid user/item (if model doesn't have them)
curl -X POST "http://localhost:8000/predict" \
  -H "X-API-Key: default-api-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 99999, "item_ids": [99999]}'

# Expected: 400 Bad Request

# 4. Invalid input
curl -X POST "http://localhost:8000/predict" \
  -H "X-API-Key: default-api-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 138131, "item_ids": []}'

# Expected: 400 Bad Request
```

## Testes de Performance

### Teste de Carga Simples

```bash
# Install Apache Bench if not installed
# sudo apt-get install apache2-utils

# Test health endpoint
ab -n 1000 -c 10 -H "X-API-Key: $API_KEY" http://localhost:8000/health

# Test prediction endpoint
ab -n 100 -c 5 -p payload.json -T application/json -H "X-API-Key: $API_KEY" http://localhost:8000/predict
```

Crie o arquivo `payload.json`:
```json
{
  "user_id": 123,
  "item_ids": [1, 2, 3, 4, 5]
}
```

## Testes com Python

### Usando requests

```python
import requests

API_KEY = "default-api-key-change-in-production"
BASE_URL = "http://localhost:8000"
headers = {"X-API-Key": API_KEY}

# Health check
response = requests.get(f"{BASE_URL}/health", headers=headers)
print(response.json())

# Single prediction
response = requests.post(
    f"{BASE_URL}/predict",
    headers=headers,
    json={"user_id": 138131, "item_ids": [430292, 277119, 183411, 457231, 259078]}
)
print(response.json())

# Top-k recommendations
response = requests.get(
    f"{BASE_URL}/recommend/138131?k=10",
    headers=headers
)
print(response.json())
```

## Solução de Problemas

### API não responde

```bash
# Verificar se o serviço está rodando
curl -X GET "http://localhost:8000/health" \
  -H "X-API-Key: default-api-key-change-in-production" \
  -v
```

### Erro de autenticação

```bash
# Verificar a API key
echo $API_KEY

# Testar com a chave explícita
curl -X GET "http://localhost:8000/health" \
  -H "X-API-Key: default-api-key-change-in-production"
```

### Modelo não carregado

```bash
# Verificar logs do serviço
# Se rodando localmente, verifique o terminal onde o uvicorn está rodando

# Verificar se o arquivo do modelo existe
ls -la models/model.pt
```

## Observações

- A API key padrão é `default-api-key-change-in-production` - **Mude isso em produção**
- Todos os endpoints requerem autenticação via header `X-API-Key`
- A API roda na porta 8000 por padrão
- O modelo deve estar em `models/model.pt` ou o caminho configurado
- Para produção, considere usar HTTPS em vez de HTTP
