# src/app — API Architecture

Documentação técnica do código em `src/app/`. Para o contexto geral do projeto (dataset, decisão batch vs. real-time, stack de nuvem), veja o [README raiz](../../README.md).

## 1. Camadas

```
api/        HTTP — rotas, validação de request, injeção de dependência.
             Nenhuma lógica de negócio aqui.

schemas/    Contratos Pydantic (request/response). A única forma que
             atravessa a fronteira HTTP.

services/   Lógica de negócio. Onde o modelo ONNX é carregado e rodado.
             Não importa nada de FastAPI.

core/       Preocupações transversais: logging e exceções de domínio.
             Não depende de nenhuma outra camada.

monitoring/ Instrumentação Prometheus. services/ chama funções daqui em
             vez de importar prometheus_client diretamente.

config.py   Settings centralizadas (Pydantic BaseSettings).

main.py     Monta tudo: cria a app, registra middlewares, inclui as
             rotas, carrega o modelo no startup via lifespan.
```

Regra geral: uma requisição entra por `api/`, é validada por `schemas/`, a rota chama `services/`, que devolve um objeto `schemas/` de volta — `api/` nunca fala diretamente com `onnxruntime`, `numpy` ou o modelo.

## 2. Mapa de arquivos

| Arquivo | Responsabilidade |
|---|---|
| `main.py` | App factory (`create_app`) + `lifespan`, que carrega o `InferenceService` uma única vez no startup e guarda em `app.state`. |
| `config.py` | `Settings` (Pydantic BaseSettings): caminho do modelo, versão, lista de classes, limites de batch/texto, nível de log. Cacheada via `get_settings()`. |
| `api/router.py` | Router raiz: monta `health` sem prefixo e `v1_router` sob `/v1`. |
| `api/dependencies.py` | `get_inference_service(request)` — recupera o `InferenceService` salvo em `app.state`, levanta `ModelNotReadyError` se chamado antes do startup terminar. |
| `api/v1/__init__.py` | Agrupa `classify` + `model_info` em um único `v1_router`. |
| `api/v1/classify.py` | `POST /v1/classify/batch` (fluxo principal) e `POST /v1/classify` (batch de tamanho 1, endpoint secundário). |
| `api/v1/health.py` | `GET /health` — retorna 503 se o modelo ainda não carregou. |
| `api/v1/model_info.py` | `GET /v1/model/info` — versão do modelo, classes, otimização aplicada. |
| `schemas/classify.py` | `ClassifyRequest`, `BatchClassifyRequest`, `PredictionResult`, `ClassifyResponse`, `BatchClassifyResponse`. Limites de tamanho lidos de `Settings`. |
| `schemas/health.py` | `HealthResponse`. |
| `schemas/model_info.py` | `ModelInfoResponse`. |
| `services/inference.py` | `InferenceService` — carrega a sessão ONNX Runtime, expõe `predict_batch()`, `is_ready()`, `warm_up()`. Único ponto do código que sabe o formato bruto da saída do modelo. |
| `services/preprocessing.py` | Limpeza mínima de texto (colapsa espaços em branco). Vetorização TF-IDF fica dentro do grafo ONNX, não aqui. |
| `core/exceptions.py` | `AppError` e subclasses (`ModelNotReadyError`, `InvalidInputError`, `InferenceError`) + handlers que garantem resposta JSON estruturada mesmo em falha inesperada. |
| `core/logging.py` | `configure_logging()` — formato e nível de log únicos para toda a aplicação. |
| `monitoring/metrics.py` | Métricas Prometheus (`http_requests_total`, `http_request_duration_seconds`, `inference_duration_seconds`, `classify_batch_size`, `classification_by_label_total`), middleware de timing, endpoint `GET /metrics`. |

## 3. Como rodar

### 3.1 Via Docker Compose (recomendado)

Sobe a API junto com Prometheus e Grafana, como documentado no README raiz:

```bash
docker compose -f docker/docker-compose.yml up --build
```

- API: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

### 3.2 Rodando localmente com uvicorn

Útil durante o desenvolvimento de `src/app/`, sem precisar rebuildar a imagem Docker a cada mudança.

```bash
# a partir da raiz do repositório
cd src
pip install -r ../requirements.txt   # ainda não existe — ver seção 6
uvicorn app.main:app --reload --app-dir .
```

- `--reload` reinicia o processo a cada alteração de arquivo — útil em desenvolvimento, nunca use em produção.
- `--app-dir .` garante que `app` seja importado como pacote a partir de `src/`, batendo com os imports absolutos usados no código (`from app.config import ...`).
- Sem `models/model.onnx` presente, o `lifespan` vai levantar `InferenceError` no startup — isso é esperado até `ml/convert_to_onnx.py` existir (seção 6). Para testar a API antes disso, aponte `APP_MODEL_PATH` para qualquer `.onnx` de teste, ou comente temporariamente o carregamento do modelo.

### 3.3 Variáveis de ambiente

Todas com prefixo `APP_` (ver `config.py`). As mais relevantes para rodar localmente:

| Variável | Padrão | Descrição |
|---|---|---|
| `APP_MODEL_PATH` | `models/model.onnx` | Caminho do artefato ONNX carregado no startup |
| `APP_MODEL_VERSION` | `tfidf-rf-v1.0-onnx` | Reportado em `/v1/model/info` e em cada resposta |
| `APP_MAX_BATCH_SIZE` | `64` | Limite de itens em `/v1/classify/batch` |
| `APP_LOG_LEVEL` | `INFO` | Nível de log |

Podem ser definidas em um arquivo `.env` na raiz (não commitado — ver `.gitignore`) ou exportadas diretamente no shell.

### 3.4 Verificando que subiu

```bash
curl http://localhost:8000/health
# {"status": "ok", "model_loaded": true}  → sucesso
# {"status": "unavailable", "model_loaded": false} com HTTP 503 → modelo não carregou
```

## 4. Como usar a API

### 4.1 Endpoints Disponíveis

A API expõe três endpoints principais:

| Endpoint | Método | Descrição |
|---|---|---|
| `/health` | GET | Health check para verificar se o modelo está carregado |
| `/v1/model/info` | GET | Informações sobre o modelo (versão, classes, otimização) |
| `/v1/classify` | POST | Classificação de único laudo médico |
| `/v1/classify/batch` | POST | Classificação em lote de múltiplos laudos |

### 4.2 Exemplos de Uso com cURL

#### Health Check
```bash
curl -s http://localhost:8000/health
```
**Resposta esperada:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

#### Informações do Modelo
```bash
curl -s http://localhost:8000/v1/model/info
```
**Resposta esperada:**
```json
{
  "model_version": "tfidf-rf-v1.0-onnx",
  "class_labels": [
    "Neoplasms",
    "Digestive system diseases",
    "Nervous system diseases",
    "Cardiovascular diseases",
    "General pathological conditions"
  ],
  "num_classes": 5,
  "optimization": "onnx"
}
```

#### Classificação de Único Laudo
```bash
curl -s -X POST "http://localhost:8000/v1/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The patient presented with a malignant neoplasm of the colon requiring surgical resection."
  }'
```
**Resposta esperada:**
```json
{
  "label": "Neoplasms",
  "confidence": 0.408,
  "scores": {
    "Neoplasms": 0.408,
    "Digestive system diseases": 0.146,
    "Nervous system diseases": 0.139,
    "Cardiovascular diseases": 0.119,
    "General pathological conditions": 0.188
  },
  "model_version": "tfidf-rf-v1.0-onnx",
  "inference_ms": 0.256
}
```

#### Classificação em Lote
```bash
curl -s -X POST "http://localhost:8000/v1/classify/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "The patient presented with a malignant neoplasm of the colon requiring surgical resection.",
      "Gastroesophageal reflux disease was diagnosed after endoscopy examination showing esophagitis.",
      "MRI revealed a demyelinating lesion consistent with multiple sclerosis in the white matter."
    ]
  }'
```
**Resposta esperada:**
```json
{
  "results": [
    {
      "label": "Neoplasms",
      "confidence": 0.408,
      "scores": {
        "Neoplasms": 0.408,
        "Digestive system diseases": 0.146,
        "Nervous system diseases": 0.139,
        "Cardiovascular diseases": 0.119,
        "General pathological conditions": 0.188
      }
    },
    {
      "label": "Digestive system diseases",
      "confidence": 0.352,
      "scores": {
        "Neoplasms": 0.201,
        "Digestive system diseases": 0.352,
        "Nervous system diseases": 0.128,
        "Cardiovascular diseases": 0.145,
        "General pathological conditions": 0.174
      }
    },
    {
      "label": "Nervous system diseases",
      "confidence": 0.391,
      "scores": {
        "Neoplasms": 0.187,
        "Digestive system diseases": 0.134,
        "Nervous system diseases": 0.391,
        "Cardiovascular diseases": 0.152,
        "General pathological conditions": 0.136
      }
    }
  ],
  "model_version": "tfidf-rf-v1.0-onnx",
  "batch_size": 3,
  "inference_ms": 0.512
}
```

### 4.3 Documentação Interativa

A API inclui documentação interativa automática via Swagger UI:

```bash
# Abrir no navegador:
http://localhost:8000/docs
```

A documentação também está disponível em formato ReDoc:

```bash
http://localhost:8000/redoc
```

### 4.4 Categorias Médicas Suportadas

O modelo classifica laudos médicos em 5 categorias:

1. **Neoplasms** - Tumores e cânceres
2. **Digestive system diseases** - Doenças do sistema digestivo
3. **Nervous system diseases** - Doenças do sistema nervoso
4. **Cardiovascular diseases** - Doenças cardiovasculares
5. **General pathological conditions** - Condições patológicas gerais

### 4.5 Validações e Limites

- **Tamanho máximo do texto**: 5.000 caracteres (configurável via `APP_MAX_TEXT_LENGTH`)
- **Tamanho máximo do batch**: 64 itens (configurável via `APP_MAX_BATCH_SIZE`)
- **Texto vazio ou apenas espaços**: Rejeitado com HTTP 422
- **Content-Type**: Deve ser `application/json`

### 4.6 Monitoramento e Métricas

A API expõe métricas Prometheus para monitoramento:

```bash
curl http://localhost:8000/metrics
```

Métricas disponíveis:
- `http_requests_total` - Contagem total de requisições HTTP
- `http_request_duration_seconds` - Duração das requisições HTTP
- `inference_duration_seconds` - Tempo de inferência do modelo
- `classify_batch_size` - Tamanho dos batches de classificação
- `classification_by_label_total` - Contagem de classificações por label

## 6. Como as peças se conectam no startup

```
main.py: create_app()
  │
  ├─ lifespan (executa antes de aceitar requisições):
  │     configure_logging(settings.log_level)
  │     InferenceService(model_path, class_labels, model_version)
  │           └─ carrega a sessão ONNX Runtime
  │     inference_service.warm_up()
  │           └─ dispara uma inferência descartável
  │     app.state.inference_service = inference_service
  │
  ├─ register_exception_handlers(app)
  ├─ setup_metrics(app)          # middleware + GET /metrics
  └─ app.include_router(api_router)
```

Depois do startup, cada requisição em `/v1/classify` ou `/v1/classify/batch` usa `Depends(get_inference_service)` para pegar a mesma instância — o modelo nunca é recarregado por requisição.

## 7. Testes Automatizados

A API possui um suite completo de testes automatizados que validam o funcionamento de todos os endpoints usando dados de diagnósticos clínicos realistas.

### 7.1 Executar os Testes

```bash
# A partir da raiz do repositório
poetry run pytest src/app/tests/ -v
```

### 7.2 Estrutura dos Testes

- **`test_classify_endpoints.py`** - 33 testes para endpoints de classificação
- **`test_health_endpoints.py`** - 18 testes para health check
- **`test_model_info.py`** - 16 testes para informações do modelo

### 7.3 Dados de Teste

Os testes utilizam dados de diagnósticos clínicos baseados na estrutura do dataset `treino_modelo`, cobrindo todas as 5 categorias médicas com casos realistas de neoplasms, doenças digestivas, neurológicas, cardiovasculares e condições patológicas gerais.

### 7.4 Cobertura

Os testes cobrem:
- ✅ Happy paths para todos os endpoints
- ✅ Validação de entrada (texto vazio, espaços, tamanho excessivo)
- ✅ Consistência de respostas
- ✅ Integração entre endpoints
- ✅ Casos de borda e caracteres especiais
- ✅ Performance e tempo de resposta

## 8. O que ainda está pendente

Este README documenta o que já existe em código. Ainda faltam, fora de `src/app/`:

- [x] `tests/` — suite completo de testes automatizados implementado em `src/app/tests/`
- [ ] `docker/Dockerfile` — build da imagem da API
- [ ] `ml/train.py`, `ml/convert_to_onnx.py` — geram o `models/model.onnx` que `InferenceService` espera encontrar

`services/inference.py` assume um contrato específico do grafo ONNX (input chamado `"input_text"`, outputs `"output_label"` e `"output_probability"`) — `ml/convert_to_onnx.py` precisa seguir exatamente esse contrato quando for escrito.
