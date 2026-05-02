# Smoke Test — Guia de Execução

Este guia descreve como executar o smoke test da API de predição de churn, disponível em `tests/smoke_test/MLTelco.postman_collection`.

---

## Pré-requisitos

- API rodando localmente ou via Docker na porta `8000`
- Uma das opções abaixo para executar a coleção Postman:
  - [Postman](https://www.postman.com/downloads/) (interface gráfica)
  - [Newman](https://www.npmjs.com/package/newman) (linha de comando, requer Node.js)

---

## Subindo a API

### Localmente

```bash
make run-api
# ou
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Via Docker Compose

```bash
make docker-compose-up
# ou
docker compose up -d
```

Aguarde a API responder em `http://localhost:8000/health` antes de rodar os testes.

---

## Opção 1 — Postman (interface gráfica)

1. Abra o Postman
2. Clique em **Import**
3. Selecione o arquivo `tests/smoke_test/MLTelco.postman_collection`
4. Abra a coleção importada **ML Telco**
5. Clique em **Run collection**
6. Clique em **Run ML Telco**
7. Verifique que todos os testes estão marcados como **passed**

---

## Opção 2 — Newman (linha de comando)

### Instalação

```bash
npm install -g newman
```

### Execução

```bash
newman run tests/smoke_test/MLTelco.postman_collection
```

### Execução com relatório HTML

```bash
npm install -g newman-reporter-htmlextra

newman run tests/smoke_test/MLTelco.postman_collection \
  --reporters htmlextra \
  --reporter-htmlextra-export reports/smoke_test_report.html
```

---

## Endpoints testados

| Requisição | Método | Endpoint | Descrição |
|---|---|---|---|
| Health | `GET` | `/health` | Verifica se a API está saudável e o modelo carregado |
| predict | `POST` | `/predict` | Predição com payload completo válido |
| predict sem o gender | `POST` | `/predict` | Valida erro 422 ao omitir campo obrigatório |

---

## Asserções cobertas

### `GET /health`
- Status HTTP 200
- Campo `status` igual a `"healthy"`
- Campo `version` é uma string
- Campo `model_loaded` igual a `true`
- Tempo de resposta abaixo de 2000 ms

### `POST /predict` (payload válido)
- Status HTTP 200
- Campo `prediction` é `0` ou `1`
- Campo `probability` entre `0` e `1`
- Campo `confidence` entre `0` e `1`
- Campo `processing_time_ms` maior que `0`
- Tempo de resposta abaixo de 5000 ms

### `POST /predict` (sem `gender`)
- Status HTTP 422
- Campo `detail` é um array não vazio
- Pelo menos um erro referencia o campo `gender`
- Cada item de erro possui `type`, `loc` e `msg`
- Tempo de resposta abaixo de 3000 ms

---

## Exemplo de payload para `/predict`

```json
{
  "features": {
    "senior_citizen": "No",
    "partner": "No",
    "dependents": "No",
    "gender": "Male",
    "tenure_months": 2,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "DSL",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Mailed check",
    "monthly_charges": 53.85,
    "total_charges": 108.15
  },
  "return_probabilities": true
}
```

