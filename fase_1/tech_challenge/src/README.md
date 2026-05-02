# 📞 Telco Churn Prediction - Sistema de ML End-to-End

**Sistema profissional de alerta precoce para identificar clientes com alto risco de cancelamento em operadoras de telecomunicações.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)

---

## 📋 Tabela de Conteúdos

- [Visão Geral](#visão-geral)
- [Quick Start](#quick-start)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [API Endpoints](#api-endpoints)
- [Feature Reference](#feature-reference)
- [Exemplos de Uso](#exemplos-de-uso)
- [Instalação e Setup](#instalação-e-setup)

---

## 🎯 Visão Geral

### Proposta de Valor
Implementar um sistema de previsão de churn que identifique clientes com alta propensão ao cancelamento, permitindo ações de retenção personalizadas antes do encerramento do contrato.

### Padrões Críticos Identificados

```
🔴 ALTO RISCO DE CHURN (40-45%):
  • Contratos mês-a-mês
  • Fibra Óptica (com altos custos)
  • Pagamento por Cheque Eletrônico
  • Novos clientes (< 6 meses)

🟢 BAIXO RISCO (< 10%):
  • Contratos 2 anos
  • Sem Internet
  • Pagamento automático (cartão/transferência)
  • Clientes leais (> 5 anos)
```

---

## 🚀 Quick Start

### 1. Clone e Setup
```bash
# Clone o repositório
git clone https://github.com/seu-user/telco-churn-prediction.git
cd telco-churn-prediction

# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou .venv\Scripts\activate  # Windows

# Instalar dependências
pip install -e ".[dev]"
```

### 2. Rodar a API
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

A API estará disponível em: `http://localhost:8000`

### 3. Documentação Interativa
- **Swagger UI**: `http://localhost:8000/api/docs`
- **ReDoc**: `http://localhost:8000/api/redoc`

---

## 📂 Estrutura do Projeto

```
telco-churn-prediction/
│
├── src/                              # Código fonte
│   ├── config.py                    # Configurações centralizadas
│   ├── logging_config.py            # Logger estruturado
│   │
│   ├── api/                         # FastAPI Application
│   │   ├── main.py                  # Endpoints (6 rotas)
│   │   ├── schemas.py               # Modelos Pydantic
│   │   └── feature_transformer.py   # Transformação de features
│   │
│   ├── data/                        # Data Processing
│   │   ├── loader.py                # Carregamento de dados
│   │   └── preprocessing.py         # TelcoDataPreprocessor
│   │
│   ├── models/                      # Machine Learning
│   │   ├── pipeline.py              # TelcoPipeline (sklearn)
│   │   ├── transformers.py          # Transformadores custom
│   │   ├── baseline.py              # Experimentos MLflow
│   │   └── inference.py             # PredictionService
│   │
│   └── evaluation/                  # Avaliação
│       └── metrics.py               # Métricas técnicas + negócio
│
├── tests/                           # Testes automatizados
│   ├── conftest.py                  # Fixtures pytest
│   ├── test_config.py
│   ├── test_models.py
│   └── test_preprocessing.py
│
├── data/                            # Dados
│   ├── raw/
│   └── processed/
│
├── notebooks/                       # Análises exploratórias
│   └── 01_eda_and_ml_canvas.ipynb
│
├── docs/                            # Documentação
│   ├── DICIONARIO_DADOS.md
│   ├── RELATORIO_EDA.md
│   └── ml_canvas.md
│
├── pyproject.toml                   # Configuração do projeto
├── Makefile                         # Comandos úteis
└── README.md                        # Este arquivo
```

---

## 🔌 API Endpoints

### 1. Health Check
```http
GET /api/health
```
**Resposta:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "model_loaded": true
}
```

### 2. Obter Informações do Modelo
```http
GET /api/model-info
```
**Resposta:**
```json
{
  "model_type": "XGBoost",
  "model_version": "0.1.0",
  "n_features": 30,
  "features_used": ["gender", "senior_citizen", ...]
}
```

### 3. Predição Individual
```http
POST /api/predict
Content-Type: application/json
```

**Body:**
```json
{
  "features": {
    "gender": "Male",
    "senior_citizen": "No",
    "partner": "No",
    "dependents": "No",
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
  }
}
```

**Resposta:**
```json
{
  "prediction": 1,
  "probability": 0.673,
  "confidence": 0.673,
  "processing_time_ms": 5.2
}
```

### 4. Predição em Lote
```http
POST /api/predict-batch
Content-Type: application/json
```

**Body:**
```json
{
  "samples": [
    { "features": { ... } },
    { "features": { ... } },
    { "features": { ... } }
  ],
  "return_probabilities": true
}
```

**Resposta:**
```json
{
  "predictions": [1, 0, 1],
  "probabilities": [0.673, 0.005, 0.521],
  "batch_size": 3,
  "processing_time_ms": 15.3
}
```

---

## 📊 Feature Reference

### Features Binárias (Yes/No)
Aceita: `"Yes"`, `"No"` (case-insensitive)

- senior_citizen
- partner
- dependents
- phone_service
- paperless_billing
- online_security
- online_backup
- device_protection
- tech_support
- streaming_tv
- streaming_movies

### gender
Aceita: `"Male"`, `"Female"` (case-insensitive)

### multiple_lines
Aceita: `"Yes"`, `"No"`, `"No phone service"` (case-insensitive)

### internet_service
Aceita: `"DSL"`, `"Fiber optic"`, `"No"` (case-insensitive)

### online_security / online_backup / device_protection / tech_support / streaming_tv / streaming_movies
Aceita: `"Yes"`, `"No"`, `"No internet service"` (case-insensitive)

### contract
Aceita: `"Month-to-month"`, `"One year"`, `"Two year"` (case-insensitive)

### payment_method
Aceita: `"Bank transfer (automatic)"`, `"Credit card (automatic)"`, `"Electronic check"`, `"Mailed check"` (case-insensitive)

### Numéricos
- **tenure_months** (int): 0-72+ (meses de contrato)
- **monthly_charges** (float): 0.00-150.00+ (dólares)
- **total_charges** (float): 0.00-10000.00+ (dólares)

---

## 💡 Exemplos de Uso

### Cliente de ALTO RISCO (Churn = 1)
```json
{
  "features": {
    "gender": "Male",
    "senior_citizen": "No",
    "partner": "No",
    "dependents": "No",
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
  }
}
```
**Resposta esperada:** `prediction: 1`, `probability: 0.67`

### Cliente de BAIXO RISCO (Churn = 0)
```json
{
  "features": {
    "gender": "Female",
    "senior_citizen": "No",
    "partner": "Yes",
    "dependents": "Yes",
    "tenure_months": 60,
    "phone_service": "Yes",
    "multiple_lines": "Yes",
    "internet_service": "DSL",
    "online_security": "Yes",
    "online_backup": "Yes",
    "device_protection": "Yes",
    "tech_support": "Yes",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
    "contract": "Two year",
    "paperless_billing": "No",
    "payment_method": "Bank transfer (automatic)",
    "monthly_charges": 89.50,
    "total_charges": 5370.00
  }
}
```
**Resposta esperada:** `prediction: 0`, `probability: 0.005`

---

## 🛠️ Instalação e Setup

### Pré-requisitos
- Python ≥ 3.10
- pip ou conda
- Git

### Passos de Instalação

1. **Clone o repositório**
   ```bash
   git clone <repo-url>
   cd telco-churn-prediction
   ```

2. **Crie ambiente virtual**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```

3. **Instale dependências**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Configure variáveis de ambiente (opcional)**
   ```bash
   cp .env.example .env
   # Edite .env conforme necessário
   ```

5. **Prepare os dados**
   ```bash
   # Os dados devem estar em: data/processed/telco_churn_processed.csv
   python 01_eda_analysis.py  # Executa EDA
   ```

6. **Treine o modelo**
   ```bash
   python train_final_model.py
   ```

7. **Inicie a API**
   ```bash
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

---

## 📦 Dependências Principais

Ver `pyproject.toml` ou `requirements.txt` para lista completa:

- **Dados**: pandas, numpy
- **ML**: scikit-learn, xgboost
- **API**: fastapi, pydantic, uvicorn
- **Monitoramento**: mlflow
- **Dev**: pytest, ruff, black

---

## 📞 Suporte

Para questões ou problemas, abra uma issue no repositório.

---

**Desenvolvido com ❤️ - Projeto FIAP MLOps**
