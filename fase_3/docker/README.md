# Docker - Configuração de Deploy em Produção

Este diretório contém a configuração completa para executar a API de classificação médica em ambiente de produção usando Docker Compose, incluindo monitoramento com Prometheus e Grafana.

## 📋 Visão Geral

A stack Docker consiste em três serviços principais:

- **API**: Serviço de classificação médica com modelo ONNX otimizado
- **Prometheus**: Coleta de métricas e monitoramento de performance
- **Grafana**: Visualização de dashboards e alertas

## 🏗️ Arquitetura

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   API (porta    │─────▶│  Prometheus     │─────▶│    Grafana      │
│   8000)         │      │  (porta 9090)   │      │  (porta 3000)   │
│                 │      │                 │      │                 │
│ - FastAPI       │      │ - Métricas      │      │ - Dashboards    │
│ - ONNX Runtime  │      │ - Scraping 5s   │      │ - Visualização  │
│ - Prometheus    │      │ - Armazenamento │      │ - Alertas       │
│   Client        │      │                 │      │                 │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```

## 📁 Estrutura de Arquivos

```
docker/
├── Dockerfile                          # Imagem da API
├── docker-compose.yml                  # Orquestração dos serviços
├── .dockerignore                       # Arquivos ignorados no build
├── prometheus.yml                      # Configuração do Prometheus
└── grafana/
    └── provisioning/
        ├── datasources/
        │   └── datasources.yml         # Configuração datasource Prometheus
        └── dashboards/
            ├── dashboard.yml           # Provider de dashboards
            └── medical-classification-dashboard.json  # Dashboard monitoramento
```

## 🚀 Como Iniciar

### Pré-requisitos

- Docker instalado (versão 20.10+)
- Docker Compose instalado (versão 2.0+)
- Mínimo de 2GB RAM disponível

### Iniciar todos os serviços

```bash
# A partir da raiz do repositório
docker-compose -f docker/docker-compose.yml up -d --build
```

### Verificar status dos serviços

```bash
docker-compose -f docker/docker-compose.yml ps
```

### Visualizar logs

```bash
# Logs de todos os serviços
docker-compose -f docker/docker-compose.yml logs

# Logs específicos da API
docker-compose -f docker/docker-compose.yml logs api

# Logs específicos do Prometheus
docker-compose -f docker/docker-compose.yml logs prometheus

# Logs específicos do Grafana
docker-compose -f docker/docker-compose.yml logs grafana
```

### Parar todos os serviços

```bash
docker-compose -f docker/docker-compose.yml down
```

### Parar e remover volumes

```bash
docker-compose -f docker/docker-compose.yml down -v
```

## 🔌 Acesso aos Serviços

### API de Classificação

- **URL**: http://localhost:8000
- **Documentação Swagger**: http://localhost:8000/docs
- **Documentação ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health
- **Métricas**: http://localhost:8000/metrics

### Prometheus

- **URL**: http://localhost:9090
- **Status**: http://localhost:9090/-/healthy
- **Targets**: http://localhost:9090/targets
- **Configuração**: http://localhost:9090/config

### Grafana

- **URL**: http://localhost:3000
- **Usuário**: `admin`
- **Senha**: `admin`
- **Dashboard Monitoramento**: http://localhost:3000/d/anh7s9/medical-classification-api-monitoramento

## 📊 Dashboard Grafana

O dashboard configurado automaticamente contém os seguintes painéis (conforme requisitos do Tech Challenge Fase 3):

### Painéis Obrigatórios

1. **Total de Requisições (por segundo)**
   - Taxa de requisições HTTP por segundo
   - Métrica: `sum(rate(http_requests_total[5m]))`

2. **Latência de Resposta (p95)**
   - 95º percentil do tempo de resposta
   - Métrica: `histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))`

3. **Taxa de Erro (4xx + 5xx)**
   - Taxa de erros HTTP (4xx e 5xx)
   - Métricas:
     - `sum(rate(http_requests_total{status_code=~"4.."}[5m])) / sum(rate(http_requests_total[5m]))`
     - `sum(rate(http_requests_total{status_code=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))`

### Painéis Adicionais

- **Taxa de Requisições por Status HTTP** - Breakdown por 2xx, 4xx, 5xx
- **Distribuição de Latência** - p50, p95, p99 percentuais
- **Requisições por Endpoint** - Taxa por endpoint da API
- **Latência por Endpoint** - p95 por endpoint
- **Requisições por Método HTTP** - GET vs POST

## ⚙️ Configuração

### Variáveis de Ambiente da API

As seguintes variáveis de ambiente são configuradas no `docker-compose.yml`:

| Variável | Valor | Descrição |
|----------|-------|-----------|
| `PYTHONUNBUFFERED` | `1` | Saída de logs sem buffer |
| `APP_MODEL_PATH` | `/app/treino_modelo/artifacts/model.onnx` | Caminho do modelo ONNX |
| `APP_MODEL_VERSION` | `tfidf-rf-v1.0-onnx` | Versão do modelo |

### Configuração Prometheus

- **Intervalo de scraping**: 5 segundos
- **Target**: `api:8000/metrics`
- **Armazenamento**: Em memória (padrão)

### Configuração Grafana

- **Usuário admin**: `admin`
- **Senha admin**: `admin`
- **Datasource**: Prometheus (auto-configurado)
- **Dashboard**: Auto-provisionado via arquivo JSON

## 🔧 Dockerfile

A imagem da API é construída com:

- **Base**: `python:3.11-slim`
- **Dependências**: Gerenciadas via Poetry
- **Locale**: `en_US.UTF-8` (necessário para ONNX Runtime)
- **Porta**: 8000
- **Comando**: `uvicorn app.main:app --host 0.0.0.0 --port 8000`

### Build da Imagem

```bash
docker build -f docker/Dockerfile -t medical-classification-api ..
```

## 🧪 Testes com Docker

### Gerar tráfego de teste

```bash
# Classificações únicas
for i in {1..10}; do
  curl -s -X POST "http://localhost:8000/v1/classify" \
    -H "Content-Type: application/json" \
    -d '{"text": "Patient with medical condition requiring diagnosis"}' > /dev/null
done

# Classificações em lote
curl -s -X POST "http://localhost:8000/v1/classify/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Patient with malignant neoplasm",
      "Gastroesophageal reflux disease",
      "MRI showing multiple sclerosis"
    ]
  }'

# Gerar erros (para testar painel de taxa de erro)
for i in {1..5}; do
  curl -s -X POST "http://localhost:8000/v1/classify" \
    -H "Content-Type: application/json" \
    -d '{"text": ""}' > /dev/null
done
```

### Verificar métricas no Prometheus

```bash
# Total de requisições
curl -s "http://localhost:9090/api/v1/query?query=http_requests_total"

# Taxa de requisições por segundo
curl -s "http://localhost:9090/api/v1/query?query=rate(http_requests_total[5m])"

# Latência p95
curl -s "http://localhost:9090/api/v1/query?query=histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))"
```

## 📈 Métricas Disponíveis

A API expõe as seguintes métricas Prometheus:

- `http_requests_total` - Contagem total de requisições HTTP
- `http_request_duration_seconds` - Duração das requisições HTTP (histograma)
- `inference_duration_seconds` - Tempo de inferência do modelo
- `classify_batch_size` - Tamanho dos batches de classificação
- `classification_by_label_total` - Contagem de classificações por label

## 🛠️ Troubleshooting

### API não inicia

```bash
# Verificar logs da API
docker-compose -f docker/docker-compose.yml logs api

# Verificar se o modelo existe
docker exec docker_api_1 ls -la /app/treino_modelo/artifacts/

# Verificar variáveis de ambiente
docker exec docker_api_1 env | grep APP_
```

### Prometheus não coleta métricas

```bash
# Verificar status dos targets
curl -s http://localhost:9090/api/v1/targets

# Verificar configuração do Prometheus
docker exec docker_prometheus_1 cat /etc/prometheus/prometheus.yml

# Testar acesso da API ao endpoint /metrics
curl -s http://localhost:8000/metrics
```

### Grafana não mostra dados

```bash
# Verificar se datasource está configurado
curl -s -u admin:admin http://localhost:3000/api/datasources

# Verificar se dashboard existe
curl -s -u admin:admin http://localhost:3000/api/search?query=Medical

# Reiniciar Grafana
docker-compose -f docker/docker-compose.yml restart grafana
```

### Erro de locale no ONNX Runtime

Se encontrar erro `Failed to construct locale with name:en_US.UTF-8`, o Dockerfile já inclui a correção:
- Instalação do pacote `locales`
- Configuração de `en_US.UTF-8` no `/etc/locale.gen`
- Variáveis de ambiente `LANG` e `LC_ALL`

## 📝 Requisitos do Tech Challenge Fase 3

Esta configuração Docker atende aos seguintes requisitos obrigatórios:

### Etapa 3 - Monitoramento e Observabilidade

✅ **Docker Compose funcional** - API + Prometheus + Grafana configurados
✅ **Instrumentação da API** - Métricas expostas via prometheus_client
✅ **Métricas básicas** - Tempo de requisição e contagem de chamadas
✅ **Dashboard Grafana** - Pelo menos 3 painéis implementados:
  - Total de requisições
  - Latência/tempo de resposta
  - Taxa de erro

### Etapa 4 - Otimização de Latência

✅ **Modelo otimizado** - ONNX Runtime para inferência rápida
✅ **Containerização** - API empacotada em Docker
✅ **Monitoramento** - Stack completa de observabilidade

## 🔐 Segurança

### Considerações de Produção

Para ambiente de produção, considere:

1. **Alterar senhas padrão** do Grafana
2. **Configurar HTTPS** usando reverse proxy (nginx/traefik)
3. **Limitar recursos** dos containers (CPU, memória)
4. **Configurar autenticação** na API
5. **Implementar rate limiting**
6. **Usar secrets management** para credenciais
7. **Configurar backups** do Grafana
8. **Implementar logging centralizado**

### Exemplo de limitação de recursos

```yaml
# Adicionar ao docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
```

## 📚 Referências

- [Documentação FastAPI](https://fastapi.tiangolo.com/)
- [Documentação Prometheus](https://prometheus.io/docs/)
- [Documentação Grafana](https://grafana.com/docs/)
- [Documentação ONNX Runtime](https://onnxruntime.ai/docs/)
- [Tech Challenge Fase 3 - PDF](../../Downloads/MLET%20-%20Tech%20Challenge%20Fase%203.pdf)

## 🤝 Suporte

Para dúvidas ou problemas:
1. Verificar os logs dos containers
2. Consultar a documentação da API em `src/README.md`
3. Verificar os requisitos do Tech Challenge no PDF
4. Testar os endpoints individualmente via cURL
