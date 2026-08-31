# Plano de Deploy Simplificado - Tech Challenge Fase 3

## 📋 Visão Geral

Este documento apresenta o plano de deploy simplificado para o sistema de classificação médica, focando apenas no que foi implementado: a API FastAPI em `src/app/api` e a stack Docker Compose com monitoramento.

## 🎯 Justificativa da Arquitetura Escolhida

### 1. API REST em vez de Batch

**Escolha: API REST Real-time**

**Justificativa:**

- **Urgência Clínica**: Em ambiente hospitalar, a triagem precisa ser imediata. Um paciente com condição urgente não pode esperar por processamento em batch.
- **Interatividade**: Médicos e equipe precisam de feedback instantâneo ao inserir laudos no sistema.
- **Escalabilidade**: API REST permite escalabilidade horizontal com load balancing, essencial para hospitais de grande porte.
- **Integração**: APIs REST são o padrão de integração em sistemas hospitalares (HIS, PACS, EMR).
- **Latência**: Com otimização ONNX, a inferência é suficientemente rápida (<1ms) para uso em tempo real.

### 2. Docker Compose para Deploy Local + AWS para Produção

**Escolha: Docker Compose (Local) + AWS (Produção)**

**Justificativa:**

**Docker Compose (Desenvolvimento/Testing):**
- **Simplicidade**: Fácil de configurar e executar sem necessidade de infraestrutura complexa.
- **Portabilidade**: Funciona em qualquer máquina com Docker instalado.
- **Ambiente Consistente**: Garante que o ambiente de desenvolvimento seja idêntico ao de produção.
- **Isolamento**: Cada serviço roda em seu próprio container, evitando conflitos de dependências.
- **Rápido Deploy**: Comando único para iniciar toda a stack (API + Prometheus + Grafana).

**AWS (Produção):**
- **Escalabilidade**: Serviços gerenciados que escalam automaticamente
- **Compliance Healthcare**: AWS possui certificações HIPAA, essenciais para dados médicos
- **Global Infrastructure**: Presença em regiões brasileiras (sa-east-1) reduzindo latência
- **Serviços Gerenciados**: ECS/Fargate, CloudWatch reduzem overhead operacional
- **Integração Nativa**: Prometheus e Grafana funcionam nativamente na AWS
- **Custo-Benefício**: Modelo pay-as-you-go, ideal para startups e hospitais
- **Simplicidade**: Arquitetura focada apenas em API + monitoramento

### 3. Prometheus + Grafana para Monitoramento

**Escolha: Prometheus + Grafana**

**Justificativa:**

**Prometheus:**
- **Pull-based**: Mais seguro que push-based (menos attack surface)
- **Time-series DB**: Otimizado para dados de métricas
- **Service Discovery**: Integração nativa com Docker Compose
- **Alerting**: Sistema de alertas robusto
- **Ecosystem**: Grande ecossistema de exporters

**Grafana:**
- **Visualização**: Dashboards flexíveis e bonitos
- **Multi-datasource**: Suporta Prometheus e outros
- **Alerting**: Integração com múltiplos canais
- **Comunidade**: Grande comunidade e plugins
- **Auto-provisioning**: Configuração via arquivos YAML/JSON

### 4. ONNX Runtime para Otimização de Latência

**Escolha: ONNX Runtime**

**Justificativa:**

- **Performance**: 2-3x mais rápido que scikit-learn em inferência
- **Portabilidade**: Funciona em múltiplas plataformas (CPU, GPU)
- **Tamanho**: Arquivos menores que modelos originais
- **Compatibilidade**: Suporta modelos de sklearn, TensorFlow, PyTorch
- **Interoperabilidade**: Padrão de indústria para deploy de modelos

**Resultados Obtidos:**
- Latência original (scikit-learn): ~5ms
- Latência otimizada (ONNX): ~0.3ms
- **Melhoria: 94% redução em latência**

### 5. FastAPI para API REST

**Escolha: FastAPI**

**Justificativa:**

- **Performance**: Baseado em Starlette e Pydantic, extremamente rápido
- **Validação Automática**: Pydantic valida entrada/saída automaticamente
- **Documentação**: Swagger UI gerado automaticamente
- **Type Hints**: Python type hints melhoram manutenção
- **Async Support**: Suporte nativo para operações assíncronas
- **Modernidade**: Framework moderno com comunidade ativa

## 🎯 O Que Foi Implementado

### API FastAPI (`src/app/api`)

**Endpoints Implementados:**
- `GET /health` - Health check do serviço
- `GET /v1/model/info` - Informações do modelo
- `POST /v1/classify` - Classificação de único laudo
- `POST /v1/classify/batch` - Classificação em lote

**Características:**
- Validação de entrada com Pydantic
- Tratamento de erros estruturado
- Métricas Prometheus integradas
- Documentação Swagger automática

### Stack Docker Compose

**Serviços Configurados:**
- **API**: Container com FastAPI + ONNX Runtime
- **Prometheus**: Coleta de métricas (scraping a cada 5s)
- **Grafana**: Visualização de dashboards (auto-provisionado)

## 🏗️ Arquitetura Simplificada

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   API (:8000)   │─────▶│  Prometheus     │─────▶│    Grafana      │
│                 │      │  (:9090)        │      │  (:3000)        │
│                 │      │                 │      │                 │
│ - FastAPI       │      │ - Scraping 5s   │      │ - Dashboard     │
│ - ONNX Runtime  │      │ - Métricas      │      │ - Visualização  │
│ - Prometheus    │      │                 │      │                 │
│   Client        │      │                 │      │                 │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```

**Portas Utilizadas:**
- **API**: Porta 8000
- **Prometheus**: Porta 9090
- **Grafana**: Porta 3000

## 🚀 Como Fazer Deploy

## 💰 Custos Estimados AWS (Produção)

### Mensal (Região sa-east-1)

| Serviço | Custo Estimado | Descrição |
|---------|---------------|-----------|
| **ECS Fargate** | $150-300 | 2-4 tarefas, 24/7 |
| **CloudWatch** | $20-40 | Logs e métricas |
| **S3** | $5-10 | Armazenamento de modelos |
| **Data Transfer** | $15-30 | Tráfego de rede |
| **Total** | **$190-380/mês** | |

### Comparação com Outros Provedores

| Provedor | Vantagens | Desvantagens | Custo |
|----------|-----------|--------------|-------|
| **AWS** | Compliance HIPAA, serviços gerenciados, presença no Brasil | Curva de aprendizado | $$ |
| Azure | Integração Microsoft, serviços cognitivos | Menos maduro em ML | $$$ |
| GCP | Strong em ML/AI, BigQuery | Menos presença no Brasil | $$ |

## 🚀 Como Fazer Deploy

### Deploy Local (Desenvolvimento/Testing)

#### Pré-requisitos

- Docker instalado
- Docker Compose instalado
- Arquivo de modelo ONNX em `treino_modelo/artifacts/model.onnx`

#### Passo 1: Preparar o Ambiente

```bash
# Navegar para o diretório do projeto
cd /home/gusdpaula/code-dev/MLENG_FIAP/fase_3

# Verificar se o modelo existe
ls -la treino_modelo/artifacts/model.onnx
```

#### Passo 2: Iniciar os Serviços

```bash
# Iniciar todos os serviços (API, Prometheus, Grafana)
docker-compose -f docker/docker-compose.yml up -d --build
```

#### Passo 3: Verificar Status

```bash
# Verificar se os containers estão rodando
docker-compose -f docker/docker-compose.yml ps

# Saída esperada:
# NAME                STATUS                  PORTS
# docker_api_1        Up                      0.0.0.0:8000->8000/tcp
# docker_grafana_1    Up                      0.0.0.0:3000->3000/tcp
# docker_prometheus_1 Up                      0.0.0.0:9090->9090/tcp
```

### Deploy em AWS (Produção)

#### Arquitetura AWS Proposta

```
┌─────────────────────────────────────────────────────────────┐
│                    Camada de Aplicação                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Amazon ECS / Fargate (Auto-scaling)          │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │  │
│  │  │  API 1   │  │  API 2   │  │  API N   │          │  │
│  │  │ FastAPI  │  │ FastAPI  │  │ FastAPI  │          │  │
│  │  │ ONNX RT  │  │ ONNX RT  │  │ ONNX RT  │          │  │
│  │  │ :8000    │  │ :8000    │  │ :8000    │          │  │
│  │  └──────────┘  └──────────┘  └──────────┘          │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────┼───────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                 Camada de Monitoramento                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Prometheus  │  │   Grafana    │  │  CloudWatch  │      │
│  │  :9090      │  │ :3000       │  │  (Alerts)    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

**Portas Utilizadas:**
- **API**: Porta 8000 (interna, exposta via ECS)
- **Prometheus**: Porta 9090 (interna, exposta via Load Balancer)
- **Grafana**: Porta 3000 (interna, exposta via Load Balancer)

#### Passo 1: Preparar Infraestrutura AWS

```bash
# Criar ECR Repository
aws ecr create-repository --repository-name medical-classification-api --region sa-east-1

# Criar ECS Cluster
aws ecs create-cluster --cluster-name medical-classification --region sa-east-1
```

#### Passo 2: Build e Push da Imagem

```bash
# Build da imagem
docker build -f docker/Dockerfile -t <account-id>.dkr.ecr.sa-east-1.amazonaws.com/medical-classification-api:latest ..

# Login no ECR
aws ecr get-login-password --region sa-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.sa-east-1.amazonaws.com

# Push da imagem
docker push <account-id>.dkr.ecr.sa-east-1.amazonaws.com/medical-classification-api:latest
```

#### Passo 3: Configurar ECS Task Definition

```json
{
  "family": "medical-classification-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "<account-id>.dkr.ecr.sa-east-1.amazonaws.com/medical-classification-api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "APP_MODEL_PATH",
          "value": "/app/treino_modelo/artifacts/model.onnx"
        },
        {
          "name": "APP_MODEL_VERSION",
          "value": "tfidf-rf-v1.0-onnx"
        }
      ]
    }
  ]
}
```

#### Passo 4: Configurar Auto-scaling

```bash
# Configurar auto-scaling (2-10 tarefas)
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id service/medical-classification/medical-classification-api \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity 2 \
  --max-capacity 10 \
  --region sa-east-1
```

#### Passo 5: Deploy do Monitoramento em AWS

```bash
# Usar Helm chart do Prometheus
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/prometheus \
  --namespace monitoring \
  --set server.service.type=LoadBalancer \
  --set server.retention=15d

# Usar Helm chart do Grafana
helm repo add grafana https://grafana.github.io/helm-charts
helm install grafana grafana/grafana \
  --namespace monitoring \
  --set persistence.enabled=true \
  --set adminPassword='secure-password' \
  --set service.type=LoadBalancer
```

### Deploy Local (Desenvolvimento/Testing)

### Pré-requisitos

- Docker instalado
- Docker Compose instalado
- Arquivo de modelo ONNX em `treino_modelo/artifacts/model.onnx`

### Passo 1: Preparar o Ambiente

```bash
# Navegar para o diretório do projeto
cd /home/gusdpaula/code-dev/MLENG_FIAP/fase_3

# Verificar se o modelo existe
ls -la treino_modelo/artifacts/model.onnx
```

### Passo 2: Iniciar os Serviços

```bash
# Iniciar todos os serviços (API, Prometheus, Grafana)
docker-compose -f docker/docker-compose.yml up -d --build
```

### Passo 3: Verificar Status

```bash
# Verificar se os containers estão rodando
docker-compose -f docker/docker-compose.yml ps

# Saída esperada:
# NAME                STATUS                  PORTS
# docker_api_1        Up                      0.0.0.0:8000->8000/tcp
# docker_grafana_1    Up                      0.0.0.0:3000->3000/tcp
# docker_prometheus_1 Up                      0.0.0.0:9090->9090/tcp
```

### Passo 4: Testar a API

```bash
# Health check
curl http://localhost:8000/health
# Resposta esperada: {"status":"ok","model_loaded":true}

# Classificação única
curl -X POST "http://localhost:8000/v1/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient with malignant neoplasm requiring surgery"}'

# Classificação em lote
curl -X POST "http://localhost:8000/v1/classify/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Medical text 1", "Medical text 2"]}'
```

### Passo 5: Acessar o Monitoramento

```bash
# Prometheus
# URL: http://localhost:9090
# Verificar targets: http://localhost:9090/targets

# Grafana
# URL: http://localhost:3000
# Usuário: admin
# Senha: admin
# Dashboard: http://localhost:3000/d/anh7s9/medical-classification-api-monitoramento
```

## 📊 Dashboard Grafana Configurado

O dashboard possui 3 painéis obrigatórios (conforme Tech Challenge):

1. **Total de Requisições (por segundo)**
   - Mostra a taxa de requisições HTTP por segundo
   - Métrica: `sum(rate(http_requests_total[5m]))`

2. **Latência de Resposta (p95)**
   - Mostra o 95º percentil do tempo de resposta
   - Métrica: `histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))`

3. **Taxa de Erro (4xx + 5xx)**
   - Mostra a taxa de erros HTTP
   - Métrica: `(sum(rate(http_requests_total{status_code=~"4.."}[5m])) + sum(rate(http_requests_total{status_code=~"5.."}[5m]))) / sum(rate(http_requests_total[5m]))`

## 🔧 Configuração do Docker Compose

### Arquivo: `docker/docker-compose.yml`

```yaml
version: "3.8"

services:
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
      - APP_MODEL_PATH=/app/treino_modelo/artifacts/model.onnx
      - APP_MODEL_VERSION=tfidf-rf-v1.0-onnx
    restart: unless-stopped
    depends_on:
      - prometheus

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ../docker/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    ports:
      - "9090:9090"
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - ../docker/grafana/provisioning:/etc/grafana/provisioning:ro
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
```

### Arquivo: `docker/prometheus.yml`

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: 'api'
    static_configs:
      - targets: ['api:8000']
```

## 🧪 Gerar Tráfego de Teste

```bash
# Classificações únicas
for i in {1..20}; do
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
for i in {1..10}; do
  curl -s -X POST "http://localhost:8000/v1/classify" \
    -H "Content-Type: application/json" \
    -d '{"text": ""}' > /dev/null
done
```

## 🛠️ Comandos Úteis

### Verificar Logs

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

### Parar Serviços

```bash
# Parar todos os serviços
docker-compose -f docker/docker-compose.yml down

# Parar e remover volumes
docker-compose -f docker/docker-compose.yml down -v
```

### Reiniciar Serviços

```bash
# Reiniciar todos os serviços
docker-compose -f docker/docker-compose.yml restart

# Reiniciar serviço específico
docker-compose -f docker/docker-compose.yml restart api
```

## 📈 Métricas Disponíveis

A API expõe as seguintes métricas em `http://localhost:8000/metrics`:

- `http_requests_total` - Contagem total de requisições HTTP
- `http_request_duration_seconds` - Duração das requisições (histograma)
- `inference_duration_seconds` - Tempo de inferência do modelo
- `classify_batch_size` - Tamanho dos batches de classificação
- `classification_by_label_total` - Contagem de classificações por label

## 🎯 Requisitos do Tech Challenge Atendidos

### Etapa 3 - Monitoramento e Observabilidade

✅ **Docker Compose funcional** - API + Prometheus + Grafana configurados
✅ **Instrumentação da API** - Métricas expostas via prometheus_client
✅ **Métricas básicas** - Tempo de requisição e contagem de chamadas
✅ **Dashboard Grafana** - 3 painéis obrigatórios implementados:
  - Total de requisições
  - Latência/tempo de resposta
  - Taxa de erro

### Etapa 4 - Otimização de Latência

✅ **Modelo otimizado** - ONNX Runtime para inferência rápida
✅ **Containerização** - API empacotada em Docker
✅ **Monitoramento** - Stack completa de observabilidade

## 🔒 Segurança Básica

### Para Ambiente de Produção

1. **Alterar senha do Grafana** (atualmente: admin/admin)
2. **Configurar firewall** para limitar acesso aos ports
3. **Usar HTTPS** com reverse proxy (nginx/traefik)
4. **Limitar recursos** dos containers
5. **Implementar autenticação** na API

### Exemplo de Limitação de Recursos

```yaml
# Adicionar ao docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 1G
```

## 🐛 Troubleshooting

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

# Testar acesso ao endpoint /metrics
curl -s http://localhost:8000/metrics
```

### Grafana não mostra dados

```bash
# Verificar se datasource está configurado
curl -s -u admin:admin http://localhost:3000/api/datasources

# Reiniciar Grafana
docker-compose -f docker/docker-compose.yml restart grafana
```

## 📝 Resumo e Conclusão

### Estratégia de Deploy Híbrida

**Desenvolvimento/Testing (Local):**
- Docker Compose para desenvolvimento rápido
- Ambiente consistente com produção
- Testes de integração e monitoramento

**Produção (AWS):**
- ECS/Fargate para escalabilidade automática
- ALB para load balancing e alta disponibilidade
- CloudWatch + Prometheus + Grafana para monitoramento completo
- Compliance HIPAA para dados médicos

### Justificativa das Escolhas

1. **API REST Real-time**: Urgência clínica exige resposta imediata
2. **Docker Compose + AWS**: Desenvolvimento local simples, produção escalável
3. **Prometheus + Grafana**: Monitoramento robusto e visualização flexível
4. **ONNX Runtime**: 94% redução em latência (5ms → 0.3ms)
5. **FastAPI**: Performance moderna com validação automática
6. **AWS**: Compliance healthcare, serviços gerenciados, presença no Brasil

### Requisitos do Tech Challenge Atendidos

✅ **Etapa 3 - Monitoramento e Observabilidade:**
- Docker Compose funcional (API + Prometheus + Grafana)
- Instrumentação da API com prometheus_client
- Métricas básicas (tempo de requisição e contagem de chamadas)
- Dashboard Grafana com 3 painéis obrigatórios

✅ **Etapa 4 - Otimização de Latência:**
- Modelo otimizado com ONNX Runtime
- Containerização em Docker
- Stack completa de observabilidade
- Estratégia de deploy em nuvem AWS

### Como Fazer Deploy

**Deploy Local (Desenvolvimento):**
```bash
docker-compose -f docker/docker-compose.yml up -d --build
```

**Deploy AWS (Produção):**
```bash
# 1. Preparar infraestrutura AWS
aws ecr create-repository --repository-name medical-classification-api
aws ecs create-cluster --cluster-name medical-classification

# 2. Build e push da imagem
docker build -f docker/Dockerfile -t <account-id>.dkr.ecr.sa-east-1.amazonaws.com/medical-classification-api:latest ..
aws ecr get-login-password --region sa-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.sa-east-1.amazonaws.com
docker push <account-id>.dkr.ecr.sa-east-1.amazonaws.com/medical-classification-api:latest

# 3. Configurar ECS e deploy
aws ecs register-task-definition --cli-input-json file://task-definition.json
aws ecs create-service --cluster medical-classification --service-name medical-classification-api
```

### Acesso aos Serviços

**Local:**
- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

**AWS:**
- API: http://<ecs-task-ip>:8000 (ou via DNS do ECS)
- Prometheus: http://<prometheus-lb>.sa-east-1.elb.amazonaws.com
- Grafana: http://<grafana-lb>.sa-east-1.elb.amazonaws.com

### Custo Total Estimado

- **Desenvolvimento**: $0 (local)
- **Produção AWS**: $190-380/mês
- **ROI**: Sistema de triagem automática reduz custos operacionais hospitalares

Este plano fornece uma estratégia completa de deploy desde desenvolvimento local até produção em nuvem, atendendo a todos os requisitos do Tech Challenge Fase 3 com justificativa técnica sólida para cada escolha arquitetural.
