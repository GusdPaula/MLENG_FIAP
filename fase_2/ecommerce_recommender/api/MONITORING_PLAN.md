# Plano de Monitoramento de Modelo

## Estado Atual

### Componentes de Monitoramento Existentes

A API de recomendação atualmente possui uma base sólida para monitoramento com os seguintes componentes:

1. **Detecção de Mudança de Dados** (`DataShiftDetector`)
   - Usa o teste Kolmogorov-Smirnov (KS) para detectar mudanças de distribuição nos scores de predição
   - Limiar de p-value configurável (padrão: 0.05)
   - Requer que métricas de baseline sejam definidas

2. **Detecção de Deriva de Desempenho** (`ModelPerformanceMonitor`)
   - Rastreia estatísticas de predição (média, desvio padrão, mínimo, máximo, contagem)
   - Usa detecção de deriva baseada em z-score (limiar padrão: 2.0)
   - Abordagem de janela deslizante (padrão: 1000 predições)
   - Requer que baseline seja definido

3. **Serviço de Monitoramento** (`MonitoringService`)
   - Orquestra detecção de mudança de dados e deriva de desempenho
   - Registra predições com timestamps, IDs de usuário, IDs de item
   - Fornece interface unificada para operações de monitoramento

4. **Integração com API**
   - Monitoramento habilitado por padrão em `PredictionService`
   - Endpoints para definir baselines: `POST /monitoring/baselines`
   - Endpoints para verificar mudanças: `GET /monitoring/check`
   - Endpoints para resumo: `GET /monitoring/summary`
   - Registro automático de predições nos métodos de serviço

## O Que Falta para Monitoramento em Produção

### 1. **Armazenamento Persistente**
**Status**: ❌ Ausente
- Implementação atual usa apenas armazenamento em memória
- Dados são perdidos quando o serviço reinicia
- Sem rastreamento histórico além da janela deslizante

**Necessário**:
- Banco de dados ou banco de dados de séries temporais para armazenamento de métricas (ex: Prometheus, InfluxDB, PostgreSQL)
- Armazenamento persistente para baselines e dados históricos
- Políticas de retenção de dados

### 2. **Sistema de Alertas**
**Status**: ❌ Ausente
- Sem alertas automatizados quando mudanças são detectadas
- Mudanças são registradas mas não propagadas para canais de alerta
- Sem políticas de escalonamento

**Necessário**:
- Integração com gerenciador de alertas (ex: Prometheus Alertmanager, PagerDuty, Opsgenie)
- Notificações por email/SMS/Slack para alertas críticos
- Níveis de severidade de alerta e regras de roteamento
- Supressão e deduplicação de alertas

### 3. **Integração de Métricas Externas**
**Status**: ❌ Ausente
- Sem integração com ferramentas de monitoramento padrão (Prometheus, Grafana)
- Sem exposição de métricas personalizadas
- Sem visualização de dashboard

**Necessário**:
- Endpoint de métricas Prometheus (`/metrics`)
- Métricas personalizadas para:
  - Taxa de requisições e latência
  - Taxas de erro por endpoint
  - Distribuições de scores de predição
  - Cobertura de usuário/item
  - Rastreamento de versão de modelo
- Dashboards Grafana para visualização

### 4. **Métricas de Negócio**
**Status**: ❌ Ausente
- Apenas métricas técnicas (scores de predição) são rastreadas
- Sem KPIs de negócio (taxa de conversão, taxa de cliques, impacto de receita)
- Sem integração com testes A/B

**Necessário**:
- Rastreamento de métricas de negócio (ex: CTR de recomendação, taxa de adição ao carrinho)
- Rastreamento de importância de features
- Métricas de diversidade de recomendação
- Rastreamento de usuário/item cold-start

### 5. **Gerenciamento Automatizado de Baseline**
**Status**: ⚠️ Apenas manual
- Baselines devem ser definidos manualmente via endpoint de API
- Sem atualizações automáticas de baseline
- Sem testes A/B para validação de baseline

**Necessário**:
- Cálculo automático de baseline a partir de dados históricos
- Atualizações agendadas de baseline (ex: diária/semanal)
- Versionamento de baseline e capacidade de rollback
- Validação estatística antes de atualizações de baseline

### 6. **Rastreamento de Versão de Modelo**
**Status**: ⚠️ Básico
- Metadados do modelo armazenados no serviço
- Sem comparação de desempenho do modelo entre versões
- Sem capacidade de deployment shadow

**Necessário**:
- Registro de versão de modelo
- Comparação de desempenho entre versões de modelo
- Suporte a deployment canary
- Monitoramento de rollout gradual

### 7. **Monitoramento em Tempo Real**
**Status**: ⚠️ Apenas em lote
- Predições registradas em lotes
- Sem métricas de streaming em tempo real
- Sem detecção instantânea de anomalias

**Necessário**:
- Pipeline de métricas em tempo real (ex: Kafka, Redis Streams)
- Detecção de anomalias em streaming
- Alertas em tempo real para quedas súbitas

### 8. **Monitoramento de Qualidade de Dados**
**Status**: ❌ Ausente
- Sem monitoramento da qualidade dos dados de entrada
- Sem rastreamento de distribuição de features
- Sem detecção de valores ausentes/outliers

**Necessário**:
- Verificações de qualidade de dados de entrada
- Monitoramento de distribuição de features
- Rastreamento de taxa de valores ausentes
- Detecção de outliers para IDs de usuário/item

### 9. **Monitoramento de Saúde do Sistema**
**Status**: ❌ Ausente
- Sem monitoramento de recursos do sistema
- Sem verificações de saúde de dependências
- Sem rastreamento de latência

**Necessário**:
- Métricas do sistema (CPU, memória, utilização de GPU)
- Saúde da conexão com banco de dados
- Verificações de saúde de serviços externos
- Rastreamento de latência de requisição (p50, p95, p99)

### 10. **Conformidade e Registro de Auditoria**
**Status**: ❌ Ausente
- Sem rastro de auditoria para predições
- Sem relatórios de conformidade
- Sem rastreamento de linhagem de dados

**Necessário**:
- Logs de auditoria de predições
- Explicabilidade de decisões do modelo
- Monitoramento de justiça e viés
- Conformidade de retenção de dados e privacidade

## Roadmap de Implementação

### Fase 1: Fundação (Semana 1-2)
- ✅ Implementar endpoint de métricas Prometheus
- ✅ Adicionar monitoramento de recursos do sistema
- ✅ Configurar dashboards Grafana
- ✅ Implementar armazenamento persistente de métricas (PostgreSQL/TimescaleDB)

### Fase 2: Alertas (Semana 3)
- ⏳ Integrar Alertmanager
- ⏳ Configurar notificações por email/Slack
- ⏳ Definir regras de alerta e níveis de severidade
- ⏳ Implementar lógica de supressão de alertas

### Fase 3: Monitoramento Avançado (Semana 4-6)
- ⏳ Adicionar rastreamento de métricas de negócio
- ⏳ Implementar monitoramento de qualidade de dados
- ⏳ Adicionar métricas de streaming em tempo real
- ⏳ Implementar gerenciamento automatizado de baseline

### Fase 4: Ciclo de Vida do Modelo (Semana 7-8)
- ⏳ Construir registro de versão de modelo
- ⏳ Implementar monitoramento de deployment canary
- ⏳ Adicionar integração com testes A/B
- ⏳ Implementar relatórios de conformidade

## Recomendações de Configuração

### Variáveis de Ambiente
```bash
# Configuração de Monitoramento
MONITORING_ENABLED=true
MONITORING_STORAGE_TYPE=postgresql|influxdb|prometheus
MONITORING_STORAGE_URL=postgresql://user:pass@localhost:5432/monitoring
MONITORING_RETENTION_DAYS=90

# Configuração de Alertas
ALERT_ENABLED=true
ALERT_WEBHOOK_URL=https://hooks.slack.com/services/...
ALERT_EMAIL_RECIPIENTS=team@company.com
ALERT_SEVERITY_THRESHOLD=warning

# Limitares
SHIFT_THRESHOLD=0.05
DRIFT_THRESHOLD=2.0
MONITORING_WINDOW_SIZE=1000

# Prometheus
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
```

### Limitares Recomendados
- **Mudança de Dados**: p-value < 0.05 (estatisticamente significativo)
- **Deriva de Desempenho**: z-score > 2.0 (2 desvios padrão)
- **Taxa de Erro**: > 5% para aviso, > 10% para crítico
- **Latência**: p95 > 500ms para aviso, > 1s para crítico
- **Taxa de Predição**: Queda > 50% para alerta imediato

## Métricas do Dashboard de Monitoramento

### Métricas Técnicas
1. **Métricas de Requisição**
   - Requisições por segundo (RPS)
   - Latência de requisição (p50, p95, p99)
   - Taxa de erro por endpoint
   - Tamanho da requisição

2. **Métricas de Predição**
   - Distribuição de scores de predição
   - Taxa de predição por usuário/item
   - Cobertura (usuários/itens únicos atendidos)
   - Percentis de score

3. **Métricas de Modelo**
   - Versão do modelo
   - Tempo de carregamento do modelo
   - Tempo de predição por requisição
   - Utilização de memória/CPU/GPU

4. **Métricas de Sistema**
   - Status de saúde do serviço
   - Pool de conexão com banco de dados
   - Taxa de acerto de cache
   - Utilização do pool de threads

### Métricas de Negócio
1. **Métricas de Engajamento**
   - Taxa de cliques (CTR)
   - Taxa de adição ao carrinho
   - Taxa de conversão
   - Valor médio do pedido

2. **Qualidade de Recomendação**
   - Diversidade de recomendação
   - Score de novidade
   - Serendipidade
   - Cobertura

3. **Métricas de Usuário**
   - Usuários ativos
   - Novos usuários
   - Retenção de usuários
   - Duração da sessão

## Cenários de Alerta

### Alertas Críticos (Ação Imediata Necessária)
- Serviço de modelo fora (taxa de erro > 50%)
- Deriva de desempenho > 5 desvios padrão
- Mudança de dados detectada com p-value < 0.01
- Latência de predição p99 > 5 segundos
- Falhas de conexão com banco de dados

### Alertas de Aviso (Investigar Dentro de 1 Hora)
- Deriva de desempenho > 2 desvios padrão
- Mudança de dados detectada com p-value < 0.05
- Taxa de erro > 10%
- Latência p95 > 1 segundo
- Queda na taxa de predição > 30%

### Alertas de Informação (Monitorar Tendência)
- Baseline precisa de atualização
- Modelo aproximando fim de vida útil
- Degradação gradual de desempenho
- Padrões de tráfego incomuns

## Próximos Passos

1. **Imediato (Esta Semana)**
   - Adicionar endpoint de métricas Prometheus ao app FastAPI
   - Configurar dashboards básicos do Grafana
   - Configurar agregação de logs (stack ELK ou similar)

2. **Curto Prazo (Próximas 2 Semanas)**
   - Implementar armazenamento persistente para métricas
   - Adicionar integração de alertas
   - Criar endpoints de saúde do sistema

3. **Médio Prazo (Próximo Mês)**
   - Adicionar rastreamento de métricas de negócio
   - Implementar monitoramento de qualidade de dados
   - Configurar gerenciamento automatizado de baseline

4. **Longo Prazo (Próximo Trimestre)**
   - Construir registro de versão de modelo
   - Implementar monitoramento de deployment canary
   - Adicionar conformidade e registro de auditoria
