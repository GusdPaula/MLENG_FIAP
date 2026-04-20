# Plano de Monitoramento - Telco Churn Prediction

**Data**: 2026-04-19  
**Versão**: 1.0  
**Dono**: ML + Platform Team

---

## 1. VISÃO GERAL

### Objetivo
Estabelecer um sistema integrado de monitoramento que:
- ✅ Detecte degradação de modelo em tempo real
- ✅ Rastreie saúde da infraestrutura 24/7
- ✅ Gere alertas acionáveis para diferentes stakeholders
- ✅ Forneça dados para análise de root cause
- ✅ Cumpra SLOs (Service Level Objectives) de negócio

### Pillars do Monitoramento

```
┌────────────────────────────────────────────────────┐
│         TELEMETRIA COMPLETA (3-PILLAR MODEL)       │
├────────────────────────────────────────────────────┤
│                                                    │
│  LOGS        METRICS        TRACES                │
│  ═══════     ══════════     ══════                │
│  • Eventos   • Performance  • Requests            │
│  • Erros     • Business KPIs• Latência            │
│  • Auditoria • Model drift  • Dependencies        │
│                                                    │
│  Ferramentas: CloudWatch Logs + Metrics + X-Ray   │
│                                                    │
└────────────────────────────────────────────────────┘
```

---

## 2. SLOs (SERVICE LEVEL OBJECTIVES)

### 2.1 SLOs de Disponibilidade

| SLO | Target | Janela | Orçamento de Erro | Ação |
|-----|--------|--------|-------------------|------|
| **Uptime API** | 99.9% | Mensal | 2.16 horas | PagerDuty |
| **Uptime Batch** | 99% | Mensal | 7.2 horas | Manual escalation |
| **Latência P95** | <200ms | Diariamente | 5% acima | Investigar |
| **Latência P99** | <500ms | Diariamente | 10% acima | Investigar |

### 2.2 SLOs de Qualidade de Modelo

| SLO | Target | Janela | Ação |
|-----|--------|--------|------|
| **F1-Score** | ≥0.67 | Mensal | Se <0.60 → retraining urgente |
| **Precision** | ≥65% | Mensal | Se <55% → alert na reunião |
| **Recall** | ≥66% | Mensal | Se <55% → campanha impactada |
| **AUC-ROC** | ≥0.84 | Mensal | Baseline de comparação |
| **Feature Drift** | KL divergence <0.1 | Semanal | Se >0.1 → retraining |

### 2.3 SLOs de Negócio

| KPI | Target | Janela | Baseline |
|-----|--------|--------|----------|
| **Churn Rate** | -5 a -10% | Trimestral | 26.5% (atual) |
| **Campaign Conversion** | 40%+ | Trimestral | TBD |
| **ROI Retenção** | >150% | Anual | TBD |

---

## 3. MÉTRICAS A MONITORAR

### 3.1 Métricas de Infraestrutura

#### EC2 / Application Servers

```yaml
Métricas Chave:
  - CPU Utilization: Target <70% (alerta >85%)
  - Memory Usage: Target <80% (alerta >90%)
  - Network In/Out: Monitor picos
  - Disk Space: Alerta >80% (RDS, logs)
  - Process Health: Gunicorn workers status
  
CloudWatch:
  - Namespace: AWS/EC2
  - Period: 1 minute
  - Retention: 7 days (detailed), 60 days (1-min aggregate)
```

#### RDS Database

```yaml
Métricas Chave:
  - Database Connections: <100 (alerta >80%)
  - CPU Utilization: <70% (alerta >80%)
  - Disk Space Free: >20GB (alerta <10GB)
  - Read/Write Latency: <50ms (alerta >100ms)
  - Query Performance Insights: Slow queries
  
Backup:
  - Automated backup running: Daily 23:00
  - Backup size: Monitor crescimento
  - Restore time test: Monthly
```

#### ElastiCache (Redis)

```yaml
Métricas Chave:
  - CPU Utilization: <70%
  - Memory Usage: <80% (eviction threshold)
  - Cache Hit Rate: >80% (alerta <60%)
  - Evictions: Monitor (se >0, significa cache cheio)
  - Commands Latency: <5ms
```

### 3.2 Métricas de Aplicação

#### Request-Level Metrics

```yaml
Endpoint: /api/predict
  - Request Rate: requisições/minuto
  - Latency Distribution:
    - P50: ~80ms
    - P95: <200ms (target)
    - P99: <500ms (target)
  - Error Rate: 4xx, 5xx
  - Status Codes: 200, 400, 422, 500, 503
  
Endpoint: /api/predict-batch
  - Request Rate: requisições/minuto
  - Latency Distribution: P95 <2s (1000 amostras)
  - Throughput: amostras/min
  - Queue Depth: Se batch queue > 100 → alerta

Endpoint: /health
  - Response Time: <50ms
  - Uptime %: 99.9%
  - Availability: Check a cada 30 segundos (ALB)
```

#### Model Inference Metrics

```yaml
Prediction Service:
  - Model Load Time: <100ms (inicio)
  - Inference Time: <150ms (p95)
  - Feature Transformation: <50ms
  - Scaler Transformation: <10ms
  - Model Forward Pass: <80ms
  
Batch Processing:
  - Batch Size: 1000-5000 amostras
  - Total Batch Time: <5 minutos (target)
  - Throughput: amostras/segundo
  - Memory Peak: Monitor (modelo + data in RAM)
```

#### Cache Metrics

```yaml
Redis Cache:
  - Hit Rate: >80%
  - Misses: <20%
  - Cache Size: Monitor (eviction se full)
  - TTL Effectiveness: Evictions/min
  
Metrics Cached:
  - Model scaler coefficients
  - Feature names list
  - Recent predictions (last 24h)
```

### 3.3 Métricas de Modelo

#### Feature Distribution

```yaml
Monitoramento Semanal:
  - Gender: % male/female (baseline vs atual)
  - Senior Citizen: % ≥65 years (baseline vs atual)
  - Tenure: median, p25, p75 (drift detection)
  - Monthly Charges: median, std dev
  - Contract Types: % distribution shift
  
Técnica: Kolmogorov-Smirnov Test
  - KL divergence < 0.05: OK
  - KL divergence 0.05-0.1: Warning
  - KL divergence > 0.1: Alert (retraining)
```

#### Prediction Distribution

```yaml
Monitoramento Diário:
  - % Churn Predicted: Target ~27% (match real)
  - Avg Probability: Should be ~0.27
  - Calibration: Plot predicted vs actual probability
  - Score Distribution: Histogram (bimodal esperado)
  
Alert Triggers:
  - % Churn Predicted > 40%: Something wrong
  - % Churn Predicted < 15%: Model too conservative
```

#### Model Performance Validation

```yaml
Sliding Window (últimos 1000 predictions):
  - Precision: Current vs baseline
  - Recall: Current vs baseline
  - F1-Score: Current vs baseline
  - Confusion Matrix: TP, FP, TN, FN
  
Frequency: Hourly computation, daily alert
Alert Threshold:
  - F1 < 0.60: Urgent investigation
  - Precision < 55%: Review needed
  - Recall < 55%: Campaign impact
```

### 3.4 Métricas de Negócio / Impacto

```yaml
Campanha de Retenção:
  - # Clientes alertados: Diariamente
  - # Clientes contatados: Diariamente (integração CRM)
  - # Clientes retidos: Semanalmente (feedback)
  - Conversion Rate: % retidos / alertados
  - ROI: (Receita preservada - Custo campanha) / Custo
  
Data Quality:
  - # Clientes com dados completos: 95%+
  - # Missing values: Monitor por feature
  - # Outliers: Monitor (pode indicar data issue)
  - Schema violations: Alertar (tipos incorretos)
```

---

## 4. DASHBOARD DE MONITORAMENTO

### 4.1 Dashboard Real-Time (Atualizado a cada 5s)

```
╔════════════════════════════════════════════════════════════════╗
║        TELCO CHURN - MONITORING DASHBOARD (Real-Time)          ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  API Health Status                                             ║
║  ✅ UP  | Requests: 245/min | Errors: 0.2% | P95: 145ms      ║
║                                                                ║
║  Infrastructure                                                ║
║  ┌──────────┬──────────┬──────────┬──────────┐                ║
║  │ CPU: 42% │ Mem: 65% │ DB Con:24│ Cache Hit│                ║
║  │ ✅ OK   │ ✅ OK   │ ✅ OK   │ 87%     │                ║
║  └──────────┴──────────┴──────────┴──────────┘                ║
║                                                                ║
║  Recent Predictions (Last 1 Hour)                              ║
║  Total: 14,532 | Churn: 3,930 (27.0%) | Avg Prob: 0.268     ║
║  ├─ Latency: P50 82ms, P95 156ms, P99 342ms ✅               ║
║  └─ Model: v1.0 | Uptime: 99.97% | Last Updated: 2h ago     ║
║                                                                ║
║  Alerts                                                        ║
║  ℹ️  [14:23] Model Feature Drift: KL divergence 0.087 (OK)  ║
║  ✅ [14:00] Batch job completed: 7000 predictions             ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

### 4.2 Dashboard Operacional (Atualizado a cada 1m)

```
Tabs:
├─ System Health
│  ├─ EC2 CPU/Memory timeline (7 dias)
│  ├─ RDS connections, latency (7 dias)
│  ├─ Network In/Out (1 dia)
│  └─ Disk space trends
│
├─ API Performance
│  ├─ Request rate (current, hourly, daily)
│  ├─ Latency percentiles (P50, P95, P99, P99.9)
│  ├─ Error rate by endpoint
│  ├─ Status code distribution
│  └─ Cache hit rate
│
├─ Model Quality
│  ├─ F1-Score trend (daily)
│  ├─ Precision/Recall trend
│  ├─ Feature distribution (vs baseline)
│  ├─ Prediction distribution
│  └─ Calibration plot
│
├─ Business Metrics
│  ├─ Churn predictions: daily, cumulative
│  ├─ Campaign send rate
│  ├─ Conversion rate (if CRM integrated)
│  └─ ROI calculation
│
└─ Alerts & Issues
   ├─ Active incidents
   ├─ Resolved incidents (last 7 days)
   ├─ Trend of issue types
   └─ MTTR (Mean Time to Resolve)
```

### 4.3 Dashboard Executivo (Atualizado a cada 15m)

```
KPIs Summary:
┌─────────────────────────────────────────┐
│ Model Performance      │ This Month     │
│ ─────────────────────────────────────  │
│ F1-Score         │ 0.67 (✅ on target) │
│ Precision        │ 68%  (✅ >65%)      │
│ Recall           │ 66%  (⚠️  <75%)     │
│ AUC-ROC          │ 0.844 (✅ >0.84)    │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Service SLOs                           │
│ ─────────────────────────────────────  │
│ Uptime           │ 99.94% (✅ >99.9%)  │
│ Latency P95      │ 156ms (✅ <200ms)   │
│ Error Rate       │ 0.2%  (✅ <0.5%)    │
│ Data Freshness   │ <1h   (✅ <24h)     │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Business Impact (This Week)            │
│ ─────────────────────────────────────  │
│ Predictions Made │ 104,640             │
│ Campaigns Sent   │ 28,125              │
│ Conv. Rate       │ 38%                 │
│ Est. Revenue Saved │ $2,850            │
└─────────────────────────────────────────┘
```

---

## 5. ALERTAS & ESCALAÇÃO

### 5.1 Configuração de Alertas

#### Alert Severity Levels

```
┌─────────────┬─────────────────────┬─────────────┬──────────────┐
│ Severity    │ Impact              │ Action      │ Response     │
├─────────────┼─────────────────────┼─────────────┼──────────────┤
│ CRITICAL    │ Service down        │ PagerDuty   │ Immediate    │
│             │ Data loss imminent  │ SMS + call  │ <5 min       │
│             │ Security breach     │ Escalation  │              │
├─────────────┼─────────────────────┼─────────────┼──────────────┤
│ HIGH        │ Degraded service    │ Slack #ops  │ <30 min      │
│             │ Model drift         │ PagerDuty   │              │
│             │ Performance issue   │ on-call     │              │
├─────────────┼─────────────────────┼─────────────┼──────────────┤
│ MEDIUM      │ Monitoring issue    │ Slack #alerts│<2 hours     │
│             │ Feature drift       │ Jira ticket │              │
│             │ Cache hit low       │             │              │
├─────────────┼─────────────────────┼─────────────┼──────────────┤
│ LOW         │ Informational       │ Slack #logs │ Async        │
│             │ Trends to watch     │             │              │
└─────────────┴─────────────────────┴─────────────┴──────────────┘
```

### 5.2 Alertas Específicos

#### Infrastructure Alerts

| Alert | Condition | Threshold | Severity | Action | Recipient |
|-------|-----------|-----------|----------|--------|-----------|
| **API Down** | Health check fails | 2 consecutive | CRITICAL | Pagerduty | on-call |
| **High CPU** | EC2 CPU > 85% for 5min | 85% | HIGH | Slack #ops | ops-team |
| **Memory Pressure** | Free mem < 500MB | <500MB | HIGH | Slack | ops-team |
| **DB Connections** | > 80 active | 80 | HIGH | Slack | db-admin |
| **RDS Disk Full** | Free space < 10GB | <10GB | CRITICAL | Pagerduty | db-admin |
| **Redis Eviction** | Evictions/sec > 0 | >0 | MEDIUM | Slack | ops-team |
| **Batch Failed** | Lambda execution error | Any | HIGH | Email | ml-lead |

#### Model Quality Alerts

| Alert | Condition | Threshold | Severity | Action | Recipient |
|-------|-----------|-----------|----------|--------|-----------|
| **F1 Degradation** | F1 < 0.60 vs baseline | <0.60 | CRITICAL | Slack #ml | ML team |
| **Feature Drift** | KL divergence > 0.1 | >0.1 | HIGH | Slack #ml | ML team |
| **Calibration Fail** | ECE > 0.15 | >0.15 | MEDIUM | Slack #ml | ML team |
| **Recall Drop** | Recall < 55% | <55% | HIGH | Slack #ml | ML team |
| **Precision Drop** | Precision < 55% | <55% | MEDIUM | Slack | product |
| **Pred Distribution** | % Churn pred > 40% | >40% | HIGH | Slack #ml | ML team |
| **No Predictions** | 0 predictions in 1h | 0 | CRITICAL | Pagerduty | on-call |

#### Application Alerts

| Alert | Condition | Threshold | Severity | Action | Recipient |
|-------|-----------|-----------|----------|--------|-----------|
| **High Error Rate** | Errors > 1% for 5min | >1% | HIGH | Pagerduty | on-call |
| **Latency P95 High** | P95 > 500ms | >500ms | MEDIUM | Slack | platform |
| **Latency P99 High** | P99 > 1s | >1s | MEDIUM | Slack | platform |
| **400 Errors Spike** | 4xx > 10%/min | >10% | MEDIUM | Slack | platform |
| **5xx Errors** | Any 5xx error | >0 | HIGH | Slack | platform |
| **API Timeout** | Request > 30s | >30s | HIGH | Pagerduty | on-call |
| **Queue Depth** | Batch queue > 1000 | >1000 | MEDIUM | Slack | ops |

### 5.3 Escalação & Runbook

```
Alert: F1-Score < 0.60

Level 1 (ML Engineer - 15 min):
  1. Check CloudWatch logs para últimas 24h
  2. Compare prediction distribution vs baseline
  3. Verificar se houve data drift
  4. Gather info para level 2
  
Level 2 (ML Lead - 30 min):
  1. Review model evaluation metrics
  2. Compare feature distributions (semanal)
  3. Decide: Retraining vs Rollback v0.9
  4. Communicate timeline ao product
  
Level 3 (VP Engineering - 1h):
  1. Assess business impact (campaigns afetadas?)
  2. Approve emergency retraining
  3. Notify stakeholders
  4. Schedule post-incident review

Rollback:
  - Time to rollback: <5 min (blue-green deployment)
  - Trigger: If F1 < 0.55 (emergency)
  - Notification: All stakeholders
```

---

## 6. PLAYBOOKS (Response Procedures)

### Playbook 1: API Latency High (P95 > 500ms)

```
Detection: CloudWatch Alert triggered
Time: ~2 min to alert

Step 1: Diagnostics
  □ Check EC2 CPU utilization (if >85%, increase ASG)
  □ Check RDS connections (if >80, investigate queries)
  □ Check Redis cache hit rate (if <70%, issue?)
  □ Review slow query logs
  
Step 2: Mitigation
  if CPU > 85%:
    → Manually trigger ASG scale out (+1 instance)
    → Should help in 2-3 min
  if DB slow:
    → Check for long-running query
    → Kill if necessary, identify query plan
  if cache miss:
    → Restart Redis, check hot keys
    
Step 3: Prevention
  → Add to capacity planning
  → Implement query optimization
  → Increase cache TTL if appropriate

Communication:
  → Slack #ops immediately
  → Update status page
  → Notify Product if impacting users
```

### Playbook 2: Model F1-Score < 0.60

```
Detection: CloudWatch Alert triggered
Time: ~1 day (detected at next daily validation)

Step 1: Root Cause Analysis
  □ Feature Drift Analysis
    - KL divergence per feature (vs training set)
    - Identify which features drifted most
  □ Prediction Analysis
    - Is model predicting all 0s? All 1s?
    - Check calibration (predicted prob vs actual)
  □ Data Quality Check
    - % missing values per feature
    - Outliers introduced
    - Schema violations
    
Step 2: Decision Tree
  if Feature Drift > 0.15:
    → Retraining required (2-3h)
    → Gather new data, retrain, validate
    → Deploy v1.1 canary (10% traffic)
    
  else if Data Quality Issue:
    → Investigate data pipeline
    → Fix upstream data source
    → Resume monitoring
    
  else if Calibration issue only:
    → Can wait until next scheduled retraining
    → Monitor weekly
    
  else (unknown):
    → Escalate to ML Lead
    → Run deeper diagnostics
    
Step 3: Implement Fix
  → Execute chosen path
  → Validate on staging
  → Deploy with canary (if model change)
  → Monitor 48h closely
  
Step 4: Communication
  → Slack #ml team immediately
  → Daily standup: status update
  → Retrospective: what caused drift?
```

### Playbook 3: RDS Disk Full (< 10GB free)

```
Detection: CloudWatch metric < 10GB

Step 1: Immediate
  □ Alert on-call DBA
  □ Check log files size (CloudWatch, RDS logs)
  □ Check backup size (may be stored locally)
  
Step 2: Quick Mitigation
  if Logs too large:
    → Purge old logs: DELETE FROM logs WHERE date < now() - 30d
    → Estimated recovery: 5-10GB
    
  if Backups stale:
    → Delete old backup snapshots
    → AWS automatic backup retention policy
    
Step 3: Longer Term
  → Increase RDS disk size (needs ~30min downtime)
  → Monitor growth rate
  → Implement log retention policy
  
Step 4: Prevention
  → Set alert for <20GB
  → Monitor growth trend monthly
```

---

## 7. RETENTION POLICY (Histórico de Dados)

### Log Retention

```
CloudWatch Logs:
├─ Application logs: 30 days online, archive to S3
├─ Prediction logs: 90 days online, archive to S3
├─ Error logs: 365 days (S3 archived)
└─ Audit logs: 7 years (compliance)

S3 Archives:
├─ Compress with gzip (90% size reduction)
├─ Store in S3 Glacier for cost savings
├─ Enable versioning for compliance
```

### Metrics Retention

```
CloudWatch Metrics:
├─ Detailed (1-min granularity): 15 days
├─ Standard (5-min): 63 days
├─ Long-term (1-hour): 455 days
└─ Custom dashboards: Alerting data 30 days
```

### Database Retention

```
RDS Tables:
├─ predictions: Keep last 90 days (query performance)
├─ prediction_history: Keep all (audit)
├─ customer_features: Keep last 30 days (current state)
└─ alerts: Keep last 90 days
```

---

## 8. INTEGRAÇÃO COM FERRAMENTAS

### 8.1 Ferramentas Utilizadas

| Componente | Ferramenta | Função |
|-----------|-----------|--------|
| **Logs** | CloudWatch Logs | Centralizar todos os logs |
| **Metrics** | CloudWatch Metrics | Rastrear KPIs |
| **Alerts** | CloudWatch Alarms | Notificações automáticas |
| **Dashboard** | Grafana (ou CloudWatch native) | Visualização |
| **Tracing** | AWS X-Ray | Distributed tracing |
| **Incident Mgmt** | PagerDuty | Escalação on-call |
| **Notifications** | Slack + SMS + Email | Multi-channel alerts |
| **Error Tracking** | Sentry | Exception monitoring |

### 8.2 Integração com Slack

```
Slack Channels:
├─ #alerts: Críticos + Highs (PagerDuty posts)
├─ #logs: Medium + Low severity
├─ #ml-team: Model quality alerts
├─ #ops: Infrastructure alerts
├─ #incidents: Incident declarations + status
└─ #metrics: Daily summary bot

Slack Apps:
- CloudWatch integration
- PagerDuty integration
- Custom status updates bot
```

---

## 9. REVIEW & OTIMIZAÇÃO

### 9.1 Reuniões de Monitoramento

```
Daily Standup (15 min):
  - Qualquer alerta overnight?
  - Sistema verde? Ou investigações em andamento?
  
Weekly Review (30 min):
  - Tendências da semana
  - Qualquer padrão emergente
  - Alertas falsas (tuning necessário?)
  
Monthly Deep Dive (60 min):
  - Model performance review
  - Infrastructure capacity planning
  - Alert tuning (reduzir false positives)
  - Roadmap otimizações
```

### 9.2 Métricas de Efetividade do Monitoramento

```
Rastreamento:
- MTTR (Mean Time To Resolve): Target <30 min
- MTTD (Mean Time To Detect): Target <5 min
- False Positive Rate: Target <10%
- Alert Response Rate: Target >95%

Goal: Detectar 95% dos problemas antes que afetem clientes
```

---

## 10. ROADMAP

### Q2 2026

- [ ] CloudWatch dashboards finalizados
- [ ] Integração PagerDuty full
- [ ] Alertas de feature drift implementados
- [ ] SLO tracking automático

### Q3 2026

- [ ] Grafana dashboards + alertas
- [ ] Distributed tracing (X-Ray) completo
- [ ] Custom metrics para business KPIs
- [ ] Automated runbooks (Lambda-based)

### Q4 2026

- [ ] AIOps: ML-based anomaly detection
- [ ] Predictive alerting (antes do problema)
- [ ] Self-healing infrastructure
- [ ] Advanced incident correlation

---

## 11. ASSINATURAS

| Papel | Nome | Data | Status |
|-------|------|------|--------|
| **ML Lead** | - | - | ____ |
| **Ops Lead** | - | - | ____ |
| **CTO** | - | - | ____ |

---

**Data**: 2026-04-19  
**Versão**: 1.0  
**Próxima Revisão**: 2026-05-19
