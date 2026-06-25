# Grafana Integration Plan

## Overview

This document outlines the plan to integrate Grafana for comprehensive monitoring and log tracking of the ML recommendation system infrastructure.

## Current Infrastructure Analysis

### Existing Logging & Monitoring

**API Module (ECS Fargate)**
- CloudWatch Logs group: `/ecs/${project_name}-api`
- Retention: 7 days
- ECS service metrics available via CloudWatch
- ALB access logs and metrics
- CloudFront distribution logs and metrics

**MLflow Module (EC2 + RDS)**
- VPC Flow Logs to CloudWatch (7-day retention)
- EC2 instance metrics via CloudWatch
- RDS PostgreSQL metrics and slow query logs
- ALB access logs and metrics
- CloudFront distribution logs and metrics

**S3 Module**
- DVC bucket access logs (if enabled)
- MLflow artifacts bucket access logs (if enabled)

**Application-Level Monitoring**
- Built-in drift detection in PredictionService
- Model performance monitoring
- Data shift detection
- Python logging throughout the application

## Proposed Grafana Architecture

### Option 1: AWS Managed Grafana (Recommended)

**Pros:**
- Fully managed by AWS
- Automatic updates and patches
- Integrated with AWS IAM
- No infrastructure to manage
- SSO integration available
- Cost-effective for small teams

**Cons:**
- Limited customization compared to self-hosted
- Regional availability constraints

### Option 2: Self-Hosted Grafana on ECS

**Pros:**
- Full control over configuration
- Can use any plugins
- Can be deployed in same VPC
- No regional constraints

**Cons:**
- Need to manage infrastructure
- Need to handle updates and patches
- Need to set up authentication
- Additional operational overhead

### Option 3: Self-Hosted Grafana on EC2

**Pros:**
- Full control
- Can use any plugins
- Simple deployment

**Cons:**
- Single point of failure
- Need to manage infrastructure
- Need to handle high availability setup

## Recommended Architecture: AWS Managed Grafana

### Components

1. **AWS Managed Grafana Workspace**
   - Region: us-east-1 (same as infrastructure)
   - Authentication: AWS IAM or SAML SSO
   - Data sources: CloudWatch Logs, CloudWatch Metrics, X-Ray

2. **Data Sources**
   - CloudWatch Logs (for log aggregation and search)
   - CloudWatch Metrics (for infrastructure metrics)
   - AWS X-Ray (for distributed tracing - optional)
   - Prometheus (for custom application metrics - optional)

3. **Dashboards**
   - API Overview Dashboard
   - MLflow Dashboard
   - Infrastructure Dashboard
   - Model Performance Dashboard
   - Security Dashboard

## Implementation Plan

### Phase 1: Infrastructure Setup

#### 1.1 Create Grafana Workspace

```hcl
# infra-api/grafana/main.tf
resource "aws_grafana_workspace" "main" {
  name              = "${var.project_name}-grafana"
  authentication_providers = ["AWS_SSO"]
  permission_type   = "SERVICE_MANAGED"
  role_arn          = aws_iam_role.grafana.arn

  data_sources {
    name = "CloudWatch"
    type = "cloudwatch"
  }
}
```

#### 1.2 IAM Role for Grafana

```hcl
resource "aws_iam_role" "grafana" {
  name = "${var.project_name}-grafana-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "grafana.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "grafana_cloudwatch" {
  role       = aws_iam_role.grafana.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchReadOnlyAccess"
}

resource "aws_iam_role_policy_attachment" "grafana_xray" {
  role       = aws_iam_role.grafana.name
  policy_arn = "arn:aws:iam::aws:policy/AWSXRayReadOnlyAccess"
}
```

#### 1.3 SSO Configuration (Optional)

```hcl
resource "aws_ssoadmin_permission_set" "grafana_admin" {
  name         = "${var.project_name}-grafana-admin"
  instance_arn = var.sso_instance_arn
}

resource "aws_ssoadmin_managed_policy_attachment" "grafana_admin" {
  managed_policy_arn = "arn:aws:iam::aws:policy/PowerUserAccess"
  permission_set_arn = aws_ssoadmin_permission_set.grafana_admin.arn
  instance_arn      = var.sso_instance_arn
}
```

### Phase 2: Data Source Configuration

#### 2.1 CloudWatch Logs Integration

Configure CloudWatch Logs as a data source in Grafana:
- Region: us-east-1
- Log groups to monitor:
  - `/ecs/${project_name}-api`
  - `${project_name}-vpc-flow-logs`
  - `/aws/lambda/*` (if using Lambda)
  - `/aws/rds/instance/${project_name}-db/error` (RDS logs)

#### 2.2 CloudWatch Metrics Integration

Configure CloudWatch Metrics for:
- ECS metrics (CPU, Memory, Network)
- ALB metrics (Request count, Latency, 5xx errors)
- CloudFront metrics (Requests, Bytes transferred, 4xx/5xx errors)
- RDS metrics (Connections, CPU, Memory, Storage)
- EC2 metrics (CPU, Memory, Network, Status checks)

#### 2.3 Custom Metrics (Optional)

Add Prometheus exporter for application-specific metrics:
- Prediction latency
- Model drift scores
- Request throughput
- Error rates
- User/item distribution

### Phase 3: Dashboard Creation

#### 3.1 API Overview Dashboard

**Panels:**
- Request rate (requests/second)
- Response time (p50, p95, p99)
- Error rate (4xx, 5xx)
- Active tasks
- CPU/Memory utilization
- Health check status
- Top endpoints by latency
- Error distribution by endpoint

**Queries:**
```
# Request rate
sum(rate(aws_applicationelb_request_count_sum[5m]))

# Response time
histogram_quantile(0.95, sum(rate(aws_applicationelb_target_response_time_sum[5m])))

# Error rate
sum(rate(aws_applicationelb_http_code_elb_5XX[5m])) / sum(rate(aws_applicationelb_request_count[5m]))
```

#### 3.2 MLflow Dashboard

**Panels:**
- Server health status
- Active experiments
- Model registry metrics
- RDS connection count
- RDS query latency
- RDS CPU utilization
- Storage usage
- API endpoint health

**Queries:**
```
# RDS CPU
aws_rds_cpuutilization_average

# RDS Connections
aws_rds_database_connections

# Storage usage
aws_rds_free_storage_space_average
```

#### 3.3 Infrastructure Dashboard

**Panels:**
- ECS cluster health
- Fargate task CPU/Memory
- EC2 instance health
- VPC flow logs analysis
- Security group rules
- ALB health
- CloudFront cache hit ratio
- S3 bucket size

**Queries:**
```
# ECS CPU
aws_ecs_cpuutilization_average

# EC2 CPU
aws_ec2_cpuutilization_average

# CloudFront cache hit ratio
sum(rate(aws_cloudfront_cache_hit_count[5m])) / sum(rate(aws_cloudfront_cache_hit_count[5m]) + rate(aws_cloudfront_cache_miss_count[5m]))
```

#### 3.4 Model Performance Dashboard

**Panels:**
- Prediction drift score over time
- Data shift detection status
- Model performance metrics
- Baseline comparison
- Alert thresholds
- Prediction distribution
- User activity heatmap
- Item popularity trends

**Data Sources:**
- Custom metrics from PredictionService
- CloudWatch Logs for drift alerts
- MLflow experiment metrics

#### 3.5 Security Dashboard

**Panels:**
- Failed authentication attempts
- Unusual access patterns
- VPC flow log anomalies
- API key usage
- Rate limiting events
- Security group changes
- S3 access patterns
- ECR image scan results

### Phase 4: Alerting Configuration

#### 4.1 Alert Channels

- Email notifications
- Slack integration
- PagerDuty (optional)
- AWS SNS integration

#### 4.2 Alert Rules

**Critical Alerts:**
- API error rate > 5%
- API response time p95 > 1s
- ECS task failures
- RDS connection failures
- Model drift detected
- Security anomalies

**Warning Alerts:**
- CPU utilization > 70%
- Memory utilization > 80%
- Storage > 80% full
- Slow query logs in RDS
- Prediction latency increase

#### 4.3 Alert Configuration Example

```json
{
  "name": "API High Error Rate",
  "conditions": [
    {
      "type": "query",
      "query": "A",
      "reducer": {
        "type": "avg"
      },
      "evaluator": {
        "params": [5],
        "type": "gt"
      }
    }
  ],
  "frequency": "1m",
  "handler": 1,
  "no_data_state": "no_data",
  "execution_error_state": "alerting"
}
```

### Phase 5: Application Integration

#### 5.1 Structured Logging

Update application logging to include structured fields:

```python
import json
import logging

class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "user_id": getattr(record, "user_id", None),
            "item_ids": getattr(record, "item_ids", None),
            "prediction_time": getattr(record, "prediction_time", None),
            "model_version": getattr(record, "model_version", None),
        }
        return json.dumps(log_data)
```

#### 5.2 Custom Metrics Export

Add Prometheus metrics endpoint:

```python
from prometheus_client import start_http_server, Counter, Histogram

# Define metrics
prediction_counter = Counter('predictions_total', 'Total predictions', ['predictor_type'])
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction duration')
drift_alerts = Counter('drift_alerts_total', 'Drift detection alerts', ['drift_type'])

# Expose metrics
start_http_server(9090)
```

#### 5.3 CloudWatch Logs Integration

Ensure logs are sent to CloudWatch with proper log groups:

```python
import watchtower

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        watchtower.CloudWatchLogHandler(
            log_group='/ecs/ml-recommender-api',
            stream_name='api-logs'
        )
    ]
)
```

### Phase 6: Terraform Implementation

#### 6.1 Module Structure

```
infra-api/
├── grafana/
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   └── dashboards/
│       ├── api-overview.json
│       ├── mlflow.json
│       ├── infrastructure.json
│       ├── model-performance.json
│       └── security.json
```

#### 6.2 Variables

```hcl
variable "project_name" {
  description = "Project name"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "enable_sso" {
  description = "Enable AWS SSO authentication"
  type        = bool
  default     = false
}

variable "sso_instance_arn" {
  description = "AWS SSO instance ARN"
  type        = string
  default     = null
}
```

#### 6.3 Outputs

```hcl
output "grafana_workspace_url" {
  description = "Grafana workspace URL"
  value       = aws_grafana_workspace.main.endpoint
}

output "grafana_workspace_id" {
  description = "Grafana workspace ID"
  value       = aws_grafana_workspace.main.id
}
```

## Migration Strategy

### Step 1: Deploy Grafana Infrastructure
- Create Terraform module for Grafana
- Deploy to development environment
- Test authentication and data sources

### Step 2: Configure Data Sources
- Add CloudWatch Logs data source
- Add CloudWatch Metrics data source
- Test queries and dashboards

### Step 3: Import Existing Dashboards
- Create basic dashboards
- Import CloudWatch Insights queries
- Test panel configurations

### Step 4: Enable Application Metrics
- Add structured logging
- Add Prometheus metrics
- Update CloudWatch log groups

### Step 5: Configure Alerting
- Set up notification channels
- Create critical alerts
- Test alert delivery

### Step 6: Deploy to Production
- Review all configurations
- Deploy to production
- Monitor Grafana performance
- Fine-tune dashboards and alerts

## Cost Estimation

### AWS Managed Grafana
- **Workspace**: $9/month (for small workspace)
- **Users**: Included in workspace cost
- **Data transfer**: Minimal (within AWS region)
- **Estimated monthly cost**: ~$10-15

### Self-Hosted Grafana on ECS
- **ECS Fargate**: $0.04048/vCPU-hour + $0.0084/GB-hour
- **ALB**: $0.0225/hour + LCU charges
- **Estimated monthly cost**: ~$30-50

### CloudWatch Costs
- **Logs**: $0.50/GB ingested + $0.03/GB stored
- **Metrics**: Custom metrics: $0.30/metric
- **Estimated monthly cost**: ~$20-50 (depending on volume)

**Total estimated cost**: $30-80/month

## Security Considerations

### Access Control
- Use AWS IAM for authentication
- Implement least privilege access
- Use SSO for centralized user management
- Enable audit logging in Grafana

### Data Protection
- Encrypt data at rest (default in AWS Managed Grafana)
- Use TLS for data in transit
- Restrict Grafana access to specific IPs or VPC
- Regularly rotate API keys and credentials

### Compliance
- Enable Grafana audit logs
- Log all data access
- Implement data retention policies
- Regular security reviews

## Maintenance

### Regular Tasks
- Review and update dashboards monthly
- Fine-tune alert thresholds quarterly
- Review Grafana version updates
- Monitor Grafana performance
- Backup dashboard configurations

### Dashboard Updates
- Add new metrics as application evolves
- Remove obsolete panels
- Optimize query performance
- Update alert rules

### Documentation
- Document custom queries
- Maintain dashboard documentation
- Update runbooks for alerts
- Document incident response procedures

## Success Criteria

- [ ] Grafana workspace deployed and accessible
- [ ] All data sources configured and working
- [ ] At least 5 dashboards created with meaningful panels
- [ ] Alert rules configured for critical metrics
- [ ] Application logs visible in Grafana
- [ ] Team trained on Grafana usage
- [ ] Incident response procedures documented
- [ ] Cost monitoring implemented

## Next Steps

1. **Review and approve** this plan
2. **Create Terraform module** for Grafana deployment
3. **Deploy to dev environment** for testing
4. **Create initial dashboards** based on requirements
5. **Integrate application metrics** and logging
6. **Configure alerting** based on SLA requirements
7. **Deploy to production** after testing
8. **Train team** on Grafana usage and maintenance

## References

- [AWS Managed Grafana Documentation](https://docs.aws.amazon.com/grafana/latest/userguide/what-is-grafana.html)
- [Grafana CloudWatch Plugin](https://grafana.com/plugins/grafana-aws-cloudwatch-datasource/)
- [CloudWatch Logs Insights](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/AnalyzingLogData.html)
- [Grafana Dashboard Best Practices](https://grafana.com/docs/grafana/latest/best-practices/best-practices-for-creating-dashboards/)
