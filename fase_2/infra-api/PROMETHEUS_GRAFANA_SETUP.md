# Prometheus and Grafana Setup and Troubleshooting

## Overview
This document summarizes the infrastructure configuration for the monitoring stack (Prometheus and Grafana) and the issues encountered and resolved during setup.

## Infrastructure Components

### Prometheus EC2 Instance
- **Instance ID**: i-0a3d9cdc4698ffbf1
- **Private IP**: 172.31.84.126
- **Public IP**: 44.203.48.206
- **Deployment**: Docker container via user_data.sh script
- **Configuration Location**: `/tmp/prometheus.yml`

### Grafana EC2 Instance
- **Instance ID**: i-0e84b330b0b4a9a0e
- **Private IP**: 172.31.80.215
- **Public IP**: 32.193.226.237
- **Deployment**: Native Grafana server service
- **Configuration Location**: `/etc/grafana/`

### API ECS Service
- **Cluster**: mlflow-fiap-api-cluster
- **Service**: mlflow-fiap-api-service
- **Task Definition**: mlflow-fiap-api-task:16
- **Private IP**: 172.31.12.159:8000
- **CloudFront URL**: https://d1b386spzciemm.cloudfront.net

## Initial Setup Process

### 1. Resource Startup
Used `scripts/start_resources.py` to start:
- MLflow EC2 instance
- Grafana EC2 instance
- Prometheus EC2 instance
- ECS API service

### 2. Issues Encountered

#### Issue 1: Prometheus Not Responding
**Problem**: Connection refused when trying to access Prometheus at http://172.31.84.126:9090

**Root Cause**:
- Prometheus was deployed as Docker container via user_data.sh
- When EC2 instance was stopped/restarted, Docker container didn't auto-start
- A native Prometheus process was running but not listening on port 9090

**Solution**:
```bash
# Killed native Prometheus process
pkill -f "prometheus"

# Removed old Docker container
docker rm -f prometheus

# Recreated Prometheus configuration with correct API endpoint
cat > /tmp/prometheus.yml <<EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'api'
    static_configs:
      - targets: ['172.31.12.159:8000']
    metrics_path: '/metrics/'
    scrape_interval: 10s
EOF

# Started Prometheus Docker container
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v /tmp/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus:latest
```

#### Issue 2: Prometheus Scraping Wrong API Endpoint
**Problem**: Prometheus was trying to scrape API at 172.31.88.58:8000 (MLflow instance) instead of the actual ECS service

**Root Cause**:
- Original user_data.sh used `${api_alb_dns}` variable which wasn't properly resolved
- Configuration pointed to wrong IP address

**Solution**:
- Queried ECS to find actual API task private IP: 172.31.12.159
- Updated Prometheus configuration to use correct IP
- Verified both targets showed "up" status in Prometheus

#### Issue 3: Grafana Not Showing Data
**Problem**: Grafana dashboard showed "No data" for all metrics

**Root Cause**:
- Grafana didn't have Prometheus configured as a datasource
- No datasource configuration existed in `/etc/grafana/provisioning/datasources/`

**Solution**:
```bash
# Created Prometheus datasource configuration
mkdir -p /etc/grafana/provisioning/datasources

cat > /etc/grafana/provisioning/datasources/prometheus.yml <<EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://172.31.84.126:9090
    isDefault: true
    editable: true
EOF

# Restarted Grafana to load new configuration
systemctl restart grafana-server
```

## Current Configuration Status

### Prometheus Status
- **Status**: Running in Docker container
- **Container ID**: bb94b8284b58
- **Port**: 9090 (mapped to 0.0.0.0:9090)
- **Targets**:
  - prometheus (localhost:9090): UP
  - api (172.31.12.159:8000): UP
- **Security Group**: sg-0cec2727e5d52467d (allows 172.31.0.0/16:9090)

### Grafana Status
- **Status**: Running as systemd service
- **Service**: grafana-server.service
- **Port**: 3000
- **Prometheus Datasource**: Configured and active
- **CloudFront URL**: https://d3naqrkpy0vqtm.cloudfront.net

### API Metrics Endpoint
- **Status**: Accessible at http://172.31.12.159:8000/metrics/
- **Scrape Interval**: 10 seconds
- **Health**: UP (successfully scraped by Prometheus)

## Access URLs

### Prometheus
- **Private**: http://172.31.84.126:9090
- **Public**: http://44.203.48.206:9090 (restricted to VPC access only)

### Grafana
- **Private**: http://172.31.80.215:3000
- **Public**: http://32.193.226.237:3000
- **CloudFront**: https://d3naqrkpy0vqtm.cloudfront.net
- **Dashboard**: https://d3naqrkpy0vqtm.cloudfront.net/d/a4vkb7/api-overview

### API
- **CloudFront**: https://d1b386spzciemm.cloudfront.net
- **Health**: https://d1b386spzciemm.cloudfront.net/health

## Monitoring Stack Architecture

```
ECS API Service (172.31.12.159:8000)
    ↓ (metrics scraped every 10s)
Prometheus (172.31.84.126:9090)
    ↓ (datasource)
Grafana (172.31.80.215:3000)
    ↓ (visualization)
CloudFront Distribution
    ↓ (public access)
Users
```

## Recommendations for Future Deployments

### 1. Add Prometheus Auto-Start to start_resources.py
The `start_resources.py` script should include a function to start the Prometheus Docker container similar to the existing `restart_mlflow_server()` function:

```python
def start_prometheus_service(instance_id):
    """Start the Prometheus Docker container using SSM."""
    ssm = boto3.client("ssm", region_name=REGION)

    script = """
    #!/bin/bash
    systemctl start docker
    docker rm -f prometheus || true
    docker run -d \
      --name prometheus \
      -p 9090:9090 \
      -v /tmp/prometheus.yml:/etc/prometheus/prometheus.yml \
      prom/prometheus:latest
    """

    # Execute via SSM...
```

### 2. Use ECS Task Discovery for Prometheus Configuration
Instead of hardcoding IP addresses, use ECS task discovery or configure Prometheus to use ECS service discovery for dynamic target configuration.

### 3. Add Grafana Datasource to Infrastructure Code
Include the Prometheus datasource configuration in the Grafana Terraform configuration or user_data script to ensure it's provisioned automatically.

### 4. Implement Health Checks
Add health check endpoints and monitoring to ensure all components of the monitoring stack are operational.

## Troubleshooting Commands

### Check Prometheus Status
```bash
# From local machine (if security group allows)
curl http://44.203.48.206:9090/-/healthy

# From Prometheus instance via SSM
curl http://localhost:9090/api/v1/targets
```

### Check Grafana Status
```bash
# From Grafana instance via SSM
systemctl status grafana-server
curl http://localhost:3000/api/health
```

### Check API Metrics
```bash
# From Prometheus instance
curl http://172.31.12.159:8000/metrics/
```

### Restart Services
```bash
# Restart Prometheus
docker restart prometheus

# Restart Grafana
systemctl restart grafana-server
```

## Summary

The monitoring stack is now fully operational:
- Prometheus successfully collects metrics from the API service
- Grafana is configured to query Prometheus as a datasource
- Metrics are available for visualization and monitoring
- Drift detection scripts can generate test data that appears in Grafana

The main issues were:
1. Docker containers not auto-starting after instance restart
2. Incorrect IP addresses in Prometheus configuration
3. Missing datasource configuration in Grafana

All issues have been resolved and the monitoring pipeline is functioning correctly.
