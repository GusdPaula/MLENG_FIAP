#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="mlflow-fiap"
AWS_REGION="us-east-1"

echo -e "${GREEN}=== Complete AWS Infrastructure Deployment ===${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"
if ! command -v terraform &> /dev/null; then
    echo -e "${RED}Error: terraform is not installed${NC}"
    exit 1
fi

if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: aws cli is not installed${NC}"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}Prerequisites check passed${NC}"
echo ""

# Get required variables
echo -e "${YELLOW}Using provided credentials:${NC}"
DOCKERHUB_USERNAME="gusdpaula404"
API_KEY="hm8K1JkR2BY4-zn1VGsFO-1MP_xp39GjdoUUacfyEvk"
PROJECT_NAME="mlflow-fiap"
# Skip MLflow URL prompt for fresh deployment - will be set after MLflow deployment
MLFLOW_URL=""

echo -e "${GREEN}Docker Hub username: ${DOCKERHUB_USERNAME}${NC}"
echo -e "${GREEN}API key: ${API_KEY}${NC}"
echo -e "${GREEN}Project name: ${PROJECT_NAME}${NC}"
echo ""

# Deployment order: S3 -> MLflow -> API -> Prometheus -> Grafana
echo -e "${GREEN}=== Step 1: Deploy S3/DVC Storage ===${NC}"
cd s3
terraform init -reconfigure
terraform apply -auto-approve
S3_OUTPUTS=$(terraform output -json)
echo -e "${GREEN}S3/DVC deployment completed${NC}"
cd ..
echo ""

# Get MLflow artifacts bucket name for MLflow deployment
MLFLOW_ARTIFACTS_BUCKET=$(echo $S3_OUTPUTS | python3 -c "import sys, json; print(json.load(sys.stdin)['bucket_name']['value'])")
echo -e "${YELLOW}MLflow artifacts bucket: ${MLFLOW_ARTIFACTS_BUCKET}${NC}"
echo ""

echo -e "${GREEN}=== Step 2: Deploy MLflow Server ===${NC}"
cd mlflow
terraform init -reconfigure
terraform apply \
  -auto-approve \
  -var="dockerhub_username=${DOCKERHUB_USERNAME}" \
  -var="bucket_name_mlflow_artifacts=${MLFLOW_ARTIFACTS_BUCKET}"
MLFLOW_OUTPUTS=$(terraform output -json)
MLFLOW_CLOUDFRONT=$(echo $MLFLOW_OUTPUTS | python3 -c "import sys, json; print(json.load(sys.stdin)['cloudfront_domain_name']['value'])")
echo -e "${GREEN}MLflow deployment completed${NC}"
echo -e "${GREEN}MLflow URL: https://${MLFLOW_CLOUDFRONT}${NC}"
cd ..
echo ""

# Use provided MLflow URL or the one just deployed
if [ -z "$MLFLOW_URL" ]; then
    MLFLOW_URL="https://${MLFLOW_CLOUDFRONT}"
fi

echo -e "${GREEN}=== Step 3: Deploy API Server ===${NC}"
cd api
terraform init
terraform apply \
  -auto-approve \
  -var="api_key=${API_KEY}" \
  -var="mlflow_tracking_uri=${MLFLOW_URL}"
API_OUTPUTS=$(terraform output -json)
API_URL=$(echo $API_OUTPUTS | python3 -c "import sys, json; print(json.load(sys.stdin)['api_url']['value'])")
API_CLOUDFRONT=$(echo $API_OUTPUTS | python3 -c "import sys, json; print(json.load(sys.stdin)['cloudfront_domain_name']['value'])")
API_ALB_DNS=$(echo $API_OUTPUTS | python3 -c "import sys, json; print(json.load(sys.stdin)['api_alb_dns']['value'])")
echo -e "${GREEN}API deployment completed${NC}"
echo -e "${GREEN}API URL: ${API_URL}${NC}"
cd ..
echo ""

echo -e "${GREEN}=== Step 4: Deploy Prometheus ===${NC}"
cd prometheus
terraform init -reconfigure
terraform apply \
  -auto-approve \
  -var="api_alb_dns=${API_ALB_DNS}"
PROMETHEUS_OUTPUTS=$(terraform output -json)
PROMETHEUS_URL=$(echo $PROMETHEUS_OUTPUTS | python3 -c "import sys, json; print(json.load(sys.stdin)['prometheus_url']['value'])")
PROMETHEUS_PUBLIC_IP=$(echo $PROMETHEUS_OUTPUTS | python3 -c "import sys, json; print(json.load(sys.stdin)['prometheus_public_ip']['value'])")
echo -e "${GREEN}Prometheus deployment completed${NC}"
echo -e "${GREEN}Prometheus URL: ${PROMETHEUS_URL}${NC}"
echo -e "${GREEN}Prometheus Public IP: ${PROMETHEUS_PUBLIC_IP}${NC}"
cd ..
echo ""

echo -e "${GREEN}=== Step 5: Deploy Grafana Monitoring ===${NC}"
cd grafana-ec2
terraform init -reconfigure
terraform apply \
  -auto-approve \
  -var="prometheus_url=${PROMETHEUS_URL}" \
  -var="project_name=${PROJECT_NAME}"
GRAFANA_OUTPUTS=$(terraform output -json)
GRAFANA_URL=$(echo $GRAFANA_OUTPUTS | python3 -c "import sys, json; print(json.load(sys.stdin)['grafana_url']['value'])")
echo -e "${GREEN}Grafana deployment completed${NC}"
echo -e "${GREEN}Grafana URL: https://${GRAFANA_URL}${NC}"
cd ..
echo ""

echo -e "${GREEN}=== Deployment Summary ===${NC}"
echo -e "${GREEN}MLflow URL: https://${MLFLOW_CLOUDFRONT}${NC}"
echo -e "${GREEN}API URL: ${API_URL}${NC}"
echo -e "${GREEN}API CloudFront: https://${API_CLOUDFRONT}${NC}"
echo -e "${GREEN}Prometheus URL: ${PROMETHEUS_URL}${NC}"
echo -e "${GREEN}Prometheus Public IP: ${PROMETHEUS_PUBLIC_IP}${NC}"
echo -e "${GREEN}Grafana URL: https://${GRAFANA_URL}${NC}"
echo -e "${GREEN}API Key: ${API_KEY}${NC}"
echo ""
echo -e "${YELLOW}Save the API key securely - it will not be shown again!${NC}"
echo ""
echo -e "${GREEN}=== All deployments completed successfully ===${NC}"
