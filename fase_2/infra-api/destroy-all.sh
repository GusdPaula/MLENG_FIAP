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

echo -e "${RED}=== Destroy All AWS Infrastructure ===${NC}"
echo ""
echo -e "${YELLOW}This will destroy all resources in the following order:${NC}"
echo -e "${YELLOW}1. Grafana Monitoring${NC}"
echo -e "${YELLOW}2. Prometheus${NC}"
echo -e "${YELLOW}3. API Server${NC}"
echo -e "${YELLOW}4. MLflow Server${NC}"
echo -e "${YELLOW}5. S3/DVC Storage${NC}"
echo -e "${YELLOW}6. CloudTrail${NC}"
echo -e "${YELLOW}7. GitHub Actions IAM${NC}"
echo -e "${YELLOW}8. Terraform State${NC}"
echo ""
read -p "Are you sure you want to continue? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo -e "${YELLOW}Aborted${NC}"
    exit 0
fi

echo ""

# Destroy in reverse order of deployment
echo -e "${YELLOW}=== Step 1: Destroy Grafana Monitoring ===${NC}"
cd grafana-ec2
terraform init -reconfigure
terraform destroy -auto-approve -var="project_name=${PROJECT_NAME}" -var="prometheus_url=http://placeholder"
echo -e "${GREEN}Grafana destroyed${NC}"
cd ..
echo ""

echo -e "${YELLOW}=== Step 2: Destroy Prometheus ===${NC}"
cd prometheus
terraform init -reconfigure
terraform destroy -auto-approve -var="api_alb_dns=placeholder"
echo -e "${GREEN}Prometheus destroyed${NC}"
cd ..
echo ""

echo -e "${YELLOW}=== Step 3: Destroy API Server ===${NC}"
cd api
terraform init -reconfigure
terraform destroy -auto-approve -var="api_key=placeholder" -var="mlflow_tracking_uri=http://placeholder"
echo -e "${GREEN}API destroyed${NC}"
cd ..
echo ""

echo -e "${YELLOW}=== Step 4: Destroy MLflow Server ===${NC}"
cd mlflow
terraform init -reconfigure
terraform destroy -auto-approve -var="dockerhub_username=placeholder" -var="bucket_name_mlflow_artifacts=placeholder"
echo -e "${GREEN}MLflow destroyed${NC}"
cd ..
echo ""

echo -e "${YELLOW}=== Step 5: Destroy S3/DVC Storage ===${NC}"
cd s3
terraform init -reconfigure
terraform destroy -auto-approve
echo -e "${GREEN}S3/DVC destroyed${NC}"
cd ..
echo ""

echo -e "${YELLOW}=== Step 6: Destroy CloudTrail ===${NC}"
cd cloudtrail
terraform init -reconfigure
terraform destroy -auto-approve -var="project_name=${PROJECT_NAME}" -var="force_destroy_bucket=true"
echo -e "${GREEN}CloudTrail destroyed${NC}"
cd ..
echo ""

echo -e "${YELLOW}=== Step 7: Destroy GitHub Actions IAM ===${NC}"
cd github-actions-iam
terraform init -reconfigure
terraform destroy -auto-approve
echo -e "${GREEN}GitHub Actions IAM destroyed${NC}"
cd ..
echo ""

echo -e "${YELLOW}=== Step 8: Destroy Terraform State ===${NC}"
cd terraform-state
terraform init -reconfigure
terraform destroy -auto-approve
echo -e "${GREEN}Terraform State destroyed${NC}"
cd ..
echo ""

echo -e "${GREEN}=== All infrastructure destroyed successfully ===${NC}"
