#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${RED}=== Destroy All AWS Infrastructure ===${NC}"
echo ""
echo -e "${YELLOW}This will destroy all resources in the following order:${NC}"
echo -e "${YELLOW}1. Grafana Monitoring${NC}"
echo -e "${YELLOW}2. API Server${NC}"
echo -e "${YELLOW}3. MLflow Server${NC}"
echo -e "${YELLOW}4. S3/DVC Storage${NC}"
echo ""
read -p "Are you sure you want to continue? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo -e "${YELLOW}Aborted${NC}"
    exit 0
fi

echo ""

# Destroy in reverse order of deployment
echo -e "${YELLOW}=== Step 1: Destroy Grafana Monitoring ===${NC}"
cd grafana
terraform destroy -auto-approve
echo -e "${GREEN}Grafana destroyed${NC}"
cd ..
echo ""

echo -e "${YELLOW}=== Step 2: Destroy API Server ===${NC}"
cd api
terraform destroy -auto-approve
echo -e "${GREEN}API destroyed${NC}"
cd ..
echo ""

echo -e "${YELLOW}=== Step 3: Destroy MLflow Server ===${NC}"
cd mlflow
terraform destroy -auto-approve
echo -e "${GREEN}MLflow destroyed${NC}"
cd ..
echo ""

echo -e "${YELLOW}=== Step 4: Destroy S3/DVC Storage ===${NC}"
cd s3
terraform destroy -auto-approve
echo -e "${GREEN}S3/DVC destroyed${NC}"
cd ..
echo ""

echo -e "${GREEN}=== All infrastructure destroyed successfully ===${NC}"
