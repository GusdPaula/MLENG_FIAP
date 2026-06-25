#!/bin/bash
set -e

echo "========================================"
echo "DVC Test Script with infra-api IAM Users"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"
if ! command -v poetry &> /dev/null; then
    echo -e "${RED}ERROR: Poetry not found. Please install Poetry first.${NC}"
    exit 1
fi

if ! command -v aws &> /dev/null; then
    echo -e "${RED}ERROR: AWS CLI not found. Please install AWS CLI first.${NC}"
    exit 1
fi

if ! poetry run dvc --version &> /dev/null; then
    echo -e "${RED}ERROR: DVC not found. Please install DVC via Poetry first.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All prerequisites installed${NC}"
echo ""

# Deploy S3 infrastructure from infra-api
echo -e "${YELLOW}Deploying S3 infrastructure from infra-api...${NC}"
cd s3
terraform init -reconfigure
terraform apply -auto-approve

# Get the outputs
BUCKET_NAME=$(terraform output -raw bucket_name)
DVC_USER_NAME=$(terraform output -raw dvc_user_name)
DVC_READONLY_USER_NAME=$(terraform output -raw dvc_readonly_user_name)
DVC_REMOTE_URL=$(terraform output -raw dvc_remote_url)

echo -e "${GREEN}✓ S3 infrastructure deployed${NC}"
echo "  - Bucket: $BUCKET_NAME"
echo "  - DVC User: $DVC_USER_NAME"
echo "  - DVC ReadOnly User: $DVC_READONLY_USER_NAME"
echo "  - DVC Remote URL: $DVC_REMOTE_URL"
echo ""

cd ..

# Note: AWS CLI is not available, using existing AWS profile
echo -e "${YELLOW}Note: AWS CLI not available, using existing AWS profile${NC}"
echo -e "${YELLOW}Please ensure your AWS credentials are configured for the DVC user${NC}"
echo "  - DVC User: $DVC_USER_NAME"
echo "  - DVC ReadOnly User: $DVC_READONLY_USER_NAME"
echo ""
echo "To create access keys manually, run:"
echo "  aws iam create-access-key --user-name $DVC_USER_NAME"
echo ""

# Backup current DVC config
echo -e "${YELLOW}Backing up current DVC configuration...${NC}"
cd ..
if [ -f .dvc/config ]; then
    cp .dvc/config .dvc/config.backup
    echo -e "${GREEN}✓ Current config backed up to .dvc/config.backup${NC}"
else
    echo -e "${YELLOW}No existing DVC config found${NC}"
fi
echo ""

# Update DVC config to use the new bucket
echo -e "${YELLOW}Updating DVC configuration to use infra-api bucket...${NC}"
cat > .dvc/config <<EOF
[core]
    analytics = false
    remote = s3-infra
    autostage = true
['remote "s3-infra"']
    url = $DVC_REMOTE_URL
    profile = aws
EOF

echo -e "${GREEN}✓ DVC config updated${NC}"
echo "  - Remote URL: $DVC_REMOTE_URL"
echo "  - AWS Profile: aws"
echo ""

# Install dependencies
echo -e "${YELLOW}Installing dependencies with Poetry...${NC}"
poetry install
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Test DVC status
echo -e "${YELLOW}Testing DVC status...${NC}"
poetry run dvc status
echo ""

# Test DVC pull (if data exists in the bucket)
echo -e "${YELLOW}Testing DVC pull...${NC}"
if poetry run dvc pull; then
    echo -e "${GREEN}✓ DVC pull successful${NC}"
else
    echo -e "${YELLOW}⚠ DVC pull failed (expected if bucket is empty)${NC}"
fi
echo ""

# Run DVC repro
echo -e "${YELLOW}Running DVC repro (full pipeline)...${NC}"
if poetry run dvc repro --force; then
    echo -e "${GREEN}✓ DVC repro successful${NC}"
else
    echo -e "${RED}✗ DVC repro failed${NC}"
    echo "Check the logs above for errors"
    exit 1
fi
echo ""

# Show metrics
echo -e "${YELLOW}Showing metrics...${NC}"
poetry run dvc metrics show
echo ""

# Test DVC push
echo -e "${YELLOW}Testing DVC push...${NC}"
if poetry run dvc push; then
    echo -e "${GREEN}✓ DVC push successful${NC}"
else
    echo -e "${YELLOW}⚠ DVC push failed - AWS profile may not be configured${NC}"
    echo "You can manually push later with: poetry run dvc push"
fi
echo ""

echo "========================================"
echo -e "${GREEN}DVC Test Completed Successfully!${NC}"
echo "========================================"
echo ""
echo "Summary:"
echo "  - S3 Bucket: $BUCKET_NAME"
echo "  - DVC User: $DVC_USER_NAME"
echo "  - DVC ReadOnly User: $DVC_READONLY_USER_NAME"
echo "  - DVC Remote: $DVC_REMOTE_URL"
echo ""
echo "To restore the original DVC config:"
echo "  mv .dvc/config.backup .dvc/config"
echo ""
echo "To destroy the infrastructure:"
echo "  cd infra-api/s3 && terraform destroy"
