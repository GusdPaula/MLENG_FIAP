# Complete AWS Infrastructure Deployment

This directory contains all Terraform configurations and instructions to deploy the complete ML infrastructure including MLflow, DVC storage, and the Recommendation API to AWS.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        AWS Cloud                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐      ┌──────────────────┐           │
│  │   MLflow Server  │      │   API Server     │           │
│  │  (EC2 + RDS)     │      │  (ECS Fargate)   │           │
│  │  CloudFront URL  │      │  CloudFront URL  │           │
│  └────────┬─────────┘      └────────┬─────────┘           │
│           │                          │                     │
│           └──────────┬───────────────┘                     │
│                      │                                     │
│              ┌───────▼────────┐                            │
│              │   S3 Buckets   │                            │
│              │  (DVC + Artifacts)                         │
│              └────────────────┘                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. MLflow Server (`mlflow/`)
- **EC2 Instance**: t3.micro running MLflow server in Docker (Free Tier eligible)
- **RDS PostgreSQL**: db.t3.micro for experiment tracking (Free Tier eligible)
- **S3 Bucket**: MLflow artifacts storage
- **CloudFront**: HTTPS endpoint with default URL
- **ACM Certificate**: SSL (default AWS certificate)

### 2. DVC Storage (`s3/`)
- **S3 Bucket**: Versioned data storage for DVC
- **IAM Users**: Read/write and read-only access
- **Public Read**: Anonymous read access enabled

### 3. API Server (`api/`)
- **ECS Fargate**: 0.5 vCPU, 1GB RAM
- **Application Load Balancer**: Traffic distribution
- **CloudFront**: HTTPS endpoint with default URL
- **ECR Repository**: Docker image storage
- **CloudWatch Logs**: Application logging

## Prerequisites

### Required Tools
- Terraform 1.5.0+
- AWS CLI configured with credentials
- Docker (for building images)
- Python 3.12+ (for API key generation)

### AWS Requirements
- AWS account with appropriate permissions
- AWS credentials configured locally or via IAM role
- Terraform state stored locally (no S3 backend required)

### Configuration Files
- Update `mlflow/variables.tf` with your Docker Hub username
- Generate API key for the API deployment

## Quick Start

### Option 1: Deploy All Components

```bash
# 1. Deploy S3/DVC storage (no dependencies)
cd s3
terraform init -reconfigure
terraform apply -auto-approve
cd ..

# 2. Deploy MLflow server (depends on S3 for artifacts)
cd mlflow
terraform init -reconfigure
terraform apply -auto-approve \
  -var="dockerhub_username=your-dockerhub-username" \
  -var="bucket_name_mlflow_artifacts=<s3-bucket-from-step-1>"
cd ..

# 3. Deploy API server (depends on MLflow)
cd api
terraform init -reconfigure
API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
terraform apply -auto-approve \
  -var="api_key=$API_KEY" \
  -var="mlflow_tracking_uri=https://your-mlflow-cloudfront-url.cloudfront.net"
cd ..
```

### Option 2: Use Deployment Script

```bash
# Make script executable
chmod +x deploy-all.sh

# Run deployment
./deploy-all.sh
```

## Component Details

### MLflow Server

**Variables:**
```hcl
dockerhub_username = "your-dockerhub-username"  # Required
docker_image_tag = "latest"                      # Optional
use_custom_domain = false                        # Use default CloudFront URL
custom_domain_name = "mlflow.yourdomain.com"     # Only if use_custom_domain = true
bucket_name_mlflow_artifacts = "s3-bucket-name"  # S3 bucket for artifacts
instance_type = "t3.micro"                       # Free Tier eligible
db_instance_class = "db.t3.micro"                 # Free Tier eligible
```

**Outputs:**
- `cloudfront_domain_name`: MLflow access URL
- `s3_artifacts_bucket`: MLflow artifacts bucket name
- `rds_instance_id`: RDS instance identifier

**Access URL:** `https://dxxxxxxxx.cloudfront.net`

### DVC Storage

**Variables:**
```hcl
bucket_name = "fiap-ml-dvc-bucket-tech-challenger"  # Base S3 bucket name (random suffix added)
bucket_name_mlflow_artifacts = "mlflow-artifacts-*"  # MLflow bucket name
iam_user_name = "fiap-dvc-user"                      # Base IAM user name (random suffix added)
```

**Note:** Random suffixes are automatically added to resource names to avoid conflicts with existing resources.

**Outputs:**
- `dvc_remote_url`: DVC remote URL for configuration
- `dvc_user_name`: IAM user name for DVC
- `dvc_readonly_user_name`: Read-only user name

**DVC Configuration:**
```bash
dvc remote add myremote s3://fiap-ml-dvc-bucket-tech-challenger
dvc remote modify myremote access_key_id <access-key>
dvc remote modify myremote secret_access_key <secret-key>
```

### API Server

**Variables:**
```hcl
api_key = "your-generated-api-key"                  # Required
mlflow_tracking_uri = "https://mlflow-url.cloudfront.net"  # Required
docker_image_tag = "latest"                         # Optional
cpu = 512                                           # Optional (0.5 vCPU)
memory = 1024                                       # Optional (1GB)
use_custom_domain = false                           # Use default CloudFront URL
```

**Outputs:**
- `api_url`: API access URL
- `cloudfront_domain_name`: CloudFront distribution URL
- `ecr_repository_url`: ECR repository for Docker images

**Access URL:** `https://dyyyyyyyy.cloudfront.net`

**Getting API URL After Deployment:**
```bash
cd api
terraform output -raw api_url
terraform output -raw cloudfront_domain_name
```

## Deployment Order

**Important:** Deploy in this order due to dependencies:

1. **S3/DVC** (no dependencies)
2. **MLflow** (depends on S3 for artifacts)
3. **API** (depends on MLflow for model loading)

## Cost Optimization

### Estimated Monthly Costs (24/7 Operation)

| Component | Cost (Free Tier) | Cost (After Free Tier) |
|-----------|-----------------|------------------------|
| MLflow EC2 (t3.micro) | FREE | ~$8.46 |
| MLflow RDS (db.t3.micro) | FREE | ~$15.00 |
| API ECS Fargate | ~$15-20 | ~$15-20 |
| API ALB | ~$18-20 | ~$18-20 |
| CloudFront | ~$2-5 | ~$2-5 |
| S3 Storage | ~$0.023/GB | ~$0.023/GB |
| **Total** | **~$35-45** | **~$76-87** |

**Free Tier Benefits:**
- 750 hours/month of t3.micro EC2 instances (first 12 months)
- 750 hours/month of db.t3.micro RDS instances (first 12 months)
- 5GB standard storage, 20,000 Get Requests, 2,000 Put Requests with S3

### Cost Reduction Strategies

1. **Start/Stop Automation**: Use GitHub Actions or Lambda to stop resources when not in use (can reduce to ~$10-15/month)
2. **ECS Fargate Spot**: Up to 70% savings on compute
3. **API Gateway**: Replace ALB with API Gateway (pay-per-use, ~$3.50/million requests)
4. **Smaller API instance**: Reduce to 0.25 vCPU, 512MB RAM
5. **Free Tier Maximization**: Use t3.micro and db.t3.micro for first 12 months

## Security

### Network Security
- Security groups restrict access to CloudFront only
- No direct public access to EC2, ECS, or RDS
- VPC with private subnets for RDS

### IAM Security
- Least privilege IAM roles
- Separate IAM users for DVC access
- Secrets Manager for sensitive data (RDS password)

### Application Security
- API key authentication for API endpoints
- HTTPS only (CloudFront SSL termination)
- Security groups with managed prefix lists

## Troubleshooting

### MLflow Server Issues

**Problem:** MLflow not accessible
```bash
# Check EC2 instance status
aws ec2 describe-instances --instance-ids <instance-id>

# Check RDS status
aws rds describe-db-instances --db-instance-identifier <db-id>

# View EC2 user data logs
aws ssm get-command-invocation --command-id <command-id>
```

### API Server Issues

**Problem:** API returning 502/503 errors
```bash
# Check ECS task status
aws ecs describe-tasks --cluster mlflow-fiap-api-cluster --tasks <task-id>

# View CloudWatch logs
aws logs tail /ecs/mlflow-fiap-api --follow

# Check ALB target health
aws elbv2 describe-target-health --target-group-arn <tg-arn>
```

### DVC Issues

**Problem:** DVC push/pull failures
```bash
# Verify IAM user credentials
aws configure get aws_access_key_id --profile dvc-profile
aws configure get aws_secret_access_key --profile dvc-profile

# Test S3 access
aws s3 ls s3://fiap-ml-dvc-bucket-tech-challenger
```

## Cleanup

To destroy all infrastructure:

```bash
# Destroy in reverse order of deployment
cd api && terraform destroy -auto-approve && cd ..
cd mlflow && terraform destroy -auto-approve && cd ..
cd s3 && terraform destroy -auto-approve && cd ..
```

Or use the destroy script:
```bash
./destroy-all.sh
```

## Next Steps

1. **Monitoring**: Set up CloudWatch alarms for critical metrics
2. **CI/CD**: Integrate with GitHub Actions for automated deployments
3. **Custom Domain**: Add custom domains if desired (update `use_custom_domain = true`)
4. **Autoscaling**: Configure ECS autoscaling for API
5. **Backup**: Enable RDS automated backups and point-in-time recovery

## Support

For issues or questions:
- Check CloudWatch logs for error messages
- Review Terraform state with `terraform show`
- Verify AWS credentials and permissions
- Check security group rules and network ACLs
