# AWS Policies and Infrastructure Guide

This guide explains AWS IAM policies, security concepts, and provides a detailed breakdown of the Terraform infrastructure in this repository.

## Table of Contents
1. [AWS IAM Policies Fundamentals](#aws-iam-policies-fundamentals)
2. [Policy Types in AWS](#policy-types-in-aws)
3. [Infrastructure Overview](#infrastructure-overview)
4. [API Infrastructure (api/)](#api-infrastructure-api)
5. [MLflow Infrastructure (mlflow/)](#mlflow-infrastructure-mlflow)
6. [S3 Infrastructure (s3/)](#s3-infrastructure-s3)
7. [Security Best Practices](#security-best-practices)

---

## AWS IAM Policies Fundamentals

### What is an IAM Policy?

An IAM (Identity and Access Management) policy is a JSON document that defines permissions. It specifies **who** can do **what** on **which resources**.

### Policy Structure

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "OptionalStatementID",
      "Effect": "Allow|Deny",
      "Action": [
        "service:action"
      ],
      "Resource": "arn:aws:service:region:account:resource"
    }
  ]
}
```

### Key Components

- **Version**: Policy language version (always use "2012-10-17")
- **Statement**: One or more permission statements
- **Sid** (Statement ID): Optional identifier for the statement
- **Effect**: Either "Allow" or "Deny"
- **Action**: Specific AWS service actions (e.g., `s3:GetObject`, `ec2:DescribeInstances`)
- **Resource**: AWS resources the action applies to (ARN format)

### ARN (Amazon Resource Name) Format

```
arn:partition:service:region:account-id:resource
```

Example: `arn:aws:s3:::my-bucket/path/to/object`

---

## Policy Types in AWS

### 1. Identity-Based Policies

Attached to IAM identities (users, groups, roles). Define what that identity can do.

**Example from our infrastructure (api/main.tf):**
```json
{
  "Effect": "Allow",
  "Action": [
    "s3:GetObject",
    "s3:ListBucket"
  ],
  "Resource": [
    "arn:aws:s3:::mlflow-artifacts-fiap-wxyhf9cb",
    "arn:aws:s3:::mlflow-artifacts-fiap-wxyhf9cb/*"
  ]
}
```

This policy allows ECS tasks to read from the MLflow S3 bucket.

### 2. Resource-Based Policies

Attached to AWS resources (S3 buckets, SNS topics, etc.). Define who can access that resource.

**Example from our infrastructure (s3/main.tf):**
```json
{
  "Sid": "PublicReadGetObject",
  "Effect": "Allow",
  "Principal": "*",
  "Action": "s3:GetObject",
  "Resource": "${aws_s3_bucket.dvc.arn}/*"
}
```

This is a bucket policy that allows anyone (Principal: "*") to read objects from the DVC bucket.

### 3. Trust Policies (Assume Role Policies)

Special policies that define which entities can assume a role.

**Example from our infrastructure (api/main.tf):**
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Action": "sts:AssumeRole",
    "Effect": "Allow",
    "Principal": {
      "Service": "ecs-tasks.amazonaws.com"
    }
  }]
}
```

This allows ECS tasks to assume this IAM role.

### 4. Inline vs Managed Policies

- **Inline Policies**: Embedded directly into an identity, specific to that identity
- **Managed Policies**: Standalone policies that can be attached to multiple identities
  - **AWS Managed**: Predefined by AWS (e.g., `arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess`)
  - **Customer Managed**: Created and managed by you

---

## Infrastructure Overview

This repository contains three main Terraform modules:

```
infra-api/
├── api/          # ECS Fargate API infrastructure
├── mlflow/       # MLflow server infrastructure
└── s3/           # S3 buckets and IAM policies for DVC
```

### Architecture Diagram

```
                    ┌─────────────────┐
                    │   CloudFront    │
                    │   (CDN)         │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Application   │
                    │  Load Balancer │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────▼──────┐ ┌─────▼──────┐ ┌────▼─────────┐
    │  ECS Fargate   │ │   EC2      │ │   S3 Buckets │
    │  API Service   │ │  MLflow    │ │   (DVC +     │
    │                │ │  Server    │ │   Artifacts) │
    └────────────────┘ └────────────┘ └──────────────┘
              │              │
              │              │
    ┌─────────▼──────┐ ┌─────▼──────┐
    │  RDS           │ │  Secrets   │
    │  PostgreSQL    │ │  Manager   │
    └────────────────┘ └────────────┘
```

---

## API Infrastructure (api/)

### Purpose
Deploys a scalable API service using ECS Fargate with CloudFront CDN.

### Components

#### 1. ECR Repository (`aws_ecr_repository`)
```hcl
resource "aws_ecr_repository" "api" {
  name                 = "${var.project_name}-api"
  image_tag_mutability = "MUTABLE"
  force_delete         = true
  image_scanning_configuration {
    scan_on_push = true
  }
}
```

**What it does:**
- Creates a Docker image registry in AWS
- Stores container images for the API
- Scans images for vulnerabilities on push
- Allows image tags to be overwritten (MUTABLE)

#### 2. ECS Cluster (`aws_ecs_cluster`)
```hcl
resource "aws_ecs_cluster" "api" {
  name = "${var.project_name}-api-cluster"
}
```

**What it does:**
- Creates a logical grouping of ECS tasks
- Acts as a container for your services

#### 3. Task Definition (`aws_ecs_task_definition`)
```hcl
resource "aws_ecs_task_definition" "api" {
  family                   = "${var.project_name}-api-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.cpu
  memory                   = var.memory
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn
  task_role_arn           = aws_iam_role.ecs_task_role.arn
  runtime_platform {
    cpu_architecture = "ARM64"
    operating_system_family = "LINUX"
  }
  # ... container definitions
}
```

**What it does:**
- Defines how to run your container (CPU, memory, image)
- Specifies IAM roles for the task
- Uses ARM64 architecture for cost savings
- Configures health checks, logging, and environment variables

**Container Environment Variables:**
- `MLFLOW_TRACKING_URIS`: URL to connect to MLflow
- `API_KEY`: Authentication key
- `MLFLOW_MODEL_ALIAS`: Which model version to use

#### 4. IAM Roles

**Execution Role** (`ecs_execution_role`):
```hcl
resource "aws_iam_role" "ecs_execution_role" {
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })
}
```

**What it does:**
- Allows ECS to pull images from ECR
- Allows ECS to send logs to CloudWatch
- Attached to `AmazonECSTaskExecutionRolePolicy` (AWS managed policy)

**Task Role** (`ecs_task_role`):
```hcl
resource "aws_iam_role_policy" "ecs_task_s3_policy" {
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::mlflow-artifacts-fiap-wxyhf9cb",
          "arn:aws:s3:::mlflow-artifacts-fiap-wxyhf9cb/*"
        ]
      }
    ]
  })
}
```

**What it does:**
- Allows the application code to access S3
- Specifically grants read-only access to MLflow artifacts
- The application uses this to load models from MLflow

#### 5. Security Groups

**API Security Group** (`api_sg`):
```hcl
resource "aws_security_group" "api_sg" {
  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb_sg.id]  # From ALB
  }
  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    prefix_list_ids = [data.aws_ec2_managed_prefix_list.cloudfront.id]  # From CloudFront
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
```

**What it does:**
- Controls inbound/outbound network traffic
- Allows port 8000 (API port) from ALB and CloudFront only
- Allows all outbound traffic (egress)
- Uses CloudFront prefix list for secure origin access

**ALB Security Group** (`alb_sg`):
```hcl
resource "aws_security_group" "alb_sg" {
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # Allow from anywhere
  }
}
```

**What it does:**
- Allows HTTP traffic from anywhere to the ALB
- CloudFront connects to ALB, ALB connects to ECS

#### 6. Application Load Balancer (`aws_lb`)
```hcl
resource "aws_lb" "api" {
  name               = "${var.project_name}-api-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = data.aws_subnets.default.ids
}
```

**What it does:**
- Distributes traffic across ECS tasks
- Performs health checks on targets
- Provides a single DNS endpoint for your service
- Works at Layer 7 (HTTP/HTTPS)

#### 7. CloudFront Distribution (`aws_cloudfront_distribution`)
```hcl
resource "aws_cloudfront_distribution" "api" {
  origin {
    domain_name = aws_lb.api.dns_name
    origin_id   = "api-alb-origin"
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "http-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }
  default_cache_behavior {
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD"]
    # ... no caching (TTL = 0) for API
  }
}
```

**What it does:**
- Provides a global CDN endpoint
- Enforces HTTPS (redirects HTTP to HTTPS)
- Caches GET/HEAD requests (but TTL set to 0 for API)
- Protects origin with CloudFront IP ranges
- Provides DDoS protection and edge caching

#### 8. CloudWatch Logs
```hcl
resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${var.project_name}-api"
  retention_in_days = 7
}
```

**What it does:**
- Centralized logging for ECS tasks
- 7-day retention period
- Accessible via AWS Console or CLI

---

## MLflow Infrastructure (mlflow/)

### Purpose
Deploys MLflow tracking server with PostgreSQL backend and S3 artifact storage.

### Components

#### 1. S3 Bucket for Artifacts (`aws_s3_bucket`)
```hcl
resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket        = "mlflow-artifacts-fiap-${random_string.bucket_suffix.result}"
  force_destroy = true
}
```

**What it does:**
- Stores MLflow model artifacts (trained models, metrics, etc.)
- Random suffix ensures globally unique name
- `force_destroy = true` allows deletion even with objects

#### 2. RDS PostgreSQL (`aws_db_instance`)
```hcl
resource "aws_db_instance" "mlflow_db" {
  identifier             = "${var.project_name}-db"
  allocated_storage      = 20
  engine                 = "postgres"
  engine_version         = "16.3"
  instance_class         = var.db_instance_class
  db_name                = "mlflow"
  username               = "mlflow_user"
  password               = random_password.db_password.result
  db_subnet_group_name   = aws_db_subnet_group.default.name
  vpc_security_group_ids = [aws_security_group.rds_sg.id]
  skip_final_snapshot    = true
  publicly_accessible    = false
}
```

**What it does:**
- Managed PostgreSQL database for MLflow metadata
- Stores experiment runs, parameters, metrics
- Not publicly accessible (private VPC)
- Auto-generated password stored in Secrets Manager

**Note:** The current user_data script actually uses SQLite instead of PostgreSQL. The RDS instance is created but not used by the MLflow server configuration.

#### 3. Secrets Manager (`aws_secretsmanager_secret`)
```hcl
resource "aws_secretsmanager_secret" "db_password_secret" {
  name = "${var.project_name}-db-password-new-${random_string.bucket_suffix.result}"
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "db_password_secret_val" {
  secret_id     = aws_secretsmanager_secret.db_password_secret.id
  secret_string = random_password.db_password.result
}
```

**What it does:**
- Securely stores the database password
- `recovery_window_in_days = 0` allows immediate deletion
- Can be rotated automatically (not configured here)

#### 4. EC2 Instance (`aws_instance`)
```hcl
resource "aws_instance" "mlflow_server" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = var.instance_type
  subnet_id                   = data.aws_subnets.default.ids[0]
  vpc_security_group_ids      = [aws_security_group.ec2_sg.id]
  iam_instance_profile        = aws_iam_instance_profile.ec2_profile.name
  associate_public_ip_address = true
  user_data_replace_on_change = true
}
```

**What it does:**
- Runs the MLflow server on Ubuntu
- Public IP for accessibility
- IAM profile for S3 and Secrets Manager access
- User data script installs and starts MLflow

#### 5. User Data Script
```bash
#!/bin/bash
apt-get update
apt-get install -y python3-pip python3-venv
python3 -m venv /opt/mlflow-venv
source /opt/mlflow-venv/bin/activate
pip install mlflow[extras] boto3
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root "s3://${aws_s3_bucket.mlflow_artifacts.bucket}" \
  --serve-artifacts \
  --allowed-hosts "*" \
  --cors-allowed-origins "*"
```

**What it does:**
- Installs Python and MLflow on EC2
- Creates virtual environment
- Starts MLflow server with:
  - SQLite backend (file-based)
  - S3 for artifact storage
  - Serves artifacts directly
  - Allows CORS for web UI access

#### 6. IAM Policy for EC2 (`aws_iam_policy`)
```hcl
resource "aws_iam_policy" "ec2_policy" {
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          aws_s3_bucket.mlflow_artifacts.arn,
          "${aws_s3_bucket.mlflow_artifacts.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = aws_secretsmanager_secret.db_password_secret.arn
      }
    ]
  })
}
```

**What it does:**
- Allows EC2 to read/write to MLflow S3 bucket
- Allows EC2 to read database password from Secrets Manager
- Full S3 access needed for artifact upload/download

#### 7. Security Groups

**EC2 Security Group** (`ec2_sg`):
```hcl
resource "aws_security_group" "ec2_sg" {
  ingress {
    from_port       = 5000
    to_port         = 5000
    protocol        = "tcp"
    prefix_list_ids = [data.aws_ec2_managed_prefix_list.cloudfront.id]
  }
  ingress {
    from_port       = 5000
    to_port         = 5000
    protocol        = "tcp"
    cidr_blocks     = ["177.39.99.133/32"]  # Developer IP (temporary)
  }
}
```

**What it does:**
- Allows port 5000 (MLflow) from CloudFront only
- Temporary exception for developer IP (should be removed)
- Protects MLflow server from direct internet access

**RDS Security Group** (`rds_sg`):
```hcl
resource "aws_security_group" "rds_sg" {
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ec2_sg.id]
  }
}
```

**What it does:**
- Allows PostgreSQL access only from EC2 instance
- No direct internet access to database

#### 8. CloudFront Distribution
Similar to API, but pointing to EC2 instance on port 5000.

---

## S3 Infrastructure (s3/)

### Purpose
Creates S3 buckets for DVC storage with appropriate IAM policies for team collaboration.

### Components

#### 1. DVC S3 Bucket (`aws_s3_bucket`)
```hcl
resource "aws_s3_bucket" "dvc_bucket" {
  bucket        = "${var.bucket_name}-${random_string.bucket_suffix.result}"
  force_destroy = true
  tags = {
    Name        = "DVC Storage"
    Environment = "MLOps"
    Project     = "Tech Challenger FIAP"
  }
}
```

**What it does:**
- Stores DVC-tracked datasets and model artifacts
- Random suffix for uniqueness
- Tagged for organization and cost tracking

#### 2. Public Access Block (`aws_s3_bucket_public_access_block`)
```hcl
resource "aws_s3_bucket_public_access_block" "public_access" {
  bucket = aws_s3_bucket.dvc_bucket.id
  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false
}
```

**What it does:**
- Disables AWS's default public access restrictions
- **Security Warning:** This allows public read access via bucket policy
- Only appropriate for public datasets

#### 3. Bucket Policy (`aws_s3_bucket_policy`)
```hcl
resource "aws_s3_bucket_policy" "public_read_policy" {
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "PublicReadGetObject"
        Effect    = "Allow"
        Principal = "*"
        Action    = "s3:GetObject"
        Resource  = "${aws_s3_bucket.dvc.arn}/*"
      },
      {
        Sid       = "PublicReadListBucket"
        Effect    = "Allow"
        Principal = "*"
        Action    = "s3:ListBucket"
        Resource  = aws_s3_bucket.dvc.arn
      }
    ]
  })
}
```

**What it does:**
- Allows anyone to read objects from the bucket
- Allows anyone to list bucket contents
- **Write operations still require authentication**

#### 4. IAM Read/Write Policy (`aws_iam_policy`)
```hcl
resource "aws_iam_policy" "dvc_rw_policy" {
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = [
          aws_s3_bucket.dvc_bucket.arn,
          "arn:aws:s3:::${var.bucket_name_mlflow_artifacts}"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          "${aws_s3_bucket.dvc_bucket.arn}/*",
          "arn:aws:s3:::${var.bucket_name_mlflow_artifacts}/*"
        ]
      }
    ]
  })
}
```

**What it does:**
- Grants full read/write access to DVC bucket
- Also grants access to MLflow artifacts bucket
- Used by developers who need to push/pull DVC data

#### 5. IAM Read-Only Policy (`aws_iam_policy`)
```hcl
resource "aws_iam_policy" "dvc_readonly_policy" {
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["s3:ListAllMyBuckets"]
        Resource = "arn:aws:s3:::*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = aws_s3_bucket.dvc_bucket.arn
      },
      {
        Effect = "Allow"
        Action = ["s3:GetObject"]
        Resource = "${aws_s3_bucket.dvc_bucket.arn}/*"
      }
    ]
  })
}
```

**What it does:**
- Grants read-only access to DVC bucket
- Can list and download objects but cannot upload/delete
- Used by team members who only need to access data

#### 6. IAM Users (`aws_iam_user`)
```hcl
resource "aws_iam_user" "dvc_user" {
  name = "${var.iam_user_name}-${random_string.bucket_suffix.result}"
}

resource "aws_iam_user" "dvc_readonly_user" {
  name = "${var.iam_user_name}-readonly-${random_string.bucket_suffix.result}"
}
```

**What it does:**
- Creates IAM users for DVC access
- Two users: one with read/write, one with read-only
- Users can generate access keys for DVC configuration

#### 7. AWS Managed Policy Attachment
```hcl
resource "aws_iam_user_policy_attachment" "signin_dev_attach" {
  user       = aws_iam_user.dvc_user.name
  policy_arn = "arn:aws:iam::aws:policy/SignInLocalDevelopmentAccess"
}
```

**What it does:**
- Attaches AWS managed policy for local development
- Allows users to sign in via AWS CLI
- Enables programmatic access with access keys

---

## Security Best Practices

### 1. Principle of Least Privilege
- Grant only the minimum permissions needed
- The read-only user can only read, not write
- ECS task role has specific S3 bucket access, not all S3

### 2. Resource-Level Permissions
- Use specific ARNs instead of wildcards
- Good: `arn:aws:s3:::my-bucket/*`
- Avoid: `arn:aws:s3:::*`

### 3. Security Groups as Network Firewall
- Restrict inbound traffic to necessary sources
- Use security groups instead of NACLs when possible
- CloudFront prefix lists for secure origin access

### 4. Secrets Management
- Never hardcode passwords in Terraform
- Use AWS Secrets Manager or Parameter Store
- The RDS password is randomly generated and stored securely

### 5. Infrastructure as Code Security
- Store Terraform state in encrypted S3 backend
- Use `.tfvars` files for sensitive values (not committed)
- Mark sensitive variables with `sensitive = true`

### 6. Temporary Access
- The developer IP in MLflow SG is temporary
- Should be removed after development is complete
- Consider using AWS SSM Session Manager instead

### 7. Monitoring and Logging
- CloudWatch Logs for ECS tasks
- Enable VPC Flow Logs for network monitoring
- Consider AWS CloudTrail for API auditing

### 8. Regular Rotation
- Rotate IAM access keys regularly
- Consider rotating database passwords
- Use IAM roles instead of access keys when possible

---

## Common S3 Actions Explained

| Action | Description | Use Case |
|--------|-------------|----------|
| `s3:ListBucket` | List objects in a bucket | Browsing contents |
| `s3:GetObject` | Download an object | Reading files |
| `s3:PutObject` | Upload an object | Writing files |
| `s3:DeleteObject` | Delete an object | Cleaning up |
| `s3:GetBucketLocation` | Get bucket region | Required for some SDKs |

---

## Common IAM Actions Explained

| Action | Description | Use Case |
|--------|-------------|----------|
| `sts:AssumeRole` | Assume an IAM role | Service-to-service auth |
| `secretsmanager:GetSecretValue` | Retrieve secret | Access credentials |
| `logs:CreateLogGroup` | Create CloudWatch log group | Application logging |
| `logs:CreateLogStream` | Create log stream | Application logging |
| `logs:PutLogEvents` | Write log events | Application logging |

---

## Terraform State Management

All modules use S3 backend for state storage:

```hcl
backend "s3" {
  bucket  = "terraform-state-mlflow-fiap-ulodyq7a"
  key     = "fase_2/infra-api/[module]/terraform.tfstate"
  region  = "us-east-1"
  encrypt = true
}
```

**Benefits:**
- State is encrypted at rest
- Team collaboration with locking
- State versioning (if enabled on bucket)
- Remote state access

---

## Deployment Workflow

1. **Initialize Terraform**
   ```bash
   cd infra-api/[module]
   terraform init
   ```

2. **Plan Changes**
   ```bash
   terraform plan -out=tfplan
   ```

3. **Apply Changes**
   ```bash
   terraform apply tfplan
   ```

4. **Destroy Infrastructure**
   ```bash
   terraform destroy
   ```

Or use the provided scripts:
- `deploy-all.sh`: Deploy all modules
- `destroy-all.sh`: Destroy all modules

---

## Troubleshooting

### Common Issues

1. **Bucket Already Exists**
   - S3 bucket names must be globally unique
   - The random suffix helps avoid conflicts

2. **IAM Permission Denied**
   - Check if the IAM user/role has the necessary policy attached
   - Verify the ARN in the policy matches the resource

3. **Security Group Blocking Traffic**
   - Ensure security groups allow traffic between components
   - Check that ports match (8000 for API, 5000 for MLflow)

4. **CloudFront Origin Access**
   - Verify the origin security group allows CloudFront prefix list
   - Check that the origin protocol policy matches (http-only vs https-only)

5. **Terraform State Lock**
   - If state is locked, another apply may be running
   - Force unlock only if necessary: `terraform force-unlock`

---

## Cost Optimization Tips

1. **Use ARM64 Architecture**
   - ECS tasks use ARM64 for ~20% cost savings
   - Ensure Docker images are built for ARM64

2. **Right-Sizing Instances**
   - Monitor CPU/memory usage
   - Adjust `cpu` and `memory` variables accordingly

3. **S3 Lifecycle Policies**
   - Add lifecycle rules to move old data to cheaper storage
   - Consider deleting old artifacts

4. **CloudFront Caching**
   - Currently disabled (TTL=0) for API
   - Can enable for static content

5. **RDS Instance Class**
   - Use `db.t3.micro` for development
   - Scale up for production as needed

---

## Further Reading

- [AWS IAM Documentation](https://docs.aws.amazon.com/IAM/)
- [AWS S3 Security](https://docs.aws.amazon.com/AmazonS3/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [ECS Fargate Best Practices](https://docs.aws.amazon.com/AmazonECS/)
- [MLflow Deployment](https://mlflow.org/docs/latest/deployment/index.html)

---

## Summary

This infrastructure provides a complete MLOps platform:

- **API Service**: Scalable, containerized API with ECS Fargate
- **MLflow Server**: Model tracking and artifact management
- **Storage**: S3 buckets for DVC and MLflow artifacts
- **Security**: IAM policies, security groups, and encryption
- **CDN**: CloudFront for global access and DDoS protection

The IAM policies follow the principle of least privilege, granting only necessary permissions to each component. The infrastructure is designed for both development and production use, with appropriate security measures in place.
