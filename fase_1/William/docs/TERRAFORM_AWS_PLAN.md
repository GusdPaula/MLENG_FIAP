# Plano de Deploy Terraform - AWS Sem SageMaker

**Data**: 2026-04-19  
**Versão**: 1.0  
**Objetivo**: Deploy de produção com custo mínimo, SLA 99.9%, API pública

---

## 1. VISÃO GERAL DA ARQUITETURA

### Stack Principal

```
┌────────────────────────────────────────────────────┐
│                    INTERNET                        │
└────────────────────┬───────────────────────────────┘
                     │
              ┌──────▼──────┐
              │ Route 53    │ ← DNS
              │ api.churn   │
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │ CloudFront  │ ← CDN (cache)
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │ ALB         │ ← Load Balancer
              │ Port 443    │ (HTTPS only)
              └──────┬──────┘
                     │
         ┌───────────┼───────────┐
         │           │           │
      ┌──▼──┐    ┌──▼──┐    ┌──▼──┐
      │EC2-1│    │EC2-2│    │EC2-3│ ← ASG (min=2, max=10)
      │Port │    │Port │    │Port │ (t3.medium)
      │8000 │    │8000 │    │8000 │
      └──┬──┘    └──┬──┘    └──┬──┘
         │           │           │
         └───────────┼───────────┘
                     │
         ┌───────────┼───────────┐
         │           │           │
      ┌──▼───┐  ┌───▼────┐ ┌───▼──┐
      │ RDS  │  │   S3   │ │Redis │
      │ DB   │  │ Model  │ │Cache │
      │t3.sm │  │Artifacts│└──────┘
      └──────┘  └────────┘

┌─────────────────────────────────────┐
│ Lambda + EventBridge (Batch Job)    │
│ Exec: 23:00 UTC, Trigger daily      │
└─────────────────────────────────────┘
```

### Custo Estimado

| Componente | Instância | Preço | Qtd | Total |
|-----------|-----------|-------|-----|-------|
| **EC2** | t3.medium | $0.0416/h | 2 | $60/mês |
| **RDS** | db.t3.small | $0.0338/h | 1 | $45/mês |
| **S3** | 10GB | $0.023/GB | 10 | $0.23/mês |
| **ALB** | 1 LB | $16/mês | 1 | $16/mês |
| **ElastiCache** | cache.t3.micro | $0.015/h | 1 | $15/mês |
| **Lambda** | 5min/dia | $0.0000002/ms | - | $1/mês |
| **CloudWatch** | Logs+Metrics | - | - | $10/mês |
| **Data Transfer** | Out <100GB | $0.09/GB | - | $5/mês |
| **Route53** | DNS | $0.5/zona | 1 | $0.5/mês |
| **ACM** | SSL Cert | FREE | - | $0/mês |
| **NAT Gateway** | (se needed) | $32/mês | - | $0 (usar VPC endpoints) |
| | | | **TOTAL** | **~$153/mês** |

**Estratégia de Custo**:
- ✅ t3.medium (não t3.large): 50% custo, performance suficiente
- ✅ db.t3.small (não db.t3.medium): Adequado para 5GB dados
- ✅ cache.t3.micro: 87% mais barato que standard
- ✅ Spot instances (ASG): 70% desconto (para scale-out)
- ✅ S3 intelligent tiering: Auto-arquiva dados pouco usados
- ✅ NAT Gateway: Desativar (usar VPC endpoints privados)

---

## 2. ESTRUTURA TERRAFORM

### Organização de Diretórios

```
terraform/
├── main.tf              # Provider, outputs
├── variables.tf         # Variables, defaults
├── terraform.tfvars     # Values (staging vs prod)
├── vpc.tf               # VPC, subnets, security groups
├── compute.tf           # EC2, ASG, ALB
├── database.tf          # RDS PostgreSQL
├── cache.tf             # ElastiCache Redis
├── storage.tf           # S3 buckets, IAM roles
├── lambda.tf            # Lambda batch job
├── dns.tf               # Route53, ACM certificate
├── monitoring.tf        # CloudWatch alarms
└── modules/             # Reusable modules (optional)
    ├── network/
    ├── compute/
    └── database/
```

### Backend State Management

```hcl
# backend.tf
terraform {
  backend "s3" {
    bucket         = "churn-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}
```

---

## 3. CONFIGURAÇÃO DETALHADA

### 3.1 VPC & Networking (vpc.tf)

```hcl
# VPC com 3 subnets (2 públicas, 1 privada)
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr  # "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "churn-vpc-${var.environment}"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags = { Name = "churn-igw" }
}

# Public Subnets (para ALB)
resource "aws_subnet" "public_1" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "us-east-1a"
  map_public_ip_on_launch = true
  tags = { Name = "public-subnet-1a" }
}

resource "aws_subnet" "public_2" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.2.0/24"
  availability_zone       = "us-east-1b"
  map_public_ip_on_launch = true
  tags = { Name = "public-subnet-1b" }
}

# Private Subnet (para EC2, RDS, Redis)
resource "aws_subnet" "private_1" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.10.0/24"
  availability_zone = "us-east-1a"
  tags = { Name = "private-subnet-1a" }
}

# Route Tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block      = "0.0.0.0/0"
    gateway_id      = aws_internet_gateway.main.id
  }
  tags = { Name = "public-rt" }
}

resource "aws_route_table_association" "public_1" {
  subnet_id      = aws_subnet.public_1.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "public_2" {
  subnet_id      = aws_subnet.public_2.id
  route_table_id = aws_route_table.public.id
}

# Security Groups
resource "aws_security_group" "alb" {
  name   = "churn-alb-sg"
  vpc_id = aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "alb-sg" }
}

resource "aws_security_group" "app" {
  name   = "churn-app-sg"
  vpc_id = aws_vpc.main.id

  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "app-sg" }
}

resource "aws_security_group" "database" {
  name   = "churn-db-sg"
  vpc_id = aws_vpc.main.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.app.id]
  }

  tags = { Name = "db-sg" }
}
```

### 3.2 Compute (compute.tf)

```hcl
# EC2 Launch Template
resource "aws_launch_template" "app" {
  name_prefix   = "churn-app-"
  image_id      = var.ami_id  # Amazon Linux 2 or Ubuntu
  instance_type = var.instance_type  # "t3.medium"

  vpc_security_group_ids = [aws_security_group.app.id]

  iam_instance_profile {
    arn = aws_iam_instance_profile.app.arn
  }

  user_data = base64encode(<<EOF
#!/bin/bash
set -e

# Update system
apt-get update && apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker ubuntu

# Install docker-compose
curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Clone repo
cd /home/ubuntu
git clone https://github.com/company/churn-prediction.git
cd churn-prediction

# Pull model from S3
aws s3 cp s3://churn-artifacts/model.xgb src/models/
aws s3 cp s3://churn-artifacts/scaler.pkl src/models/

# Start Docker
docker-compose -f docker-compose.prod.yml up -d

# Health check script
cat > /opt/health_check.sh << 'HEALTHCHECK'
#!/bin/bash
curl -f http://localhost:8000/health || exit 1
HEALTHCHECK
chmod +x /opt/health_check.sh

# CloudWatch logs agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
dpkg -i -E ./amazon-cloudwatch-agent.deb
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config \
  -m ec2 \
  -c s3:churn-artifacts/cloudwatch-config.json \
  -s

echo "Setup complete" | logger -t churn-app
EOF

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = "churn-api-${var.environment}"
    }
  }
}

# Auto Scaling Group
resource "aws_autoscaling_group" "app" {
  name                = "churn-asg-${var.environment}"
  vpc_zone_identifier = [aws_subnet.private_1.id, aws_subnet.public_1.id]
  min_size            = var.asg_min_size    # 2
  max_size            = var.asg_max_size    # 10
  desired_capacity    = var.asg_desired     # 2
  health_check_type   = "ELB"
  health_check_grace_period = 300
  
  launch_template {
    id      = aws_launch_template.app.id
    version = "$Latest"
  }

  tag {
    key                 = "Name"
    value               = "churn-app-asg"
    propagate_at_launch = true
  }
}

# Scaling Policies
resource "aws_autoscaling_policy" "scale_up" {
  name                   = "churn-scale-up"
  autoscaling_group_name = aws_autoscaling_group.app.name
  adjustment_type        = "ChangeInCapacity"
  scaling_adjustment     = 2
  cooldown               = 300

  depends_on = [aws_autoscaling_group.app]
}

resource "aws_autoscaling_policy" "scale_down" {
  name                   = "churn-scale-down"
  autoscaling_group_name = aws_autoscaling_group.app.name
  adjustment_type        = "ChangeInCapacity"
  scaling_adjustment     = -1
  cooldown               = 300

  depends_on = [aws_autoscaling_group.app]
}

# CloudWatch Alarms for Scaling
resource "aws_cloudwatch_metric_alarm" "high_cpu" {
  alarm_name          = "churn-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 300
  statistic           = "Average"
  threshold           = 70
  alarm_actions       = [aws_autoscaling_policy.scale_up.arn]

  dimensions = {
    AutoScalingGroupName = aws_autoscaling_group.app.name
  }
}

resource "aws_cloudwatch_metric_alarm" "low_cpu" {
  alarm_name          = "churn-low-cpu"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 5
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 300
  statistic           = "Average"
  threshold           = 30
  alarm_actions       = [aws_autoscaling_policy.scale_down.arn]

  dimensions = {
    AutoScalingGroupName = aws_autoscaling_group.app.name
  }
}

# Application Load Balancer
resource "aws_lb" "app" {
  name               = "churn-alb-${var.environment}"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = [aws_subnet.public_1.id, aws_subnet.public_2.id]

  enable_deletion_protection = true

  tags = { Name = "churn-alb" }
}

# ALB Target Group
resource "aws_lb_target_group" "app" {
  name        = "churn-tg-${var.environment}"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  
  health_check {
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
    path                = "/health"
    matcher             = "200"
  }

  tags = { Name = "churn-tg" }
}

# ALB Listener (HTTP → HTTPS redirect)
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.app.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "redirect"

    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}

# ALB Listener (HTTPS)
resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.app.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS-1-2-2017-01"
  certificate_arn   = aws_acm_certificate.main.arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}

# Attach ASG to ALB
resource "aws_autoscaling_attachment" "app" {
  autoscaling_group_name = aws_autoscaling_group.app.id
  lb_target_group_arn    = aws_lb_target_group.app.arn
}
```

### 3.3 Database (database.tf)

```hcl
# RDS Security Group (already defined in vpc.tf)

# RDS Subnet Group
resource "aws_db_subnet_group" "main" {
  name       = "churn-db-subnet-group"
  subnet_ids = [aws_subnet.private_1.id, aws_subnet.public_1.id]

  tags = { Name = "churn-db-subnet-group" }
}

# RDS PostgreSQL
resource "aws_db_instance" "main" {
  identifier              = "churn-postgres-${var.environment}"
  engine                  = "postgres"
  engine_version          = "14.7"
  instance_class          = var.db_instance_class  # "db.t3.small"
  allocated_storage       = 100  # GB
  storage_type            = "gp3"
  storage_encrypted       = true
  kms_key_id              = aws_kms_key.rds.arn

  db_name                = "churn_db"
  username               = "admin"
  password               = random_password.db_password.result  # From AWS Secrets Manager
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.database.id]

  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "mon:04:00-mon:05:00"
  multi_az               = true  # High availability
  
  enabled_cloudwatch_logs_exports = ["postgresql"]

  skip_final_snapshot = false
  final_snapshot_identifier = "churn-postgres-final-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"

  tags = { Name = "churn-postgres" }
}

# Store password in Secrets Manager
resource "random_password" "db_password" {
  length  = 32
  special = true
}

resource "aws_secretsmanager_secret" "db_password" {
  name = "churn/db-password"
}

resource "aws_secretsmanager_secret_version" "db_password" {
  secret_id     = aws_secretsmanager_secret.db_password.id
  secret_string = random_password.db_password.result
}

# KMS Key for encryption
resource "aws_kms_key" "rds" {
  description             = "KMS key for RDS encryption"
  deletion_window_in_days = 10
  enable_key_rotation     = true
}

resource "aws_kms_alias" "rds" {
  name          = "alias/churn-rds"
  target_key_id = aws_kms_key.rds.key_id
}
```

### 3.4 Cache (cache.tf)

```hcl
# ElastiCache Subnet Group
resource "aws_elasticache_subnet_group" "main" {
  name       = "churn-redis-subnet-group"
  subnet_ids = [aws_subnet.private_1.id]
}

# Security Group for Redis
resource "aws_security_group" "redis" {
  name   = "churn-redis-sg"
  vpc_id = aws_vpc.main.id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.app.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "redis-sg" }
}

# ElastiCache Redis
resource "aws_elasticache_cluster" "main" {
  cluster_id           = "churn-redis-${var.environment}"
  engine               = "redis"
  engine_version       = "7.0"
  node_type            = var.cache_node_type  # "cache.t3.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379
  
  subnet_group_name          = aws_elasticache_subnet_group.main.name
  security_group_ids         = [aws_security_group.redis.id]
  
  automatic_failover_enabled = false  # Not needed for single node
  at_rest_encryption_enabled = true
  transit_encryption_enabled = false  # Simplify for now
  
  log_delivery_configuration {
    destination      = aws_cloudwatch_log_group.redis.name
    destination_type = "cloudwatch-logs"
    log_format       = "json"
    enabled          = true
  }

  tags = { Name = "churn-redis" }
}

resource "aws_cloudwatch_log_group" "redis" {
  name              = "/aws/elasticache/churn-redis"
  retention_in_days = 7
}
```

### 3.5 Storage (storage.tf)

```hcl
# S3 Bucket for Model Artifacts
resource "aws_s3_bucket" "artifacts" {
  bucket = "churn-artifacts-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# S3 Lifecycle Policy (archive old models)
resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    id     = "archive-old-models"
    status = "Enabled"

    transitions {
      days          = 90
      storage_class = "GLACIER"
    }

    expiration {
      days = 730  # 2 years
    }
  }
}

# IAM Role for EC2
resource "aws_iam_role" "app" {
  name = "churn-app-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "app" {
  name   = "churn-app-policy"
  role   = aws_iam_role.app.id
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
          aws_s3_bucket.artifacts.arn,
          "${aws_s3_bucket.artifacts.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "app" {
  name = "churn-app-profile"
  role = aws_iam_role.app.name
}
```

### 3.6 Lambda Batch Job (lambda.tf)

```hcl
# Lambda IAM Role
resource "aws_iam_role" "lambda" {
  name = "churn-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "lambda" {
  name   = "churn-lambda-policy"
  role   = aws_iam_role.lambda.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow"
        Action = [
          "rds-db:connect"
        ]
        Resource = aws_db_instance.main.resource_id
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject"
        ]
        Resource = "${aws_s3_bucket.artifacts.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "ec2:CreateNetworkInterface",
          "ec2:DescribeNetworkInterfaces",
          "ec2:DeleteNetworkInterface"
        ]
        Resource = "*"
      }
    ]
  })
}

# Lambda Function (Python)
resource "aws_lambda_function" "batch_predict" {
  filename      = "lambda_batch_predict.zip"  # Zip file com código
  function_name = "churn-batch-predict"
  role          = aws_iam_role.lambda.arn
  handler       = "index.lambda_handler"
  runtime       = "python3.11"
  timeout       = 300  # 5 minutes max
  memory_size   = 2048

  vpc_config {
    subnet_ids         = [aws_subnet.private_1.id]
    security_group_ids = [aws_security_group.app.id]
  }

  environment {
    variables = {
      DB_HOST        = aws_db_instance.main.endpoint
      DB_PORT        = 5432
      DB_NAME        = "churn_db"
      DB_USER        = aws_db_instance.main.username
      DB_PASSWORD    = aws_secretsmanager_secret.db_password.arn
      S3_BUCKET      = aws_s3_bucket.artifacts.id
      MODEL_KEY      = "models/model.xgb"
      SCALER_KEY     = "models/scaler.pkl"
    }
  }
}

# EventBridge Rule (Schedule)
resource "aws_cloudwatch_event_rule" "batch_daily" {
  name                = "churn-batch-daily"
  description         = "Trigger batch prediction daily at 23:00 UTC"
  schedule_expression = "cron(0 23 * * ? *)"  # 23:00 UTC
}

resource "aws_cloudwatch_event_target" "lambda" {
  rule      = aws_cloudwatch_event_rule.batch_daily.name
  target_id = "ChurnBatchLambda"
  arn       = aws_lambda_function.batch_predict.arn
}

resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.batch_predict.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.batch_daily.arn
}
```

### 3.7 DNS & Certificates (dns.tf)

```hcl
# Route53 Hosted Zone (assuming already exists or create new)
data "aws_route53_zone" "main" {
  name = var.domain_name  # "churn.company.com"
}

# ACM Certificate
resource "aws_acm_certificate" "main" {
  domain_name       = var.domain_name
  validation_method = "DNS"

  lifecycle {
    create_before_destroy = true
  }

  tags = { Name = "churn-cert" }
}

# DNS Validation Records
resource "aws_route53_record" "cert_validation" {
  for_each = {
    for dvo in aws_acm_certificate.main.domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  }

  allow_overwrite = true
  name            = each.value.name
  records         = [each.value.record]
  ttl             = 60
  type            = each.value.type
  zone_id         = data.aws_route53_zone.main.zone_id
}

# Certificate validation (wait for DNS)
resource "aws_acm_certificate_validation" "main" {
  certificate_arn           = aws_acm_certificate.main.arn
  timeouts {
    create = "5m"
  }

  depends_on = [aws_route53_record.cert_validation]
}

# Route53 A Record (pointing to ALB)
resource "aws_route53_record" "api" {
  zone_id = data.aws_route53_zone.main.zone_id
  name    = var.domain_name
  type    = "A"

  alias {
    name                   = aws_lb.app.dns_name
    zone_id                = aws_lb.app.zone_id
    evaluate_target_health = true
  }
}
```

### 3.8 Variables (variables.tf)

```hcl
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "staging"
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.medium"
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.small"
}

variable "cache_node_type" {
  description = "ElastiCache node type"
  type        = string
  default     = "cache.t3.micro"
}

variable "asg_min_size" {
  description = "ASG minimum size"
  type        = number
  default     = 2
}

variable "asg_max_size" {
  description = "ASG maximum size"
  type        = number
  default     = 10
}

variable "asg_desired" {
  description = "ASG desired capacity"
  type        = number
  default     = 2
}

variable "domain_name" {
  description = "Domain name for API"
  type        = string
  example     = "api.churn.company.com"
}

variable "tags" {
  description = "Common tags"
  type        = map(string)
  default = {
    Project = "telco-churn"
    Team    = "ml-platform"
  }
}
```

### 3.9 Outputs (outputs.tf)

```hcl
output "alb_dns_name" {
  value       = aws_lb.app.dns_name
  description = "ALB DNS name"
}

output "api_endpoint" {
  value       = "https://${var.domain_name}"
  description = "API public endpoint"
}

output "rds_endpoint" {
  value       = aws_db_instance.main.endpoint
  description = "RDS PostgreSQL endpoint"
  sensitive   = true
}

output "redis_endpoint" {
  value       = aws_elasticache_cluster.main.cache_nodes[0].address
  description = "ElastiCache Redis endpoint"
}

output "s3_bucket_name" {
  value       = aws_s3_bucket.artifacts.id
  description = "S3 bucket for artifacts"
}

output "lambda_function_name" {
  value       = aws_lambda_function.batch_predict.function_name
  description = "Lambda batch prediction function"
}
```

---

## 4. DEPLOYMENT STEPS

### 4.1 Pré-Requisitos

```bash
# Instalar Terraform (v1.5+)
brew install terraform  # macOS
sudo apt install terraform  # Linux

# Configurar AWS credentials
aws configure
export AWS_PROFILE=default

# Verificar acesso
aws sts get-caller-identity
```

### 4.2 Inicializar Terraform

```bash
cd terraform/

# Criar S3 bucket para state (one-time)
aws s3 mb s3://churn-terraform-state-$(aws sts get-caller-identity --query Account --output text)

# Criar DynamoDB table para locks (one-time)
aws dynamodb create-table \
  --table-name terraform-locks \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5 \
  --region us-east-1

# Inicializar Terraform
terraform init
```

### 4.3 Planejar Deploy

```bash
# Validar configuração
terraform validate

# Formatar código
terraform fmt -recursive

# Planejar (dry-run)
terraform plan -out=tfplan

# Revisar output antes de aplicar
cat tfplan
```

### 4.4 Aplicar Configuração

```bash
# Aplicar
terraform apply tfplan

# Isso vai levar ~15-20 minutos
# Monitorar progresso
# - VPC + subnets (1 min)
# - Security Groups (30s)
# - RDS (5-10 min - requer provisioning)
# - EC2 + ASG (2 min)
# - ALB (1 min)
# - Lambda (1 min)
```

### 4.5 Pós-Deploy

```bash
# Obter outputs
terraform output

# Testar API
curl https://api.churn.company.com/health

# Verificar EC2 instances
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=churn-api*" \
  --region us-east-1

# Ver logs
aws logs tail /aws/ec2/churn-api --follow

# SSH to instance (if needed)
aws ssm start-session --target <instance-id>
```

---

## 5. MANUTENÇÃO & UPDATES

### 5.1 Escalando

```bash
# Aumentar desiredCapacity temporariamente
terraform apply -var="asg_desired=5"

# Ou via AWS console (ASG → Edit desired capacity)
```

### 5.2 Updatando Código

```bash
# SSH to EC2
aws ssm start-session --target <instance-id>

# Pull novo código
cd /home/ubuntu/churn-prediction
git pull

# Restart Docker
docker-compose -f docker-compose.prod.yml restart

# Verificar
curl http://localhost:8000/health
```

### 5.3 Backup Database

```bash
# Manual snapshot
aws rds create-db-snapshot \
  --db-instance-identifier churn-postgres-staging \
  --db-snapshot-identifier churn-manual-backup-$(date +%Y%m%d)

# Ver snapshots
aws rds describe-db-snapshots \
  --db-instance-identifier churn-postgres-staging
```

---

## 6. CLEANUP (Deletar Resources)

```bash
# Destruir tudo (use com cuidado!)
terraform destroy

# Ou especificar recurso
terraform destroy -target=aws_lb.app
```

---

## 7. ESTIMATIVA DE CUSTO

### Staging (2x t3.medium)
- EC2: $60/mês
- RDS: $45/mês
- Cache: $15/mês
- Outros: $30/mês
- **Total: ~$150/mês**

### Produção (2-5x t3.medium + scaling)
- EC2 (ASG avg 3): $90/mês
- RDS (multi-AZ): $90/mês
- Cache: $15/mês
- ALB + Data: $40/mês
- **Total: ~$235-300/mês**

### Formas de Economizar
1. Usar Spot instances (70% desconto) → -$60/mês
2. Reserved instances (40% desconto) → -$50/mês  
3. Reduzir retention RDS (7d → 3d) → -$10/mês
4. S3 Intelligent Tiering → -$5/mês

**Custo Total Otimizado: ~$100-120/mês**

---

## 8. CHECKLIST PRÉ-PRODUÇÃO

- [ ] Terraform apply bem-sucedido
- [ ] Todos outputs verificados
- [ ] API respondendo ✅ /health
- [ ] EC2 health checks passando
- [ ] RDS conectável
- [ ] Redis conectável
- [ ] S3 com modelos uploaded
- [ ] Lambda batch job funcionando
- [ ] CloudWatch logs visíveis
- [ ] Alertas configurados
- [ ] Backup RDS automático 7 dias
- [ ] HTTPS ativo (certificate valid)
- [ ] Security groups corretos
- [ ] IAM roles principle of least privilege

---

**Status**: Ready for staging deployment  
**Data**: 2026-04-19  
**Versão**: 1.0
