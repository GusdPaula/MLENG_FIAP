terraform {
  required_version = ">= 1.0.0"
  backend "s3" {
    bucket         = "terraform-state-mlflow-fiap-ulodyq7a"
    key            = "fase_2/infra-api/mlflow/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

data "aws_ec2_managed_prefix_list" "cloudfront" {
  name = "com.amazonaws.global.cloudfront.origin-facing"
}

# --- S3 Bucket for Artifacts ---
resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket        = "mlflow-artifacts-fiap-${random_string.bucket_suffix.result}"
  force_destroy = true

  tags = var.common_tags
}

# Configure encryption at rest for MLflow artifacts bucket
resource "aws_s3_bucket_server_side_encryption_configuration" "encryption" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Configure versioning for MLflow artifacts bucket
resource "aws_s3_bucket_versioning" "versioning" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

# --- Secrets Manager for RDS ---
resource "random_password" "db_password" {
  length           = 16
  special          = true
  override_special = "!#$%&*()-_=+[]{}<>:?"
}

resource "aws_secretsmanager_secret" "db_password_secret" {
  name = "${var.project_name}-db-password-new-${random_string.bucket_suffix.result}"

  recovery_window_in_days = 0

  tags = var.common_tags
}

resource "aws_secretsmanager_secret_version" "db_password_secret_val" {
  secret_id     = aws_secretsmanager_secret.db_password_secret.id
  secret_string = random_password.db_password.result
}

# --- Security Groups ---
resource "aws_security_group" "ec2_sg" {
  name        = "${var.project_name}-ec2-sg"
  description = "Security group for MLflow EC2 instance"
  vpc_id      = data.aws_vpc.default.id

  tags = var.common_tags

  ingress {
    description     = "Allow HTTPS traffic from ALB"
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    security_groups = [aws_security_group.alb_sg.id]
  }

  ingress {
    description     = "Allow traffic from CloudFront (backup)"
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    prefix_list_ids = [data.aws_ec2_managed_prefix_list.cloudfront.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "rds_sg" {
  name        = "${var.project_name}-rds-sg"
  description = "Security group for MLflow RDS instance"
  vpc_id      = data.aws_vpc.default.id

  tags = var.common_tags

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ec2_sg.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "alb_sg" {
  name        = "${var.project_name}-alb-sg"
  description = "Security group for MLflow ALB"
  vpc_id      = data.aws_vpc.default.id

  tags = var.common_tags

  ingress {
    description     = "Allow HTTP traffic from anywhere"
    from_port       = 80
    to_port         = 80
    protocol        = "tcp"
    cidr_blocks     = ["0.0.0.0/0"]
  }

  ingress {
    description     = "Allow HTTPS traffic from anywhere"
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    cidr_blocks     = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# --- RDS PostgreSQL ---
resource "aws_db_subnet_group" "default" {
  name       = "${var.project_name}-db-subnet-group"
  subnet_ids = data.aws_subnets.default.ids

  tags = var.common_tags
}

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
  backup_retention_period = 7
  multi_az                = false
  storage_encrypted       = true
  monitoring_interval    = 0

  tags = var.common_tags
}

# --- IAM Role for EC2 ---
resource "aws_iam_role" "ec2_role" {
  name = "${var.project_name}-ec2-role"

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

  tags = var.common_tags
}

resource "aws_iam_policy" "ec2_policy" {
  name        = "${var.project_name}-ec2-policy"
  description = "Policy for EC2 to access S3 and Secrets Manager"

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

  tags = var.common_tags
}

resource "aws_iam_role_policy_attachment" "ec2_attach" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = aws_iam_policy.ec2_policy.arn
}

resource "aws_iam_role_policy_attachment" "ec2_ssm" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${var.project_name}-ec2-profile"
  role = aws_iam_role.ec2_role.name

  tags = var.common_tags
}

# --- EC2 Instance ---
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
}

resource "aws_instance" "mlflow_server" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = var.instance_type

  subnet_id                   = data.aws_subnets.default.ids[0]
  vpc_security_group_ids      = [aws_security_group.ec2_sg.id]
  iam_instance_profile        = aws_iam_instance_profile.ec2_profile.name
  associate_public_ip_address = true
  user_data_replace_on_change = true

  tags = merge(var.common_tags, {
    Name = "${var.project_name}-server"
  })

  user_data = templatefile("${path.module}/user_data.sh", {
    db_password_secret_arn = aws_secretsmanager_secret.db_password_secret.id
    rds_endpoint          = aws_db_instance.mlflow_db.endpoint
    s3_bucket             = aws_s3_bucket.mlflow_artifacts.bucket
  })
}

# --- Application Load Balancer ---
resource "aws_lb" "mlflow" {
  name               = "${var.project_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = data.aws_subnets.default.ids

  enable_deletion_protection = false

  tags = var.common_tags
}

resource "aws_lb_target_group" "mlflow" {
  name        = "${var.project_name}-tg"
  port        = 443
  protocol    = "HTTPS"
  vpc_id      = data.aws_vpc.default.id
  target_type = "instance"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    path                = "/health"
    port                = 443
    protocol            = "HTTPS"
    timeout             = 5
    unhealthy_threshold = 3
  }

  tags = var.common_tags
}

resource "aws_lb_listener" "mlflow_http" {
  load_balancer_arn = aws_lb.mlflow.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "forward"
    forward {
      target_group {
        arn = aws_lb_target_group.mlflow.arn
      }
    }
  }
}

resource "aws_lb_listener" "mlflow_https" {
  count             = var.use_custom_domain ? 1 : 0
  load_balancer_arn = aws_lb.mlflow.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS-1-2-2017-01"
  certificate_arn   = aws_acm_certificate.mlflow_cert[0].arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.mlflow.arn
  }
}

resource "aws_lb_target_group_attachment" "mlflow" {
  target_group_arn = aws_lb_target_group.mlflow.arn
  target_id        = aws_instance.mlflow_server.id
  port             = 443
}

# --- ACM Certificate ---
resource "aws_acm_certificate" "mlflow_cert" {
  count             = var.use_custom_domain ? 1 : 0
  domain_name       = var.custom_domain_name
  validation_method = "DNS"

  tags = merge(var.common_tags, {
    Name = "${var.project_name}-cert"
  })

  lifecycle {
    create_before_destroy = true
  }
}

# --- CloudFront Distribution ---
resource "aws_cloudfront_distribution" "mlflow_distribution" {
  enabled = true
  comment = "MLflow Distribution for ${var.project_name}"
  aliases = var.use_custom_domain ? [var.custom_domain_name] : []

  origin {
    domain_name = aws_lb.mlflow.dns_name
    origin_id   = "mlflow-alb-origin"

    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = var.use_custom_domain ? "https-only" : "http-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  default_cache_behavior {
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "mlflow-alb-origin"

    viewer_protocol_policy = "redirect-to-https"

    forwarded_values {
      query_string = true
      headers      = ["*"]

      cookies {
        forward = "all"
      }
    }

    min_ttl     = 0
    default_ttl = 0
    max_ttl     = 0
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = !var.use_custom_domain
    acm_certificate_arn            = var.use_custom_domain ? aws_acm_certificate.mlflow_cert[0].arn : null
    ssl_support_method             = var.use_custom_domain ? "sni-only" : null
    minimum_protocol_version       = var.use_custom_domain ? "TLSv1.2_2021" : null
  }

  tags = merge(var.common_tags, {
    Name = "${var.project_name}-cf"
  })
}

# --- VPC Flow Logs ---
resource "aws_iam_role" "flow_log_role" {
  name = "${var.project_name}-flow-log-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "vpc-flow-logs.amazonaws.com"
      }
    }]
  })

  tags = var.common_tags
}

resource "aws_iam_role_policy_attachment" "flow_log_attach" {
  role       = aws_iam_role.flow_log_role.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
}

resource "aws_cloudwatch_log_group" "flow_log" {
  name              = "${var.project_name}-vpc-flow-logs"
  retention_in_days = 7

  tags = var.common_tags
}

resource "aws_cloudwatch_log_group" "mlflow" {
  name              = "/mlflow-logs"
  retention_in_days = 7

  tags = var.common_tags
}

resource "aws_flow_log" "vpc_flow_log" {
  iam_role_arn    = aws_iam_role.flow_log_role.arn
  log_destination = aws_cloudwatch_log_group.flow_log.arn
  traffic_type    = "ALL"
  vpc_id         = data.aws_vpc.default.id
}
