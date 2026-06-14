terraform {
  required_version = ">= 1.0.0"
  backend "s3" {
    bucket  = "terraform-state-mlflow-fiap"
    key     = "fase_2/infra/mlflow/terraform.tfstate"
    region  = "us-east-1"
    encrypt = true
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

# --- S3 Bucket for Artifacts ---
resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket        = "mlflow-artifacts-fiap-${random_string.bucket_suffix.result}"
  force_destroy = true
}

# --- Secrets Manager for RDS ---
resource "random_password" "db_password" {
  length           = 16
  special          = true
  override_special = "!#$%&*()-_=+[]{}<>:?"
}

resource "aws_secretsmanager_secret" "db_password_secret" {
  name = "${var.project_name}-db-password-${random_string.bucket_suffix.result}"
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

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
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

# --- RDS PostgreSQL ---
resource "aws_db_subnet_group" "default" {
  name       = "${var.project_name}-db-subnet-group"
  subnet_ids = data.aws_subnets.default.ids
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

  tags = {
    Name = "${var.project_name}-server"
  }

  user_data = <<-EOF
    #!/bin/bash
    set -e
    
    # Install AWS CLI, Docker, and jq
    apt-get update
    apt-get install -y docker.io awscli jq

    # Get the db password from Secrets Manager (with retry loop for IAM propagation delay)
    for i in {1..12}; do
      DB_PASSWORD=$(aws secretsmanager get-secret-value --secret-id ${aws_secretsmanager_secret.db_password_secret.name} --region ${var.aws_region} --query SecretString --output text 2>/dev/null || echo "")
      if [ -n "$DB_PASSWORD" ]; then
        break
      fi
      echo "Waiting for Secrets Manager permissions..."
      sleep 5
    done

    if [ -z "$DB_PASSWORD" ]; then
      echo "Failed to retrieve database password from Secrets Manager"
      exit 1
    fi
    
    # Construct DB URI
    DB_URI="postgresql://mlflow_user:$${DB_PASSWORD}@${aws_db_instance.mlflow_db.endpoint}/mlflow"
    
    # Pull and Run from Docker Hub
    docker pull ${var.dockerhub_username}/mlflow-server:${var.docker_image_tag}
    docker run -d --restart always -p 5000:5000 \
      ${var.dockerhub_username}/mlflow-server:${var.docker_image_tag} \
      --host 0.0.0.0 \
      --backend-store-uri "$DB_URI" \
      --default-artifact-root "s3://${aws_s3_bucket.mlflow_artifacts.bucket}"
  EOF
}

# --- CloudFront Distribution ---
resource "aws_cloudfront_distribution" "mlflow_distribution" {
  enabled = true
  comment = "MLflow Distribution for ${var.project_name}"

  origin {
    domain_name = aws_instance.mlflow_server.public_dns
    origin_id   = "mlflow-ec2-origin"

    custom_origin_config {
      http_port              = 5000
      https_port             = 443
      origin_protocol_policy = "http-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  default_cache_behavior {
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "mlflow-ec2-origin"

    viewer_protocol_policy = "allow-all"

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
    cloudfront_default_certificate = true
  }

  tags = {
    Name = "${var.project_name}-cf"
  }
}
