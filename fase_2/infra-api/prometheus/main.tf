terraform {
  required_version = ">= 1.0.0"
  backend "s3" {
    bucket         = "terraform-state-mlflow-fiap-ulodyq7a"
    key            = "fase_2/infra-api/prometheus/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
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

# --- Get API Security Group ---
data "aws_security_group" "api_sg" {
  name = "${var.project_name}-api-sg"
}

# --- Security Group for Prometheus ---
resource "aws_security_group" "prometheus_sg" {
  name        = "${var.project_name}-prometheus-sg"
  description = "Security group for Prometheus EC2 instance"
  vpc_id      = data.aws_vpc.default.id

  tags = var.common_tags

  ingress {
    description     = "Allow Prometheus web UI from CloudFront"
    from_port       = 9090
    to_port         = 9090
    protocol        = "tcp"
    prefix_list_ids = [data.aws_ec2_managed_prefix_list.cloudfront.id]
  }

  ingress {
    description     = "Allow Prometheus to scrape API"
    from_port       = 9090
    to_port         = 9090
    protocol        = "tcp"
    security_groups = [data.aws_security_group.api_sg.id]
  }

  ingress {
    description = "Allow Grafana to access Prometheus"
    from_port   = 9090
    to_port     = 9090
    protocol    = "tcp"
    cidr_blocks = [data.aws_vpc.default.cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# --- IAM Role for Prometheus ---
resource "aws_iam_role" "prometheus_role" {
  name = "${var.project_name}-prometheus-role"

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

resource "aws_iam_role_policy_attachment" "prometheus_ssm_policy" {
  role       = aws_iam_role.prometheus_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
}

# --- EC2 Instance for Prometheus ---
resource "aws_instance" "prometheus" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = var.instance_type
  iam_instance_profile = aws_iam_instance_profile.prometheus_profile.name
  vpc_security_group_ids = [aws_security_group.prometheus_sg.id]
  subnet_id                   = data.aws_subnets.default.ids[0]
  associate_public_ip_address = true

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    api_alb_dns = var.api_alb_dns
  }))

  tags = merge(var.common_tags, {
    Name = "${var.project_name}-prometheus"
  })

  volume_tags = var.common_tags
}

# --- IAM Instance Profile ---
resource "aws_iam_instance_profile" "prometheus_profile" {
  name = "${var.project_name}-prometheus-profile"
  role = aws_iam_role.prometheus_role.name
}

# --- CloudWatch Log Group ---
resource "aws_cloudwatch_log_group" "prometheus" {
  name              = "/ec2/prometheus"
  retention_in_days = 7

  tags = var.common_tags
}
