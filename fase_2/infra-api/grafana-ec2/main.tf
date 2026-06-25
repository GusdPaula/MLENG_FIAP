terraform {
  required_version = ">= 1.0.0"
  backend "s3" {
    bucket         = "terraform-state-mlflow-fiap-ulodyq7a"
    key            = "fase_2/infra-api/grafana-ec2/terraform.tfstate"
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

# --- Security Groups ---
resource "aws_security_group" "grafana_sg" {
  name        = "${var.project_name}-grafana-sg"
  description = "Security group for Grafana EC2 instance"
  vpc_id      = data.aws_vpc.default.id

  tags = var.common_tags

  ingress {
    description     = "Allow HTTP traffic from ALB"
    from_port       = 80
    to_port         = 80
    protocol        = "tcp"
    security_groups = [aws_security_group.alb_sg.id]
  }

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

resource "aws_security_group" "alb_sg" {
  name        = "${var.project_name}-grafana-alb-sg"
  description = "Security group for Grafana ALB"
  vpc_id      = data.aws_vpc.default.id

  tags = var.common_tags

  ingress {
    description = "Allow HTTP traffic from anywhere"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Allow HTTPS traffic from anywhere"
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
}

# --- IAM Role for EC2 ---
resource "aws_iam_role" "grafana_ec2_role" {
  name = "${var.project_name}-grafana-ec2-role"

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

resource "aws_iam_policy" "grafana_ec2_policy" {
  name        = "${var.project_name}-grafana-ec2-policy"
  description = "Policy for Grafana EC2 to access CloudWatch"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:ListMetrics",
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:GetMetricData"
        ]
        Resource = "*"
      }
    ]
  })

  tags = var.common_tags
}

resource "aws_iam_role_policy_attachment" "grafana_ec2_attach" {
  role       = aws_iam_role.grafana_ec2_role.name
  policy_arn = aws_iam_policy.grafana_ec2_policy.arn
}

resource "aws_iam_role_policy_attachment" "grafana_ec2_ssm" {
  role       = aws_iam_role.grafana_ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "grafana_ec2_profile" {
  name = "${var.project_name}-grafana-ec2-profile"
  role = aws_iam_role.grafana_ec2_role.name

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

resource "aws_instance" "grafana_server" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = var.instance_type

  subnet_id                   = data.aws_subnets.default.ids[0]
  vpc_security_group_ids      = [aws_security_group.grafana_sg.id]
  iam_instance_profile        = aws_iam_instance_profile.grafana_ec2_profile.name
  associate_public_ip_address = true
  user_data_replace_on_change = true

  tags = merge(var.common_tags, {
    Name = "${var.project_name}-grafana-server"
  })

  user_data = templatefile("${path.module}/user_data.sh", {})
}

# --- Application Load Balancer ---
resource "aws_lb" "grafana" {
  name               = "${var.project_name}-grafana-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = data.aws_subnets.default.ids

  enable_deletion_protection = false

  tags = var.common_tags
}

resource "aws_lb_target_group" "grafana" {
  name        = "${var.project_name}-grafana-tg"
  port        = 80
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.default.id
  target_type = "instance"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    path                = "/api/health"
    port                = 80
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 3
  }

  tags = var.common_tags
}

resource "aws_lb_listener" "grafana_http" {
  load_balancer_arn = aws_lb.grafana.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "forward"
    forward {
      target_group {
        arn = aws_lb_target_group.grafana.arn
      }
    }
  }
}

resource "aws_lb_target_group_attachment" "grafana" {
  target_group_arn = aws_lb_target_group.grafana.arn
  target_id        = aws_instance.grafana_server.id
  port             = 80
}

# --- CloudFront Distribution ---
resource "aws_cloudfront_distribution" "grafana_distribution" {
  enabled = true
  comment = "Grafana Distribution for ${var.project_name}"

  origin {
    domain_name = aws_lb.grafana.dns_name
    origin_id   = "grafana-alb-origin"

    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "http-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  default_cache_behavior {
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "grafana-alb-origin"

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
    cloudfront_default_certificate = true
  }

  tags = merge(var.common_tags, {
    Name = "${var.project_name}-grafana-cf"
  })
}
