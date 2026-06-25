terraform {
  required_version = ">= 1.0.0"
  backend "s3" {
    bucket         = "terraform-state-mlflow-fiap-ulodyq7a"
    key            = "fase_2/infra-api/api/terraform.tfstate"
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

# --- ECR Repository for API Image ---
resource "aws_ecr_repository" "api" {
  name                 = "${var.project_name}-api"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = var.common_tags
}

# --- ECR Lifecycle Policy for Image Scanning Enforcement ---
# TODO: Fix policy format - current schema is invalid for imageScanFindingCount
# resource "aws_ecr_lifecycle_policy" "scan_policy" {
#   repository = aws_ecr_repository.api.name
#   policy = jsonencode({
#     rules = [{
#       rulePriority = 1
#       description  = "Block images with high vulnerabilities for production"
#       selection = {
#         tagStatus     = "tagged"
#         tagPrefixList = ["prod"]
#         countType     = "imageScanFindingCount"
#         countUnit     = "count"
#         countNumber   = 0
#       }
#       action = {
#         type = "expire"
#       }
#     }]
#   })
# }

# --- ECS Cluster ---
resource "aws_ecs_cluster" "api" {
  name = "${var.project_name}-api-cluster"

  tags = var.common_tags
}

# --- Task Definition ---
resource "aws_ecs_task_definition" "api" {
  family                   = "${var.project_name}-api-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.cpu
  memory                   = var.memory
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn
  task_role_arn           = aws_iam_role.ecs_task_role.arn

  tags = var.common_tags
  runtime_platform {
    cpu_architecture = "ARM64"
    operating_system_family = "LINUX"
  }

  container_definitions = jsonencode([
    {
      name      = "api"
      image     = "${aws_ecr_repository.api.repository_url}:${var.docker_image_tag}"
      cpu       = var.cpu
      memory    = var.memory
      essential = true

      portMappings = [
        {
          containerPort = 8000
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "MLFLOW_TRACKING_URIS"
          value = var.mlflow_tracking_uri
        },
        {
          name  = "API_KEY"
          value = var.api_key
        },
        {
          name  = "MLFLOW_MODEL_ALIAS"
          value = var.mlflow_model_alias
        }
      ]

      secrets = []

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.api.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "api"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])
}

# --- Security Groups ---
resource "aws_security_group" "api_sg" {
  name        = "${var.project_name}-api-sg"
  description = "Security group for API ECS tasks"
  vpc_id      = data.aws_vpc.default.id

  tags = var.common_tags

  ingress {
    description     = "Allow traffic from ALB"
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb_sg.id]
  }

  ingress {
    description     = "Allow traffic from CloudFront"
    from_port       = 8000
    to_port         = 8000
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

# --- IAM Roles ---
resource "aws_iam_role" "ecs_execution_role" {
  name = "${var.project_name}-ecs-execution-role"

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

  tags = var.common_tags
}

resource "aws_iam_role_policy_attachment" "ecs_execution_role_policy" {
  role       = aws_iam_role.ecs_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "ecs_task_role" {
  name = "${var.project_name}-ecs-task-role"

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

  tags = var.common_tags
}

resource "aws_iam_role_policy" "ecs_task_s3_policy" {
  name = "${var.project_name}-ecs-task-s3-policy"
  role = aws_iam_role.ecs_task_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket",
          "s3:GetObjectVersion"
        ]
        Resource = [
          "arn:aws:s3:::mlflow-artifacts-fiap-7jy4yo6d",
          "arn:aws:s3:::mlflow-artifacts-fiap-7jy4yo6d/*"
        ]
      }
    ]
  })
}

# --- CloudWatch Logs ---
resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/api-logs"
  retention_in_days = 7

  tags = var.common_tags
}

# --- ECS Service ---
resource "aws_ecs_service" "api" {
  name            = "${var.project_name}-api-service"
  cluster         = aws_ecs_cluster.api.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  tags = var.common_tags

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.api_sg.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }
}

# --- Application Load Balancer ---
resource "aws_lb" "api" {
  name               = "${var.project_name}-api-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = data.aws_subnets.default.ids

  enable_deletion_protection = false

  tags = var.common_tags
}

resource "aws_security_group" "alb_sg" {
  name        = "${var.project_name}-api-alb-sg"
  description = "Security group for API ALB"
  vpc_id      = data.aws_vpc.default.id

  tags = var.common_tags

  ingress {
    description     = "Allow HTTP traffic from anywhere"
    from_port       = 80
    to_port         = 80
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

resource "aws_lb_target_group" "api" {
  name        = "${var.project_name}-api-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.default.id
  target_type = "ip"

  tags = var.common_tags

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    path                = "/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 3
  }
}

resource "aws_lb_listener" "api_http" {
  load_balancer_arn = aws_lb.api.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "forward"
    forward {
      target_group {
        arn = aws_lb_target_group.api.arn
      }
    }
  }
}

resource "aws_lb_listener" "api_https" {
  count             = var.use_custom_domain ? 1 : 0
  load_balancer_arn = aws_lb.api.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS-1-2-2017-01"
  certificate_arn   = aws_acm_certificate.api_cert[0].arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

# --- ACM Certificates ---
resource "aws_acm_certificate" "api_cert" {
  count             = var.use_custom_domain ? 1 : 0
  domain_name       = var.custom_domain_name
  validation_method = "DNS"

  tags = var.common_tags

  lifecycle {
    create_before_destroy = true
  }
}

# --- CloudFront Distribution ---
resource "aws_cloudfront_distribution" "api" {
  enabled = true
  comment = "API Distribution for ${var.project_name}"
  aliases = var.use_custom_domain ? [var.custom_domain_name] : []

  origin {
    domain_name = aws_lb.api.dns_name
    origin_id   = "api-alb-origin"

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
    target_origin_id = "api-alb-origin"

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
    acm_certificate_arn            = var.use_custom_domain ? aws_acm_certificate.api_cert[0].arn : null
    ssl_support_method             = var.use_custom_domain ? "sni-only" : null
    minimum_protocol_version       = var.use_custom_domain ? "TLSv1.2_2021" : null
  }

  tags = merge(var.common_tags, {
    Name = "${var.project_name}-api-cf"
  })
}
