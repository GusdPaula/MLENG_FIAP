terraform {
  required_version = ">= 1.0.0"
  backend "s3" {
    bucket         = "terraform-state-mlflow-fiap-ulodyq7a"
    key            = "fase_2/infra-api/s3/terraform.tfstate"
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

# Random suffix for unique bucket name
resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

# Bucket S3 para armazenar os dados do DVC
resource "aws_s3_bucket" "dvc_bucket" {
  bucket        = "${var.bucket_name}-${random_string.bucket_suffix.result}"
  force_destroy = true

  tags = merge(var.common_tags, {
    Name        = "DVC Storage"
    Environment = "MLOps"
    Project     = "Tech Challenger FIAP"
  })
}

# Configura criptografia no descanso para o bucket S3
resource "aws_s3_bucket_server_side_encryption_configuration" "encryption" {
  bucket = aws_s3_bucket.dvc_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Configura versionamento para o bucket S3
resource "aws_s3_bucket_versioning" "versioning" {
  bucket = aws_s3_bucket.dvc_bucket.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Bloqueia o acesso público ao bucket S3
# Acesso deve ser através de políticas IAM apenas
resource "aws_s3_bucket_public_access_block" "public_access" {
  bucket = aws_s3_bucket.dvc_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Acesso ao bucket S3 é controlado através de políticas IAM apenas
# Ver aws_iam_policy.dvc_rw_policy e aws_iam_policy.dvc_readonly_policy

# Política IAM de Leitura e Escrita para o bucket do DVC e MLflow Artifacts
resource "aws_iam_policy" "dvc_rw_policy" {
  name        = "DVC-Bucket-ReadWrite-Policy-${random_string.bucket_suffix.result}"
  description = "Politica IAM para permitir leitura e escrita no bucket S3 do DVC e no MLflow Artifacts"

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

  tags = var.common_tags
}

# Usuário IAM para usar com o DVC
resource "aws_iam_user" "dvc_user" {
  name = "${var.iam_user_name}-${random_string.bucket_suffix.result}"

  tags = merge(var.common_tags, {
    Name        = "DVC User"
    Environment = "MLOps"
  })
}

# Anexa a política customizada de Leitura/Escrita ao usuário IAM
resource "aws_iam_user_policy_attachment" "dvc_rw_attach" {
  user       = aws_iam_user.dvc_user.name
  policy_arn = aws_iam_policy.dvc_rw_policy.arn
}

# Anexa a política gerenciada AWS SignInLocalDevelopmentAccess ao usuário IAM
# para permitir o login via terminal/Console Credentials
resource "aws_iam_user_policy_attachment" "signin_dev_attach" {
  user       = aws_iam_user.dvc_user.name
  policy_arn = "arn:aws:iam::aws:policy/SignInLocalDevelopmentAccess"
}

# Política IAM de somente leitura para o bucket do DVC
resource "aws_iam_policy" "dvc_readonly_policy" {
  name        = "DVC-Bucket-ReadOnly-Policy-${random_string.bucket_suffix.result}"
  description = "Politica IAM de somente leitura para listar e obter objetos do bucket S3 do DVC"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:ListAllMyBuckets"
        ]
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
        Action = [
          "s3:GetObject"
        ]
        Resource = "${aws_s3_bucket.dvc_bucket.arn}/*"
      }
    ]
  })

  tags = var.common_tags
}

# Usuário IAM de somente leitura para o DVC
resource "aws_iam_user" "dvc_readonly_user" {
  name = "${var.iam_user_name}-readonly-${random_string.bucket_suffix.result}"

  tags = merge(var.common_tags, {
    Name        = "DVC ReadOnly User"
    Environment = "MLOps"
  })
}

# Anexa a política de somente leitura ao usuário IAM readonly
resource "aws_iam_user_policy_attachment" "dvc_readonly_attach" {
  user       = aws_iam_user.dvc_readonly_user.name
  policy_arn = aws_iam_policy.dvc_readonly_policy.arn
}

# Anexa a política gerenciada AWS SignInLocalDevelopmentAccess ao usuário readonly
resource "aws_iam_user_policy_attachment" "readonly_signin_dev_attach" {
  user       = aws_iam_user.dvc_readonly_user.name
  policy_arn = "arn:aws:iam::aws:policy/SignInLocalDevelopmentAccess"
}

# Usuário IAM para revisão (professor/avaliador)
resource "aws_iam_user" "reviewer_user" {
  name = "${var.iam_user_name}-reviewer-${random_string.bucket_suffix.result}"

  tags = merge(var.common_tags, {
    Name        = "Reviewer User"
    Purpose     = "Teacher access for grading"
    Environment = "MLOps"
  })
}

# Política IAM de revisão (somente leitura para todos os recursos)
resource "aws_iam_policy" "reviewer_policy" {
  name        = "Reviewer-ReadOnly-Policy-${random_string.bucket_suffix.result}"
  description = "Politica IAM de somente leitura para revisão do projeto (acesso do professor)"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:ListAllMyBuckets"
        ]
        Resource = "arn:aws:s3:::*"
      },
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
          "s3:GetObject"
        ]
        Resource = [
          "${aws_s3_bucket.dvc_bucket.arn}/*",
          "arn:aws:s3:::${var.bucket_name_mlflow_artifacts}/*"
        ]
      }
    ]
  })

  tags = var.common_tags
}

# Anexa a política de revisão ao usuário IAM reviewer
resource "aws_iam_user_policy_attachment" "reviewer_attach" {
  user       = aws_iam_user.reviewer_user.name
  policy_arn = aws_iam_policy.reviewer_policy.arn
}

# Anexa a política gerenciada AWS SignInLocalDevelopmentAccess ao usuário reviewer
resource "aws_iam_user_policy_attachment" "reviewer_signin_dev_attach" {
  user       = aws_iam_user.reviewer_user.name
  policy_arn = "arn:aws:iam::aws:policy/SignInLocalDevelopmentAccess"
}
