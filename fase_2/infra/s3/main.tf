terraform {
  required_version = ">= 1.0.0"
  backend "s3" {
    bucket  = "terraform-state-mlflow-fiap"
    key     = "fase_2/infra/s3/terraform.tfstate"
    region  = "us-east-1"
    encrypt = true
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

# Bucket S3 para armazenar os dados do DVC
resource "aws_s3_bucket" "dvc_bucket" {
  bucket        = var.bucket_name
  force_destroy = true

  tags = {
    Name        = "DVC Storage"
    Environment = "MLOps"
    Project     = "Tech Challenger FIAP"
  }
}

# Desativa o bloqueio de acesso público ao bucket para permitir a leitura anônima
resource "aws_s3_bucket_public_access_block" "public_access" {
  bucket = aws_s3_bucket.dvc_bucket.id

  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false
}

# Política do bucket que permite apenas leitura pública/anônima (GetObject e ListBucket)
# mantendo a escrita restrita às credenciais da conta do proprietário.
resource "aws_s3_bucket_policy" "public_read_policy" {
  depends_on = [aws_s3_bucket_public_access_block.public_access]
  bucket     = aws_s3_bucket.dvc_bucket.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "PublicReadGetObject"
        Effect    = "Allow"
        Principal = "*"
        Action    = "s3:GetObject"
        Resource  = "${aws_s3_bucket.dvc_bucket.arn}/*"
      },
      {
        Sid       = "PublicReadListBucket"
        Effect    = "Allow"
        Principal = "*"
        Action    = "s3:ListBucket"
        Resource  = aws_s3_bucket.dvc_bucket.arn
      }
    ]
  })
}

# Política IAM de Leitura e Escrita para o bucket do DVC e MLflow Artifacts
resource "aws_iam_policy" "dvc_rw_policy" {
  name        = "DVC-Bucket-ReadWrite-Policy"
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
}

# Usuário IAM para usar com o DVC
resource "aws_iam_user" "dvc_user" {
  name = var.iam_user_name

  tags = {
    Name        = "DVC User"
    Environment = "MLOps"
  }
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
  name        = "DVC-Bucket-ReadOnly-Policy"
  description = "Politica IAM de somente leitura para listar e obter objetos do bucket S3 do DVC"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
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
}

# Usuário IAM de somente leitura para o DVC
resource "aws_iam_user" "dvc_readonly_user" {
  name = "${var.iam_user_name}-readonly"

  tags = {
    Name        = "DVC ReadOnly User"
    Environment = "MLOps"
  }
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
