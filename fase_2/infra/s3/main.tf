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
