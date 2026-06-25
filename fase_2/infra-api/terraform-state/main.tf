terraform {
  required_version = ">= 1.0.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

# --- Terraform State Bucket Security ---
resource "aws_s3_bucket_versioning" "terraform_state" {
  bucket = "terraform-state-mlflow-fiap-ulodyq7a"

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "terraform_state" {
  bucket = "terraform-state-mlflow-fiap-ulodyq7a"

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "terraform_state" {
  bucket = "terraform-state-mlflow-fiap-ulodyq7a"

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_policy" "terraform_state" {
  bucket = "terraform-state-mlflow-fiap-ulodyq7a"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "EnforceTLS"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource  = [
          "arn:aws:s3:::terraform-state-mlflow-fiap-ulodyq7a",
          "arn:aws:s3:::terraform-state-mlflow-fiap-ulodyq7a/*"
        ]
        Condition = {
          Bool = {
            "aws:SecureTransport" = "false"
          }
        }
      },
      {
        Sid       = "DenyNonTerraformDelete"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:DeleteObject"
        Resource  = "arn:aws:s3:::terraform-state-mlflow-fiap-ulodyq7a/*"
        Condition = {
          StringNotLike = {
            "aws:UserAgent" = ["Terraform/*"]
          }
        }
      }
    ]
  })
}

# --- DynamoDB Table for State Locking ---
resource "aws_dynamodb_table" "terraform_locks" {
  name           = "terraform-locks"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }

  server_side_encryption {
    enabled = true
  }

  point_in_time_recovery {
    enabled = true
  }

  tags = {
    Name        = "Terraform State Lock Table"
    Environment = "production"
    ManagedBy   = "Terraform"
  }
}
