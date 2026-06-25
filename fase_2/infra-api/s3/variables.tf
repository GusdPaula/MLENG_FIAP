variable "aws_region" {
  type        = string
  description = "AWS region to deploy resources"
  default     = "us-east-1"
}

variable "bucket_name" {
  type        = string
  description = "Name of the S3 bucket for DVC storage"
  default     = "fiap-ml-dvc-bucket-tech-challenger"
}

variable "bucket_name_mlflow_artifacts" {
  type        = string
  description = "Name of the S3 bucket for MLflow artifacts"
  default     = "mlflow-artifacts-fiap-rsnnnlwu"
}

variable "iam_user_name" {
  type        = string
  description = "Nome do usuario IAM para o DVC"
  default     = "fiap-dvc-user"
}

variable "common_tags" {
  type        = map(string)
  description = "Common tags to apply to all resources"
  default = {
    Project     = "MLflow FIAP"
    Environment = "Production"
    ManagedBy   = "Terraform"
    Owner       = "MLOps Team"
    CostCenter  = "Engineering"
  }
}
