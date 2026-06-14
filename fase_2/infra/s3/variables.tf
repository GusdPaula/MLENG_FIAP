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

variable "iam_user_name" {
  type        = string
  description = "Nome do usuario IAM para o DVC"
  default     = "fiap-dvc-user"
}
