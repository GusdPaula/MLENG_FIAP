variable "aws_region" {
  description = "AWS Region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project Name"
  type        = string
  default     = "mlflow-fiap"
}

variable "instance_type" {
  description = "EC2 Instance Type"
  type        = string
  default     = "t3.medium"
}

variable "db_instance_class" {
  description = "RDS Instance Class"
  type        = string
  default     = "db.t3.micro"
}

variable "dockerhub_username" {
  description = "Docker Hub Username"
  type        = string
  default     = "dummy"
}

variable "docker_image_tag" {
  description = "Docker Image Tag"
  type        = string
  default     = "latest"
}

variable "use_custom_domain" {
  description = "Whether to use custom domain with ACM certificate"
  type        = bool
  default     = false
}

variable "custom_domain_name" {
  description = "Custom domain name for MLflow"
  type        = string
  default     = "mlflow.asgardprint.com.br"
}

variable "bucket_name_mlflow_artifacts" {
  description = "Name of the S3 bucket for MLflow artifacts"
  type        = string
  default     = ""
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
