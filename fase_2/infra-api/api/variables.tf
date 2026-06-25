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

variable "cpu" {
  description = "CPU units for the task (256 = 0.25 vCPU, 512 = 0.5 vCPU, 1024 = 1 vCPU)"
  type        = number
  default     = 512
}

variable "memory" {
  description = "Memory for the task in MB"
  type        = number
  default     = 1024
}

variable "desired_count" {
  description = "Number of tasks to run"
  type        = number
  default     = 1
}

variable "docker_image_tag" {
  description = "Docker Image Tag"
  type        = string
  default     = "latest"
}

variable "mlflow_tracking_uri" {
  description = "MLflow Tracking URI for the API to connect to"
  type        = string
  default     = "https://d2i3ddfklul6yu.cloudfront.net/"
}

variable "api_key" {
  description = "API Key for authentication"
  type        = string
  sensitive   = true
}

variable "mlflow_model_alias" {
  description = "MLflow model alias to use (e.g., champion, staging)"
  type        = string
  default     = "champion"
}

variable "use_custom_domain" {
  description = "Whether to use custom domain with ACM certificate"
  type        = bool
  default     = false
}

variable "custom_domain_name" {
  description = "Custom domain name for API"
  type        = string
  default     = "api.asgardprint.com.br"
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
