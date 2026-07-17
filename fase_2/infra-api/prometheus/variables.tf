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

variable "ami_id" {
  description = "AMI ID for Prometheus EC2 instance"
  type        = string
  default     = "ami-0c55b159cbfafe1f0" # Ubuntu 22.04 LTS in us-east-1
}

variable "instance_type" {
  description = "EC2 instance type for Prometheus"
  type        = string
  default     = "t3.micro"
}

variable "api_alb_dns" {
  description = "DNS name of the API Application Load Balancer"
  type        = string
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
