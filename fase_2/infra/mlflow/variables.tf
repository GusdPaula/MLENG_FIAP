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
  default     = "db.t4g.micro"
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
  default     = true
}

variable "custom_domain_name" {
  description = "Custom domain name for MLflow"
  type        = string
  default     = "mlflow.asgardprint.com.br"
}



