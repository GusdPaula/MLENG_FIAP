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
  default     = "t3.micro"
}

variable "db_instance_class" {
  description = "RDS Instance Class"
  type        = string
  default     = "db.t4g.micro"
}
