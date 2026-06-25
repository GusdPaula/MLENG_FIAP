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

variable "is_multi_region_trail" {
  description = "Whether to create a multi-region trail"
  type        = bool
  default     = true
}

variable "force_destroy_bucket" {
  description = "Whether to force destroy the S3 bucket even if it contains objects"
  type        = bool
  default     = false
}

variable "common_tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
  default     = {}
}
