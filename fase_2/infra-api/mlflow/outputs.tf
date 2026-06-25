output "ec2_instance_id" {
  description = "ID of the EC2 instance"
  value       = aws_instance.mlflow_server.id
}

output "rds_instance_id" {
  description = "ID of the RDS instance"
  value       = aws_db_instance.mlflow_db.identifier
}

output "cloudfront_distribution_id" {
  description = "ID of the CloudFront distribution"
  value       = aws_cloudfront_distribution.mlflow_distribution.id
}

output "cloudfront_domain_name" {
  description = "Domain name of the CloudFront distribution"
  value       = aws_cloudfront_distribution.mlflow_distribution.domain_name
}

output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = aws_lb.mlflow.dns_name
}

output "alb_arn" {
  description = "ARN of the Application Load Balancer"
  value       = aws_lb.mlflow.arn
}

output "teacher_mlflow_url" {
  description = "Teacher access URL for MLflow UI (use this for grading)"
  value       = "https://${aws_cloudfront_distribution.mlflow_distribution.domain_name}"
}

output "s3_artifacts_bucket" {
  description = "Name of the S3 artifacts bucket"
  value       = aws_s3_bucket.mlflow_artifacts.bucket
}

output "acm_validation_record_name" {
  description = "CNAME record name for ACM certificate validation"
  value       = var.use_custom_domain ? try(tolist(aws_acm_certificate.mlflow_cert[0].domain_validation_options)[0].resource_record_name, "") : ""
}

output "acm_validation_record_value" {
  description = "CNAME record value for ACM certificate validation"
  value       = var.use_custom_domain ? try(tolist(aws_acm_certificate.mlflow_cert[0].domain_validation_options)[0].resource_record_value, "") : ""
}
