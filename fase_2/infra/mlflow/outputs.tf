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

output "s3_artifacts_bucket" {
  description = "Name of the S3 artifacts bucket"
  value       = aws_s3_bucket.mlflow_artifacts.bucket
}
