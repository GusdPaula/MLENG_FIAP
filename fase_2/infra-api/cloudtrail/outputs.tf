output "cloudtrail_arn" {
  description = "ARN of the CloudTrail"
  value       = aws_cloudtrail.main.arn
}

output "cloudtrail_home_region" {
  description = "Home region of the CloudTrail"
  value       = aws_cloudtrail.main.home_region
}

output "cloudtrail_s3_bucket_arn" {
  description = "ARN of the S3 bucket storing CloudTrail logs"
  value       = aws_s3_bucket.cloudtrail_bucket.arn
}

output "cloudtrail_s3_bucket_name" {
  description = "Name of the S3 bucket storing CloudTrail logs"
  value       = aws_s3_bucket.cloudtrail_bucket.id
}
