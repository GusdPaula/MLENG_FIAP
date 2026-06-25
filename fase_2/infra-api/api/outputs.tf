output "ecs_cluster_id" {
  description = "ID of the ECS cluster"
  value       = aws_ecs_cluster.api.id
}

output "ecs_service_name" {
  description = "Name of the ECS service"
  value       = aws_ecs_service.api.name
}

output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = aws_lb.api.dns_name
}

output "alb_arn" {
  description = "ARN of the Application Load Balancer"
  value       = aws_lb.api.arn
}

output "cloudfront_distribution_id" {
  description = "ID of the CloudFront distribution"
  value       = aws_cloudfront_distribution.api.id
}

output "cloudfront_domain_name" {
  description = "Domain name of the CloudFront distribution"
  value       = aws_cloudfront_distribution.api.domain_name
}

output "ecr_repository_url" {
  description = "URL of the ECR repository"
  value       = aws_ecr_repository.api.repository_url
}

output "api_url" {
  description = "URL to access the API"
  value       = var.use_custom_domain ? "https://${var.custom_domain_name}" : "https://${aws_cloudfront_distribution.api.domain_name}"
}

output "teacher_api_url" {
  description = "Teacher access URL for the API (use this for grading)"
  value       = var.use_custom_domain ? "https://${var.custom_domain_name}" : "https://${aws_cloudfront_distribution.api.domain_name}"
}

output "acm_validation_record_name" {
  description = "CNAME record name for ACM certificate validation"
  value       = var.use_custom_domain ? try(tolist(aws_acm_certificate.api_cert[0].domain_validation_options)[0].resource_record_name, "") : ""
}

output "acm_validation_record_value" {
  description = "CNAME record value for ACM certificate validation"
  value       = var.use_custom_domain ? try(tolist(aws_acm_certificate.api_cert[0].domain_validation_options)[0].resource_record_value, "") : ""
}
