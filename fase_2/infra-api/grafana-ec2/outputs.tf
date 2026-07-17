output "grafana_url" {
  description = "Grafana URL via CloudFront"
  value       = aws_cloudfront_distribution.grafana_distribution.domain_name
}

output "grafana_admin_username" {
  description = "Grafana admin username"
  value       = "admin"
}

output "grafana_admin_password" {
  description = "Grafana admin password"
  value       = random_password.grafana_admin.result
  sensitive   = true
}

output "grafana_alb_url" {
  description = "Grafana ALB URL"
  value       = aws_lb.grafana.dns_name
}

output "grafana_instance_id" {
  description = "Grafana EC2 instance ID"
  value       = aws_instance.grafana_server.id
}

output "grafana_instance_public_ip" {
  description = "Grafana EC2 instance public IP"
  value       = aws_instance.grafana_server.public_ip
}
