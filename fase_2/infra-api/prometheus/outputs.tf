output "prometheus_public_ip" {
  description = "Public IP address of Prometheus EC2 instance"
  value       = aws_instance.prometheus.public_ip
}

output "prometheus_url" {
  description = "URL to access Prometheus"
  value       = "http://${aws_instance.prometheus.private_ip}:9090"
}

output "prometheus_instance_id" {
  description = "Instance ID of Prometheus EC2"
  value       = aws_instance.prometheus.id
}
