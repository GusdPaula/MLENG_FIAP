output "bucket_name" {
  value       = aws_s3_bucket.dvc_bucket.id
  description = "Nome do bucket S3 criado"
}

output "bucket_arn" {
  value       = aws_s3_bucket.dvc_bucket.arn
  description = "ARN do bucket S3 criado"
}

output "dvc_remote_url" {
  value       = "s3://${aws_s3_bucket.dvc_bucket.id}"
  description = "A URL do remote do DVC correspondente ao S3"
}

output "dvc_rw_policy_arn" {
  value       = aws_iam_policy.dvc_rw_policy.arn
  description = "ARN da política IAM de Leitura/Escrita para o bucket do DVC"
}
