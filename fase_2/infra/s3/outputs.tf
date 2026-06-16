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

output "dvc_user_name" {
  value       = aws_iam_user.dvc_user.name
  description = "Nome do usuário IAM criado"
}

output "dvc_user_arn" {
  value       = aws_iam_user.dvc_user.arn
  description = "ARN do usuário IAM criado"
}

output "dvc_readonly_policy_arn" {
  value       = aws_iam_policy.dvc_readonly_policy.arn
  description = "ARN da política IAM de somente leitura para o bucket do DVC"
}

output "dvc_readonly_user_name" {
  value       = aws_iam_user.dvc_readonly_user.name
  description = "Nome do usuário IAM de somente leitura"
}

output "dvc_readonly_user_arn" {
  value       = aws_iam_user.dvc_readonly_user.arn
  description = "ARN do usuário IAM de somente leitura"
}
