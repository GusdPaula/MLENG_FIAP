terraform {
  required_version = ">= 1.0.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

# IAM User for GitHub Actions
resource "aws_iam_user" "github_actions" {
  name = "github-actions-mlflow-fiap"

  tags = {
    Name        = "GitHub Actions User"
    Purpose     = "CI/CD Resource Management"
    Environment = "production"
  }
}

# IAM Policy for GitHub Actions
resource "aws_iam_policy" "github_actions_policy" {
  name        = "GitHubActionsResourceManagement"
  description = "Policy for GitHub Actions to manage AWS resources"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ec2:DescribeInstances",
          "ec2:StartInstances",
          "ec2:StopInstances"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecs:DescribeServices",
          "ecs:UpdateService",
          "ecs:DescribeTasks"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "rds:DescribeDBInstances",
          "rds:StartDBInstance",
          "rds:StopDBInstance"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ssm:SendCommand",
          "ssm:GetCommandInvocation"
        ]
        Resource = "*"
      }
    ]
  })

  tags = {
    Name        = "GitHub Actions Policy"
    Purpose     = "CI/CD Resource Management"
  }
}

# Attach policy to user
resource "aws_iam_user_policy_attachment" "github_actions_attach" {
  user       = aws_iam_user.github_actions.name
  policy_arn = aws_iam_policy.github_actions_policy.arn
}

# Output access key (you'll need to create this manually via AWS Console)
output "github_actions_user_name" {
  description = "GitHub Actions IAM user name"
  value       = aws_iam_user.github_actions.name
}

output "github_actions_user_arn" {
  description = "GitHub Actions IAM user ARN"
  value       = aws_iam_user.github_actions.arn
}
