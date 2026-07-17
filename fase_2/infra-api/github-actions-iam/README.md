# GitHub Actions IAM User

This Terraform configuration creates an IAM user for GitHub Actions to manage AWS resources.

## Usage

### 1. Apply Terraform Configuration

```bash
cd infra-api/github-actions-iam
terraform init
terraform apply
```

### 2. Create Access Keys (Manual Step)

For security reasons, access keys must be created manually via the AWS Console:

1. Go to AWS Console → IAM → Users
2. Find the user `github-actions-mlflow-fiap`
3. Click on the user → Security credentials tab
4. Click "Create access key"
5. Select "Application running outside AWS" as the use case
6. Click "Next"
7. Add a description tag (optional): "GitHub Actions CI/CD"
8. Click "Create access key"
9. **IMPORTANT**: Copy and save the Access Key ID and Secret Access Key
   - You won't be able to see the Secret Access Key again!

### 3. Add Secrets to GitHub

1. Go to your GitHub repository
2. Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Add the following secrets:

   - **Name**: `AWS_ACCESS_KEY_ID`
   - **Value**: The Access Key ID you copied

   - **Name**: `AWS_SECRET_ACCESS_KEY`
   - **Value**: The Secret Access Key you copied

## Permissions

The IAM user has the following permissions:

- **EC2**: Start/Stop instances, Describe instances
- **ECS**: Update services, Describe services/tasks
- **RDS**: Start/Stop DB instances, Describe DB instances
- **SSM**: Send commands, Get command invocation

These permissions are scoped to only the actions needed by the GitHub Actions workflows.

## Security Best Practices

- **Least Privilege**: The user only has permissions for specific actions
- **No Console Access**: This user is meant for programmatic access only
- **Rotate Keys**: Rotate access keys regularly (recommended every 90 days)
- **Monitor Usage**: Use CloudTrail to monitor the user's activity
- **Delete Unused Keys**: If you need to recreate keys, delete old ones first

## Troubleshooting

### Access Denied Errors
- Verify the access keys are correctly added to GitHub Secrets
- Check that the IAM user has the correct policy attached
- Ensure the AWS region in workflows matches your resources

### Key Creation Issues
- If you lose the Secret Access Key, you must create a new access key
- Delete old access keys before creating new ones for security
- Ensure you have the necessary IAM permissions to create access keys

## Cleanup

To remove the IAM user:

```bash
terraform destroy
```

**Before destroying**, ensure you:
1. Delete the access keys from the AWS Console
2. Remove the secrets from GitHub repository
3. Update any workflows that might be using these credentials
