# Teacher Access Guide

This guide provides instructions for teachers/reviewers to access the MLflow and API applications for grading purposes.

## Overview

The infrastructure includes a dedicated IAM user with read-only access for teachers to review the project without risk of accidentally modifying resources.

## Access URLs

After deploying the infrastructure, run `terraform output` in each module to get the access URLs:

### API Endpoint
```bash
cd api
terraform output teacher_api_url
```
This will return the URL to access the API (e.g., `https://xxxxxxxx.cloudfront.net`)

### MLflow UI
```bash
cd mlflow
terraform output teacher_mlflow_url
```
This will return the URL to access the MLflow UI (e.g., `https://xxxxxxxx.cloudfront.net`)

## AWS Credentials Setup

### Step 1: Get the Reviewer User Name
```bash
cd s3
terraform output reviewer_user_name
```

This will return something like `fiap-dvc-user-reviewer-abc12345`

### Step 2: Create Access Keys for the Reviewer User

**Option A: Using AWS Console**
1. Log in to AWS Console
2. Navigate to IAM → Users
3. Find the reviewer user (from Step 1)
4. Click "Security credentials" tab
5. Click "Create access key"
6. Select "Application running outside AWS" or "Command Line Interface (CLI)"
7. Create and download the access key (Access Key ID and Secret Access Key)

**Option B: Using AWS CLI** (if you have admin access)
```bash
aws iam create-access-key --user-name $(cd s3 && terraform output reviewer_user_name)
```

### Step 3: Configure AWS CLI

Install AWS CLI if not already installed:
```bash
# On macOS
brew install awscli

# On Linux
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

Configure credentials:
```bash
aws configure --profile teacher-reviewer
```

Enter the credentials when prompted:
- AWS Access Key ID: [from Step 2]
- AWS Secret Access Key: [from Step 2]
- Default region: `us-east-1`
- Default output format: `json`

### Step 4: Verify Access

Test S3 access (read-only):
```bash
aws s3 ls --profile teacher-reviewer
```

Test listing the DVC bucket:
```bash
aws s3 ls s3://$(cd s3 && terraform output bucket_name) --profile teacher-reviewer
```

## Accessing the Applications

### API Access

The API is accessible via CloudFront. No authentication is required for basic endpoints.

**Test the API:**
```bash
# Get API URL
API_URL=$(cd api && terraform output teacher_api_url)

# Test health endpoint
curl https://$API_URL/health

# Test other endpoints as documented in the API documentation
curl https://$API_URL/predict
```

### MLflow UI Access

The MLflow UI is accessible via CloudFront. No authentication is configured by default.

**Access in Browser:**
1. Get the MLflow URL:
```bash
cd mlflow
terraform output teacher_mlflow_url
```

2. Open the URL in your browser
3. You should see the MLflow experiment tracking UI

**View Experiments:**
- Browse experiments and runs
- View metrics and parameters
- Download artifacts (read-only access to S3)

## S3 Data Access

The reviewer user has read-only access to:
- DVC bucket (datasets and model artifacts)
- MLflow artifacts bucket

**List DVC bucket contents:**
```bash
aws s3 ls s3://$(cd s3 && terraform output bucket_name) --recursive --profile teacher-reviewer
```

**Download a specific file:**
```bash
aws s3 cp s3://$(cd s3 && terraform output bucket_name)/path/to/file . --profile teacher-reviewer
```

**Sync entire bucket (for review):**
```bash
aws s3 sync s3://$(cd s3 && terraform output bucket_name) ./dvc-review --profile teacher-reviewer
```

## What You Can Review

### Read-Only Access Includes:
- ✅ List and view all S3 objects (datasets, models, artifacts)
- ✅ Access MLflow UI and view all experiments
- ✅ Access API endpoints
- ✅ View infrastructure via AWS Console (if given console access)

### Restricted Actions:
- ❌ Cannot modify S3 objects
- ❌ Cannot delete resources
- ❌ Cannot modify MLflow experiments
- ❌ Cannot change infrastructure configuration

## Troubleshooting

### Access Denied Errors
If you see "Access Denied" errors:
1. Verify you're using the correct profile: `--profile teacher-reviewer`
2. Check that the access keys are valid and not expired
3. Ensure the reviewer user exists: `aws iam get-user --user-name $(cd s3 && terraform output reviewer_user_name) --profile default`

### CloudFront URL Not Working
1. Wait 5-10 minutes after deployment for CloudFront to propagate
2. Check that the distribution is deployed: `aws cloudfront get-distribution --id $(cd api && terraform output cloudfront_distribution_id)`
3. Verify the distribution status is "Deployed"

### MLflow UI Not Loading
1. Check EC2 instance is running: `aws ec2 describe-instance-status --instance-ids $(cd mlflow && terraform output ec2_instance_id)`
2. Verify security group allows CloudFront access
3. Check CloudWatch logs for MLflow server errors

## Security Notes

- The reviewer user has **read-only** access to prevent accidental modifications
- Access keys should be rotated periodically (every 90 days recommended)
- After grading, consider disabling or deleting the reviewer user
- Never share access keys via email or unencrypted channels
- Use AWS IAM Access Analyzer to review access periodically

## Cleanup After Grading

To revoke teacher access after grading:

```bash
# Delete access keys
aws iam list-access-keys --user-name $(cd s3 && terraform output reviewer_user_name) --query 'AccessKeyMetadata[].AccessKeyId' --output text | xargs -I {} aws iam delete-access-key --access-key-id {} --user-name $(cd s3 && terraform output reviewer_user_name)

# Or delete the entire user (if no longer needed)
aws iam delete-user --user-name $(cd s3 && terraform output reviewer_user_name)
```

## Contact

If you encounter issues accessing the application for grading, contact the project team with:
- The specific error message
- The command or URL you tried to access
- Your AWS CLI version (`aws --version`)
