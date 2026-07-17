# GitHub Actions for AWS Resource Management

This directory contains GitHub Actions workflows to manage AWS resources for cost optimization.

## Workflows

### Manual Workflows

#### Start Resources
- **File**: `start-resources.yml`
- **Trigger**: Manual (workflow_dispatch)
- **Action**: Starts MLflow, Grafana, Prometheus EC2 instances and ECS service
- **Optional**: Can start RDS instance if checkbox is enabled

#### Stop Resources
- **File**: `stop-resources.yml`
- **Trigger**: Manual (workflow_dispatch)
- **Action**: Stops MLflow, Grafana, Prometheus EC2 instances and ECS service
- **Optional**: Can stop RDS instance if checkbox is enabled

### Scheduled Workflows

#### Scheduled Start
- **File**: `scheduled-start.yml`
- **Trigger**: Scheduled (8 AM UTC on weekdays)
- **Action**: Automatically starts all resources in the morning

#### Scheduled Stop
- **File**: `scheduled-stop.yml`
- **Trigger**: Scheduled (8 PM UTC on weekdays)
- **Action**: Automatically stops all resources in the evening for cost savings

## Setup Instructions

### 1. Create AWS IAM User

Create an IAM user with the following permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeInstances",
        "ec2:StartInstances",
        "ec2:StopInstances",
        "ecs:DescribeServices",
        "ecs:UpdateService",
        "rds:DescribeDBInstances",
        "rds:StartDBInstance",
        "rds:StopDBInstance",
        "ssm:SendCommand",
        "ssm:GetCommandInvocation"
      ],
      "Resource": "*"
    }
  ]
}
```

### 2. Generate AWS Access Keys

1. Go to AWS Console → IAM → Users → Select your user
2. Security credentials tab → Create access key
3. Save the Access Key ID and Secret Access Key

### 3. Add Secrets to GitHub Repository

1. Go to your GitHub repository
2. Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Add the following secrets:

   - **Name**: `AWS_ACCESS_KEY_ID_API`
   - **Value**: Your AWS Access Key ID

   - **Name**: `AWS_SECRET_ACCESS_KEY_API`
   - **Value**: Your AWS Secret Access Key

### 4. Enable Workflows

The workflows are now ready to use:

- Go to Actions tab in your repository
- Select the workflow you want to run
- Click "Run workflow" button
- For start/stop workflows, you can optionally enable RDS start/stop

## Cost Savings

The scheduled workflows provide automatic cost savings by:
- Starting resources at 8 AM UTC (4 AM EST) on weekdays
- Stopping resources at 8 PM UTC (5 PM EST) on weekdays
- Keeping resources off on weekends
- RDS remains running to preserve data (can be manually stopped if needed)

## Security Notes

- **Least Privilege**: The IAM user only has permissions for the specific actions needed
- **Secrets Management**: AWS credentials are stored as GitHub Secrets, never in code
- **No RDS Auto-Stop**: RDS is not stopped by default to prevent data loss
- **SSM Access**: Workflows use AWS Systems Manager for service management

## Troubleshooting

### Workflow Fails with "Access Denied"
- Verify AWS credentials are correct
- Check IAM user has required permissions
- Ensure credentials haven't expired

### Instances Not Starting/Stopping
- Check if instances are in a transition state
- Verify instance tags match the script expectations
- Check CloudTrail logs for detailed error information

### Scheduled Workflows Not Running
- Verify GitHub Actions is enabled for the repository
- Check timezone settings (cron uses UTC)
- Ensure repository has at least one commit

## Customization

### Change Schedule Times

Edit the `cron` expression in the workflow files:

```yaml
schedule:
  - cron: '0 20 * * 1-5'  # 8 PM UTC, Mon-Fri
```

Cron format: `minute hour day month day-of-week`

### Add More Resources

Update the corresponding Python scripts in `scripts/` directory and the workflows will automatically use the changes.

### Disable Scheduled Workflows

Comment out the `schedule` section in the workflow files to disable automatic execution while keeping manual triggers available.
