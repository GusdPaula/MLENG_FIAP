# Grafana EC2 Module

This module deploys a self-hosted Grafana on EC2 for monitoring and log tracking of the ML recommendation system infrastructure.

## Overview

The Grafana EC2 module provides comprehensive monitoring capabilities including:
- CloudWatch Logs integration for log aggregation and search
- CloudWatch Metrics integration for infrastructure metrics
- **Anonymous access (no authentication required) - like MLflow**
- Pre-configured dashboards for API, MLflow, Infrastructure, Model Performance, and Security monitoring
- Prometheus metrics endpoint for custom application metrics

## Resources Created

- EC2 Instance running Grafana with anonymous access enabled
- Application Load Balancer (ALB) for HTTPS access
- CloudFront Distribution for CDN and global access
- Security Groups for Grafana and ALB
- IAM Role with CloudWatch permissions
- Nginx reverse proxy with SSL

## Usage

### Basic Usage

```bash
cd grafana-ec2
terraform init
terraform apply -var="project_name=ml-recommender"
```

### Accessing Grafana

After deployment, access Grafana using:
- **CloudFront URL**: From `terraform output grafana_url` (recommended)
- **ALB URL**: From `terraform output grafana_alb_url`
- **No authentication required** - anonymous viewer access is enabled

## Variables

| Variable | Description | Type | Default |
|----------|-------------|------|---------|
| `project_name` | Project name used for resource naming | string | required |
| `aws_region` | AWS region | string | `"us-east-1"` |
| `instance_type` | EC2 instance type for Grafana | string | `"t3.medium"` |
| `common_tags` | Common tags to apply to all resources | map(string) | `{Project="ML-Recommender", Environment="production"}` |

## Outputs

| Output | Description |
|--------|-------------|
| `grafana_url` | Grafana URL via CloudFront |
| `grafana_alb_url` | Grafana ALB URL |
| `grafana_instance_id` | Grafana EC2 instance ID |
| `grafana_instance_public_ip` | Grafana EC2 instance public IP |

## Authentication

### Anonymous Access (Academic Project)

Grafana is configured with anonymous viewer access, allowing anyone to view dashboards without authentication. This is similar to the MLflow server setup for this academic project.

- **Role**: Viewer (read-only access)
- **No login required**: Anyone can access the dashboards
- **Configuration**: Set in `/etc/grafana/grafana.ini` via user_data script

## Data Sources

Grafana is pre-configured with:
- **CloudWatch** - For AWS metrics and logs

## Cost Estimation

- **EC2 (t3.micro)**: ~$8/month (default for academic project)
- **EC2 (t3.medium)**: ~$25/month (for heavier workloads)
- **ALB**: ~$20/month
- **CloudFront**: ~$5-10/month (depending on usage)
- **Data Transfer**: Varies by usage
- **Total estimated cost**: ~$35-45/month (with t3.micro)

## Security

- IAM-based authentication with least privilege
- Nginx reverse proxy with SSL/TLS
- Security groups limiting access
- Anonymous viewer access only (no write permissions)

## Maintenance

### Regular Tasks

- Review and update dashboards monthly
- Monitor EC2 instance health
- Check SSL certificate expiration (self-signed for internal use)
- Monitor CloudFront and ALB costs

### Dashboard Updates

- Add new metrics as application evolves
- Remove obsolete panels
- Optimize query performance

## Troubleshooting

### Grafana Not Accessible

1. Check EC2 instance status in AWS Console
2. Verify security group rules
3. Check Nginx logs: `sudo tail -f /var/log/nginx/error.log`
4. Check Grafana logs: `sudo tail -f /var/log/grafana/grafana.log`

### Data Source Connection Issues

1. Verify IAM role has CloudWatch permissions
2. Check data source configuration in Grafana
3. Test connectivity from EC2 instance

## Comparison with AWS Managed Grafana

This self-hosted solution provides:
- **Anonymous access**: No authentication required (like MLflow)
- **Full control**: Complete configuration flexibility
- **Lower cost**: No managed service premium
- **Academic-friendly**: Simple setup for educational purposes

AWS Managed Grafana (alternative module):
- Requires authentication (no anonymous access)
- Managed service with less control
- Higher cost but less maintenance
- Enterprise features (SSO, etc.)

## Next Steps

1. Deploy the Grafana EC2 infrastructure using Terraform
2. Access Grafana via the CloudFront URL from outputs
3. Import dashboard JSON files from `dashboards/` directory:
   - `api-overview.json` - API metrics dashboard
   - `infrastructure.json` - Infrastructure metrics
   - `mlflow.json` - MLflow metrics
   - `model-performance.json` - Model performance metrics
   - `security.json` - Security metrics
   - `api-logs.json` - API logs (CloudWatch Logs)
   - `dvc-logs.json` - DVC logs (CloudWatch Logs)
   - `mlflow-logs.json` - MLflow logs (CloudWatch Logs)
4. Configure alert channels (email, Slack, etc.)
5. Set up alert rules based on SLA requirements

## Log Monitoring Setup

### Importing Log Dashboards

1. Access Grafana at the CloudFront URL
2. Go to Dashboards → Import
3. Upload the log dashboard JSON files:
   - `dashboards/api-logs.json`
   - `dashboards/dvc-logs.json`
   - `dashboards/mlflow-logs.json`
4. Select "CloudWatch Logs" as the data source

### Manual Data Source Configuration

If the CloudWatch Logs data source was not configured automatically during instance creation, you can configure it manually by SSHing into the Grafana EC2 instance:

1. Get the EC2 instance public IP from Terraform outputs:
   ```bash
   terraform output grafana_instance_public_ip
   ```

2. SSH into the instance:
   ```bash
   ssh -i <your-key.pem> ubuntu@<instance-public-ip>
   ```

3. Run the configuration script:
   ```bash
   sudo bash /tmp/configure_datasource.sh
   ```

   Or manually execute the commands:
   ```bash
   GRAFANA_URL="http://localhost:3000"
   GRAFANA_API_KEY=$(curl -s -X POST -H "Content-Type: application/json" \
     -d '{"name":"setup-key","role":"Admin"}' \
     $GRAFANA_URL/api/auth/keys | jq -r '.key')

   curl -s -X POST -H "Authorization: Bearer $GRAFANA_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "name":"CloudWatch Logs",
       "type":"cloudwatch-logs",
       "access":"proxy",
       "jsonData":{
         "authType":"default",
         "defaultRegion":"us-east-1"
       }
     }' \
     $GRAFANA_URL/api/datasources
   ```

4. Verify the data source is configured:
   ```bash
   curl -s $GRAFANA_URL/api/datasources
   ```

### Verifying Log Data Sources

1. Go to Configuration → Data Sources
2. Verify "CloudWatch Logs" is configured
3. Test the connection to ensure it's working
4. Check that the IAM role has CloudWatch Logs permissions

### Configuring Services to Send Logs

**API (ECS):**
- Ensure ECS task definition has CloudWatch log driver configured
- Log group should be named like `/ecs/api-logs`

**MLflow (EC2):**
- MLflow should be configured to send logs to CloudWatch
- Log group should be named like `/mlflow-logs`

**DVC:**
- Configure DVC to send logs to CloudWatch
- Log group should be named like `/dvc-logs`

## References

- [Grafana Documentation](https://grafana.com/docs/)
- [Grafana Anonymous Access](https://grafana.com/docs/grafana/latest/setup-grafana/configure-security/configure-authentication/anonymous-authentication/)
- [CloudWatch Plugin](https://grafana.com/plugins/grafana-aws-cloudwatch-datasource/)
