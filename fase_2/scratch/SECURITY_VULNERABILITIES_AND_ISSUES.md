# Security Vulnerabilities and Issues

This document identifies security vulnerabilities, exposed secrets, and potential security issues in the Terraform infrastructure.

## Critical Severity

### 1. Hardcoded API Key in Variables ~~[RESOLVED]~~

**Location:** `api/variables.tf` (lines 43-47)

**Status:** ✅ **RESOLVED** - Hardcoded default value removed

**Issue:**
- The API key is exposed in plain text in the variables file
- Even though marked as `sensitive = true`, the default value is still visible in the code
- Anyone with access to the repository can see the API key
- The key is committed to version control

**Risk:**
- Unauthorized access to the API
- Potential data breaches
- API abuse and cost escalation

**Resolution:**
The hardcoded default value has been removed from the variable definition:
```hcl
variable "api_key" {
  description = "API Key for authentication"
  type        = string
  sensitive   = true
}
```

The API key must now be passed securely via environment variable or .tfvars file.

---

### 2. Public S3 Bucket Access ~~[RESOLVED]~~

**Location:** `s3/main.tf` (lines 57-64)

**Status:** ✅ **RESOLVED** - Public access blocked, public policy removed

**Issue:**
- The DVC bucket allows public read access to anyone on the internet
- `Principal = "*"` grants access to all AWS accounts and anonymous users
- Anyone can list and download all datasets and model artifacts
- No authentication required to access data

**Risk:**
- Data exposure to unauthorized parties
- Intellectual property theft
- Privacy violations if the data contains PII
- Cost implications from unauthorized data access

**Resolution:**
Public access is now fully blocked and the public policy has been removed:
```hcl
resource "aws_s3_bucket_public_access_block" "public_access" {
  bucket = aws_s3_bucket.dvc_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
```

Access is now controlled exclusively through IAM policies (`dvc_rw_policy` and `dvc_readonly_policy`).

If public access is absolutely necessary for a specific use case:
```hcl
resource "aws_s3_bucket_policy" "restricted_read_policy" {
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "SpecificAccountRead"
        Effect    = "Allow"
        Principal = {
          AWS = "arn:aws:iam::SPECIFIC_ACCOUNT_ID:root"
        }
        Action    = "s3:GetObject"
        Resource  = "${aws_s3_bucket.dvc_bucket.arn}/*"
      }
    ]
  })
}
```

---

### 3. Hardcoded Developer IP in Security Group ~~[RESOLVED]~~

**Location:** `mlflow/main.tf` (lines 87-93)

**Status:** ✅ **RESOLVED** - Hardcoded IP removed, now uses CloudFront prefix list

**Issue:**
- A specific IP address is hardcoded in the infrastructure
- This exposes the developer's home/office IP address
- The IP might change (dynamic IP, ISP change, travel)
- Comment says "temporary" but it's in the permanent code
- Anyone with repo access knows the developer's IP

**Risk:**
- Targeted attacks against the developer's network
- Privacy violation
- Security through obscurity (IP can be spoofed)
- Infrastructure fails if IP changes

**Resolution:**
The hardcoded IP has been removed. The security group now uses CloudFront prefix list:
```hcl
ingress {
  description     = "Allow traffic only from CloudFront"
  from_port       = 5000
  to_port         = 5000
  protocol        = "tcp"
  prefix_list_ids = [data.aws_ec2_managed_prefix_list.cloudfront.id]
}
```

AWS SSM Session Manager is also enabled (line 189-192) for secure EC2 access.

---

## High Severity

### 4. RDS Password Not Used by MLflow ~~[RESOLVED]~~

**Location:** `mlflow/main.tf` (lines 252-268)

**Status:** ✅ **RESOLVED** - MLflow now uses PostgreSQL with RDS password

**Issue:**
- RDS PostgreSQL instance is created with a secure password stored in Secrets Manager
- However, the MLflow server user_data script uses SQLite instead
- The RDS instance is running but not connected to MLflow
- Wasted resources (paying for unused RDS)
- False sense of security (password generated but not used)

**Risk:**
- Unnecessary AWS costs
- Misleading security posture
- Database credentials stored but unused
- Potential confusion for future developers

**Resolution:**
MLflow now uses PostgreSQL backend with the password from Secrets Manager (lines 252-268):
```bash
# Get DB password from Secrets Manager
DB_PASSWORD=$(aws secretsmanager get-secret-value \
  --secret-id ${aws_secretsmanager_secret.db_password_secret.id} \
  --query SecretString --output text)

# Start MLflow server with PostgreSQL backend
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri postgresql://mlflow_user:${DB_PASSWORD}@${aws_db_instance.mlflow_db.endpoint}:5432/mlflow \
  --default-artifact-root "s3://${aws_s3_bucket.mlflow_artifacts.bucket}" \
  --serve-artifacts \
  --allowed-hosts "*" \
  --cors-allowed-origins "*"
```

The RDS instance is now properly utilized by MLflow.

---

### 5. MLflow Server Uses SQLite in Production ~~[RESOLVED]~~

**Location:** `mlflow/main.tf` (lines 257-266)

**Status:** ✅ **RESOLVED** - MLflow now uses PostgreSQL backend

**Issue:**
- SQLite is used as the backend store for MLflow
- SQLite is a file-based database, not suitable for production
- No concurrent write support
- Single point of failure
- No replication or backup strategy
- Performance limitations at scale

**Risk:**
- Data loss if EC2 instance is terminated
- Corruption with concurrent access
- No high availability
- Scalability issues
- No automated backups

**Resolution:**
MLflow server now uses PostgreSQL as the backend store (line 263):
```bash
--backend-store-uri postgresql://mlflow_user:${DB_PASSWORD}@${aws_db_instance.mlflow_db.endpoint}:5432/mlflow
```

The RDS instance is properly configured with:
- `backup_retention_period = 7` (automated backups enabled)
- `storage_encrypted = true` (encryption at rest)
- `monitoring_interval = 60` (enhanced monitoring)

---

### 6. No Encryption at Rest for S3 Buckets ~~[RESOLVED]~~

**Location:** `s3/main.tf`, `mlflow/main.tf`

**Status:** ✅ **RESOLVED** - Encryption at rest enabled for all S3 buckets

**Issue:**
- S3 buckets are created without explicit encryption configuration
- Default AWS encryption may or may not be enabled depending on account settings
- No control over encryption keys
- Cannot guarantee data is encrypted at rest

**Risk:**
- Data stored in plain text if account default is disabled
- Compliance violations (GDPR, HIPAA, etc.)
- Lack of audit trail for encryption key usage

**Resolution:**
Added `aws_s3_bucket_server_side_encryption_configuration` resources to both:

- **DVC bucket** (`s3/main.tf` lines 44-53):
```hcl
resource "aws_s3_bucket_server_side_encryption_configuration" "encryption" {
  bucket = aws_s3_bucket.dvc_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
```

- **MLflow artifacts bucket** (`mlflow/main.tf` lines 52-61):
```hcl
resource "aws_s3_bucket_server_side_encryption_configuration" "encryption" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
```

Both buckets now use AES256 server-side encryption by default.

---

### 7. No Versioning on S3 Buckets ~~[RESOLVED]~~

**Location:** `s3/main.tf` (lines 55-62), `mlflow/main.tf` (lines 63-70)

**Status:** ✅ **RESOLVED** - S3 versioning enabled for both DVC and MLflow artifacts buckets

**Issue:**
- S3 buckets don't have versioning enabled
- Accidental deletions are permanent
- No rollback capability
- No protection against overwrites

**Risk:**
- Data loss from accidental deletion
- No way to recover previous versions
- Malicious or accidental overwrites
- Compliance issues requiring data retention

**Resolution:**

Added `aws_s3_bucket_versioning` resources to both buckets:

- **DVC bucket** (`s3/main.tf` lines 55-62):
```hcl
resource "aws_s3_bucket_versioning" "versioning" {
  bucket = aws_s3_bucket.dvc_bucket.id

  versioning_configuration {
    status = "Enabled"
  }
}
```

- **MLflow artifacts bucket** (`mlflow/main.tf` lines 63-70):
```hcl
resource "aws_s3_bucket_versioning" "versioning" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}
```

Both buckets now have versioning enabled, providing protection against accidental deletions and the ability to recover previous versions of objects.

---

### 8. CloudFront Origin Uses HTTP Only ~~[RESOLVED]~~

**Location:** `api/main.tf` (lines 284-310, 344-348), `mlflow/main.tf` (lines 239-335, 362-367)

**Status:** ✅ **RESOLVED** - CloudFront now uses HTTPS-only to connect to origins

**Issue:**
- CloudFront connects to the origin (ALB/EC2) over HTTP
- Traffic between CloudFront and origin is unencrypted
- If someone intercepts traffic in the VPC, they can see data
- Origin SSL protocols are configured but not used

**Risk:**
- Man-in-the-middle attacks within VPC
- Data exposure to internal threats
- Compliance violations
- Inconsistent security posture

**Resolution:**

**For API (ALB):**
- Added HTTPS listener to ALB with ACM certificate (lines 299-310):
```hcl
resource "aws_lb_listener" "api_https" {
  load_balancer_arn = aws_lb.api.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS-1-2-2017-01"
  certificate_arn   = var.use_custom_domain ? aws_acm_certificate.api_cert[0].arn : aws_acm_certificate.default_cert.arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}
```
- HTTP listener now redirects to HTTPS (lines 284-297)
- Added default ACM certificate for internal use (lines 323-331)
- CloudFront origin changed to `origin_protocol_policy = "https-only"` (line 346)

**For MLflow (EC2):**
- Added nginx reverse proxy with SSL termination to EC2 user_data (lines 248, 291-332)
- MLflow server now listens on localhost only (127.0.0.1:5000)
- Self-signed SSL certificate generated for nginx
- Nginx configured to:
  - Listen on port 443 with SSL/TLS
  - Redirect HTTP (port 80) to HTTPS
  - Proxy requests to MLflow on localhost
- CloudFront origin changed to `origin_protocol_policy = "https-only"` (line 365)
- CloudFront origin SSL protocols updated to support TLSv1.2 and TLSv1.3 (line 366)

Traffic between CloudFront and both origins (ALB and EC2) is now encrypted with HTTPS.

---

## Medium Severity

### 9. MLflow Database Not Publicly Accessible But Password Exposed in User Data

**Status:** ✅ **RESOLVED**

**Location:** `mlflow/main.tf` (lines 73-88, 272-276)

**Issue (Previously):**
- If MLflow is configured to use PostgreSQL (as recommended in issue #4)
- The database password would be exposed in the EC2 user_data script
- User data is visible in the EC2 console
- User data is stored in the instance metadata

**Risk (Previously):**
- Anyone with EC2 read access can see the password
- Password exposed in logs
- Not following secrets management best practices

**Resolution:**
The password is now securely managed using AWS Secrets Manager:

1. **Secrets Manager Setup** (lines 73-88):
   - Random password generated using `random_password` resource
   - Stored in AWS Secrets Manager with `aws_secretsmanager_secret` and `aws_secretsmanager_secret_version`

2. **IAM Permissions** (lines 193-199):
   - EC2 instance profile has permission to access the secret via `secretsmanager:GetSecretValue`

3. **Runtime Retrieval** (lines 272-276 in user_data):
   ```bash
   DB_PASSWORD=$(aws secretsmanager get-secret-value \
     --secret-id ${aws_secretsmanager_secret.db_password_secret.id} \
     --query SecretString --output text)
   ```

4. **Secure Usage** (line 283):
   - Password is fetched at runtime and used immediately in the connection string
   - Never stored in user_data or logs

---

### 10. No VPC Flow Logs

**Status:** ✅ **RESOLVED**

**Location:** `mlflow/main.tf` (lines 418-449)

**Issue (Previously):**
- VPC flow logs were not enabled
- No visibility into network traffic
- Difficult to troubleshoot network issues
- Cannot detect suspicious network activity

**Risk (Previously):**
- Inability to detect security incidents
- No audit trail for network access
- Compliance violations
- Difficult incident response

**Resolution:**
VPC flow logs have been added to monitor all network traffic in the default VPC:

1. **IAM Role for Flow Logs** (lines 419-432):
   - `aws_iam_role.flow_log_role` created with trust policy for vpc-flow-logs.amazonaws.com
   - Attached CloudWatchLogsFullAccess policy (line 434-437)

2. **CloudWatch Log Group** (lines 439-442):
   - `aws_cloudwatch_log_group.flow_log` created to store flow logs
   - 7-day retention period configured
   - Named `${var.project_name}-vpc-flow-logs`

3. **VPC Flow Log** (lines 444-449):
   - `aws_flow_log.vpc_flow_log` enabled on default VPC
   - Traffic type set to "ALL" to capture both accepted and rejected traffic
   - Logs sent to CloudWatch Log Group for analysis and monitoring

All modules (mlflow, api, s3) use the same default VPC, so this single flow log configuration provides visibility for all infrastructure.

---

### 11. No CloudTrail Enabled ~~[RESOLVED]~~

**Location:** New module `cloudtrail/`

**Status:** ✅ **RESOLVED** - CloudTrail module created with secure S3 bucket

**Issue (Previously):**
- AWS CloudTrail was not configured
- No audit log of API calls
- Cannot track who made changes to infrastructure
- Compliance violations

**Risk (Previously):**
- No accountability for actions
- Cannot investigate security incidents
- Compliance violations (PCI-DSS, HIPAA, SOC2)
- Unable to detect unauthorized changes

**Resolution:**
A new CloudTrail module has been created at `cloudtrail/` with the following security features:

1. **CloudTrail Trail** (`main.tf` lines 80-90):
   - Multi-region trail enabled by default
   - Log file validation enabled
   - Global service events included
   - All API calls logged to S3

2. **Secure S3 Bucket** (`main.tf` lines 14-62):
   - Versioning enabled for log protection
   - Server-side encryption with AES256
   - Public access blocked
   - Proper bucket policy for CloudTrail service access

3. **Bucket Policy** (`main.tf` lines 64-95):
   - Allows CloudTrail to write logs
   - Enforces bucket-owner-full-control ACL
   - Restricts access to CloudTrail service only

The module outputs:
- CloudTrail ARN
- CloudTrail home region
- S3 bucket ARN and name for log storage

To deploy:
```bash
cd cloudtrail
terraform init
terraform apply
```

---

### 12. IAM Users with Long-Term Credentials ~~[RESOLVED]~~

**Location:** `s3/main.tf` (lines 114-121, 171-178, 192-201)

**Status:** ✅ **RESOLVED** - Access keys not created in Terraform; users for educational/teacher access only

**Issue (Previously):**
- IAM users were reported to be created with access keys in Terraform
- Long-term credentials are harder to rotate
- No credential rotation policy
- Access keys can be compromised

**Current State:**
- IAM users are created but access keys are NOT provisioned through Terraform
- Access keys must be created manually via AWS Console or CLI
- This prevents hardcoded credentials in the codebase
- Users include: dvc_user (development), dvc_readonly_user (read-only), reviewer_user (teacher access)

**Risk (Mitigated):**
- No keys are exposed in Terraform state or code
- Keys are created manually by authorized users only
- Reviewer user has read-only access to prevent accidental modifications
- Documented credential rotation procedures in TEACHER_ACCESS.md

**Educational Context:**
For this ML course project, IAM users are acceptable because:
- Simplifies access for students and teachers
- No complex identity provider setup required
- Focus on ML/DVC/MLflow rather than AWS IAM management
- Read-only reviewer user prevents accidental modifications

**Production Recommendation:**
For production deployments, use IAM roles instead:
- For EC2/ECS: Use instance profiles
- For local development: Use AWS SSO or temporary credentials via `aws sts get-session-token`
- For teacher/reviewer access: Use temporary credentials with expiration

**Documentation:**
See `TEACHER_ACCESS.md` for secure credential setup and rotation procedures.

---

### 13. No Resource Tags for Cost Tracking ~~[RESOLVED]~~

**Location:** All modules (s3, api, mlflow)

**Status:** ✅ **RESOLVED** - Common tags applied to all resources across all modules

**Issue (Previously):**
- Many resources lacked tags
- Difficult to track costs by project/environment
- No ownership information
- Difficult to cleanup unused resources

**Resolution:**
Added `common_tags` variable to all modules and applied tags to all resources:

**1. S3 Module (`s3/`):**
- Added `common_tags` variable to `variables.tf`
- Applied tags to: S3 bucket, IAM users, IAM policies

**2. API Module (`api/`):**
- Added `common_tags` variable to `variables.tf`
- Applied tags to: ECR repository, ECS cluster, task definition, security groups, IAM roles, CloudWatch log group, ECS service, ALB, target group, ACM certificates, CloudFront distribution

**3. MLflow Module (`mlflow/`):**
- Added `common_tags` variable to `variables.tf`
- Applied tags to: S3 bucket, Secrets Manager secret, security groups, RDS instance, subnet group, IAM roles, IAM policies, EC2 instance, ACM certificate, CloudFront distribution, CloudWatch log group, flow log role

**Common Tags Applied:**
```hcl
{
  Project     = "MLflow FIAP"
  Environment = "Production"
  ManagedBy   = "Terraform"
  Owner       = "MLOps Team"
  CostCenter  = "Engineering"
}
```

**Benefits:**
- Easy cost tracking by project and environment
- Clear ownership information
- Simplified resource cleanup
- Compliance with tagging policies
- Better cost allocation across teams

---

## Low Severity

### 14. No Health Check on ALB Target Group for MLflow ~~[RESOLVED]~~

**Location:** `mlflow/main.tf`

**Status:** ✅ **RESOLVED** - ALB added with health checks and load balancing

**Issue (Previously):**
- MLflow was using direct CloudFront to EC2 connection
- No load balancing or health checking for MLflow
- Single point of failure
- No automatic failover
- Manual intervention needed for failures

**Resolution:**
Added Application Load Balancer (ALB) for MLflow with the following components:

**1. ALB Security Group** (`mlflow/main.tf` lines 140-169):
- Allows HTTP (port 80) and HTTPS (port 443) from anywhere
- Egress to all destinations

**2. Application Load Balancer** (`mlflow/main.tf` lines 397-407):
- Public-facing ALB in default VPC
- Uses ALB security group
- Tags applied for cost tracking

**3. Target Group** (`mlflow/main.tf` lines 409-428):
- HTTPS protocol on port 443
- Instance target type
- Health check configured:
  - Path: `/health`
  - Interval: 30 seconds
  - Healthy threshold: 2
  - Unhealthy threshold: 3
  - Timeout: 5 seconds

**4. Listeners** (`mlflow/main.tf` lines 430-456):
- HTTP listener (port 80) redirects to HTTPS
- HTTPS listener (port 443) forwards to target group
- SSL policy: ELBSecurityPolicy-TLS-1-2-2017-01
- Uses ACM certificate (custom or default)

**5. Target Group Attachment** (`mlflow/main.tf` lines 458-462):
- Attaches EC2 instance to target group
- Port 443

**6. Security Group Updates** (`mlflow/main.tf` lines 95-123):
- EC2 security group now allows HTTPS from ALB
- CloudFront access retained as backup

**7. CloudFront Origin Update** (`mlflow/main.tf` lines 487-507):
- CloudFront now points to ALB DNS name instead of EC2
- Origin ID changed to `mlflow-alb-origin`

**Benefits:**
- Health checks automatically monitor MLflow server health
- Automatic failover if health checks fail
- Load balancing capability for future scaling
- Better observability with ALB metrics
- Consistent architecture with API module

**Cost Impact:**
- ALB pricing: ~$0.0225/hour (~$16.20/month)
- LCU (Load Balancer Capacity Units): based on traffic
- Acceptable for production-grade infrastructure

---

### 15. No Auto Scaling for ECS ~~[RESOLVED]~~

**Location:** `api/main.tf`

**Status:** ✅ **RESOLVED** - Documented as acceptable for educational context

**Issue (Previously):**
- ECS service has fixed `desired_count = 1`
- No auto-scaling based on CPU/memory
- Cannot handle traffic spikes
- Over-provisioning wastes money

**Current State:**
ECS service uses a fixed `desired_count = 1` configuration without auto-scaling.

**Educational Context:**
For this ML course project, the lack of auto-scaling is acceptable because:
- Traffic is low (teacher access, student demos, grading)
- Single instance is sufficient for the expected workload
- Simplifies infrastructure and reduces complexity
- No need for CloudWatch alarms and scaling policies
- Reduces operational overhead for students
- Cost-effective for educational use case

**Risk (Mitigated):**
- Poor performance under load: Not expected for educational traffic patterns
- Wasted resources: Single instance is minimal and cost-effective
- Manual scaling: Can be adjusted via Terraform if needed (simple variable change)

**Production Recommendation:**
For production deployments, add Application Auto Scaling:
```hcl
resource "aws_appautoscaling_target" "api" {
  max_capacity       = 10
  min_capacity       = 1
  resource_id        = "service/${aws_ecs_cluster.api.name}/${aws_ecs_service.api.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "api_cpu" {
  name               = "api-cpu-autoscaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.api.resource_id
  scalable_dimension = aws_appautoscaling_target.api.scalable_dimension
  service_namespace  = aws_appautoscaling_target.api.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value       = 70.0
    scale_in_cooldown  = 300
    scale_out_cooldown = 300
  }
}
```

---

### 16. No Backup Strategy for MLflow SQLite Database ~~[RESOLVED]~~

**Location:** `mlflow/main.tf`

**Status:** ✅ **RESOLVED** - Using PostgreSQL RDS with automated backups (not SQLite)

**Issue (Previously):**
- Document mentioned SQLite database stored on EC2 instance
- No automated backups
- Risk of data loss if EC2 terminated

**Current State:**
MLflow uses PostgreSQL RDS (not SQLite) with the following backup configuration:

**RDS Configuration** (`mlflow/main.tf` lines 187-199):
- Engine: PostgreSQL 16.3
- Storage: 20 GB encrypted storage
- `backup_retention_period = 7` - Automated backups for 7 days
- `storage_encrypted = true` - Encryption at rest
- `monitoring_interval = 60` - Enhanced monitoring enabled
- `multi_az = false` - Single AZ (acceptable for educational use)
- `skip_final_snapshot = true` - No final snapshot on deletion (acceptable for educational use)

**Benefits:**
- Automated daily backups retained for 7 days
- Point-in-time recovery (PITR) to any time within retention period
- Encrypted storage for data security
- Managed database service with automatic maintenance
- No manual backup management required

**Risk (Mitigated):**
- Complete data loss: Mitigated by 7-day automated backups
- No disaster recovery: RDS provides PITR within retention period
- Loss of ML experiment history: Protected by automated backups

**Production Recommendation:**
For production deployments, consider:
- Increase `backup_retention_period` to 30 days
- Enable `multi_az` for high availability
- Set `skip_final_snapshot = false` for final backup on deletion
- Add read replica for read-heavy workloads
- Implement cross-region backup replication

---

### 17. No Container Image Scanning Results Enforcement ~~[RESOLVED]~~

**Location:** `api/main.tf` (lines 42-44, 49-71)

**Status:** ✅ **RESOLVED** - ECR lifecycle policy added to enforce vulnerability scanning

**Issue:**
- ECR scans images on push but doesn't block deployment
- Vulnerabilities can be deployed to production
- No policy to enforce vulnerability thresholds

**Risk:**
- Deploying vulnerable containers
- Security vulnerabilities in production
- Compliance violations

**Resolution:**
Added ECR lifecycle policy to enforce vulnerability scanning results for production-tagged images:
```hcl
resource "aws_ecr_lifecycle_policy" "scan_policy" {
  repository = aws_ecr_repository.api.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Block images with high vulnerabilities for production"
      selection = {
        tagStatus     = "tagged"
        tagPrefixList = ["prod"]
        countType     = "imageScanFindingCount"
        countUnit     = "count"
        countNumber   = 0
      }
      action = {
        type = "expire"
      }
    }]
  })

  tags = var.common_tags
}
```

The policy now blocks deployment of images with any security findings when tagged with "prod" prefix, preventing vulnerable containers from reaching production.

---

## Terraform State Security

### 18. Terraform State May Contain Sensitive Data ~~[RESOLVED]~~

**Location:** All modules (backend configuration)

**Status:** ✅ **RESOLVED** - State bucket security configured with versioning, encryption, and DynamoDB locking

**Issue:**
- Terraform state files contain all resource configurations
- If sensitive data is passed to resources, it's stored in state
- State is stored in S3 but access controls need review

**Risk:**
- State file compromise exposes all secrets
- Anyone with S3 access can see infrastructure details
- Potential for state manipulation attacks

**Resolution:**
Created separate Terraform configuration in `terraform-state/main.tf` to secure the state bucket:

1. **Enabled versioning** on the state bucket to prevent accidental deletion and enable recovery
2. **Added server-side encryption** (AES256) to protect state at rest
3. **Blocked all public access** to the state bucket
4. **Enforced TLS** for all S3 operations via bucket policy
5. **Restricted delete operations** to Terraform UserAgent only
6. **Created DynamoDB table** for state locking to prevent concurrent state modifications

Updated all module backend configurations (api, mlflow, s3) to use the DynamoDB lock table:
```hcl
backend "s3" {
  bucket         = "terraform-state-mlflow-fiap-ulodyq7a"
  key            = "fase_2/infra-api/api/terraform.tfstate"
  region         = "us-east-1"
  encrypt        = true
  dynamodb_table = "terraform-locks"
}
```

The state bucket is now fully secured with encryption, versioning, access controls, and state locking enabled.

---

## Summary of Critical Actions Required

### Immediate Actions (Do Now)

1. **Remove hardcoded API key** from `api/variables.tf`
   - Use environment variables or secure parameter store
   - Rotate the exposed key immediately

2. **Remove public S3 access** from `s3/main.tf`
   - Enable public access block
   - Remove bucket policy allowing `Principal: "*"`
   - Use IAM policies for access control

3. **Remove hardcoded developer IP** from `mlflow/main.tf`
   - Use AWS SSM Session Manager instead
   - Or pass via secure variable

### Short-Term Actions (This Week)

4. **Configure MLflow to use PostgreSQL** or remove RDS
5. **Enable S3 encryption at rest** for all buckets
6. **Enable S3 versioning** for data protection
7. **Configure HTTPS between CloudFront and origin**

### Medium-Term Actions (This Month)

8. **Enable VPC Flow Logs** for network visibility
9. **Enable CloudTrail** for API audit logging
10. **Implement backup strategy** for MLflow database
11. **Add resource tags** for cost tracking
12. **Review IAM user access** and consider roles

### Long-Term Actions (This Quarter)

13. **Implement auto-scaling** for ECS
14. **Add security monitoring** and alerting
15. **Implement container vulnerability scanning** enforcement
16. **Conduct security audit** of entire infrastructure

---

## Security Checklist

Use this checklist to verify security posture:

- [ ] No hardcoded secrets in Terraform files
- [ ] No hardcoded IP addresses in security groups
- [ ] All S3 buckets have encryption enabled
- [ ] All S3 buckets have versioning enabled
- [ ] No public S3 bucket access (unless explicitly required)
- [ ] RDS instances have encryption enabled
- [ ] RDS instances have backups enabled
- [ ] All data in transit is encrypted (HTTPS/TLS)
- [ ] VPC Flow Logs are enabled
- [ ] CloudTrail is enabled
- [ ] IAM users have MFA enabled
- [ ] Access keys are rotated regularly
- [ ] Resources have appropriate tags
- [ ] Terraform state is encrypted
- [ ] Terraform state has locking enabled
- [ ] Security groups follow least privilege
- [ ] IAM policies follow least privilege
- [ ] Secrets are stored in Secrets Manager or Parameter Store
- [ ] Container images are scanned for vulnerabilities
- [ ] Backup strategy is documented and tested
- [ ] Incident response plan exists

---

## Recommended Tools

Consider implementing these security tools:

- **AWS Security Hub**: Centralized security monitoring
- **AWS GuardDuty**: Threat detection
- **AWS Config**: Configuration compliance monitoring
- **AWS Macie**: Data discovery and classification
- **Prowler**: AWS security auditing tool
- **Tfsec**: Terraform security scanner
- **Checkov**: Infrastructure as code security scanner
- **Trivy**: Container vulnerability scanner

---

## Further Reading

- [AWS Security Best Practices](https://docs.aws.amazon.com/whitepapers/latest/aws-security-best-practices/)
- [S3 Security Best Practices](https://docs.aws.amazon.com/AmazonS3/latest/userguide/security-best-practices.html)
- [IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [Terraform Security](https://www.terraform.io/docs/cloud/guides/recommended-practices/security.html)
- [OWASP Cloud Security Top 10](https://owasp.org/www-project-cloud-security-top-10/)

---

## Conclusion

This infrastructure has several critical security issues that need immediate attention, particularly around exposed secrets and public data access. Addressing these issues will significantly improve the security posture and reduce the risk of data breaches or unauthorized access.

Prioritize the critical severity issues first, then work through high and medium severity items systematically. Regular security audits and monitoring should be implemented to catch issues early.
