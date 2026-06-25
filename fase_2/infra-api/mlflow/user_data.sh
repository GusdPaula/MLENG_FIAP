#!/bin/bash
set -e

# Log all output to /var/log/user-data.log
exec > /var/log/user-data.log 2>&1

echo "=== Starting MLflow Server Setup ==="
date

# Install Python and MLflow
echo "Installing Python and MLflow..."
apt-get update
apt-get install -y python3-pip python3-venv nginx awscli

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv /opt/mlflow-venv
source /opt/mlflow-venv/bin/activate

# Install MLflow and dependencies
echo "Installing MLflow and dependencies..."
pip install mlflow[extras] boto3 psycopg2-binary

# Create MLflow directory
mkdir -p /home/ubuntu/mlflow
cd /home/ubuntu/mlflow

# Get DB password from Secrets Manager
echo "Fetching DB password from Secrets Manager..."
export AWS_DEFAULT_REGION=us-east-1
DB_PASSWORD=$(aws secretsmanager get-secret-value \
  --secret-id ${db_password_secret_arn} \
  --query SecretString --output text)

# Create CloudWatch logging script
cat > /home/ubuntu/mlflow/cloudwatch_logger.py <<'CW_LOGGER'
import boto3
import logging
from datetime import datetime
import subprocess
import time

# Configure CloudWatch Logs
logs_client = boto3.client('logs', region_name='us-east-1')
log_group_name = '/mlflow-logs'
log_stream_name = 'mlflow-server'

# Create log group if it doesn't exist
try:
    logs_client.create_log_group(logGroupName=log_group_name)
except logs_client.exceptions.ResourceAlreadyExistsException:
    pass

# Create log stream if it doesn't exist
try:
    logs_client.create_log_stream(
        logGroupName=log_group_name,
        logStreamName=log_stream_name
    )
except logs_client.exceptions.ResourceAlreadyExistsException:
    pass

# Tail the MLflow log file and send to CloudWatch
sequence_token = None
log_file = '/var/log/mlflow-server.log'

while True:
    try:
        with open(log_file, 'r') as f:
            f.seek(0, 2)  # Go to the end of the file
            while True:
                line = f.readline()
                if line:
                    try:
                        log_events = [{
                            'timestamp': int(datetime.now().timestamp() * 1000),
                            'message': line.strip()
                        }]
                        if sequence_token:
                            logs_client.put_log_events(
                                logGroupName=log_group_name,
                                logStreamName=log_stream_name,
                                logEvents=log_events,
                                sequenceToken=sequence_token
                            )
                        else:
                            response = logs_client.put_log_events(
                                logGroupName=log_group_name,
                                logStreamName=log_stream_name,
                                logEvents=log_events
                            )
                            sequence_token = response['nextSequenceToken']
                    except Exception as e:
                        print(f"Failed to send log to CloudWatch: {e}")
                        sequence_token = None  # Reset token on error
                else:
                    time.sleep(1)
    except FileNotFoundError:
        time.sleep(5)  # Wait for log file to be created
    except Exception as e:
        print(f"Error reading log file: {e}")
        time.sleep(5)
CW_LOGGER

# Start CloudWatch logger in background
python3 /home/ubuntu/mlflow/cloudwatch_logger.py &

# Start MLflow server with PostgreSQL backend (on localhost only)
echo "Starting MLflow server with PostgreSQL backend..."
nohup mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri postgresql://mlflow_user:$DB_PASSWORD@${rds_endpoint}/mlflow \
  --default-artifact-root "s3://${s3_bucket}" \
  --serve-artifacts \
  --allowed-hosts "*" \
  --cors-allowed-origins "*" \
  > /var/log/mlflow-server.log 2>&1 &

echo "MLflow server started"

# Wait for server to be ready
echo "Waiting for server to be ready..."
sleep 20

# Check server status
echo "Server status:"
curl -f http://localhost:5000/health || echo "Health check failed"

# Generate self-signed SSL certificate for nginx
echo "Generating SSL certificate..."
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/ssl/private/mlflow.key \
  -out /etc/ssl/certs/mlflow.crt \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

# Configure nginx as reverse proxy with SSL
echo "Configuring nginx..."
cat > /etc/nginx/sites-available/mlflow <<'NGINX_CONF'
server {
    listen 80;
    server_name _;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name _;

    ssl_certificate /etc/ssl/certs/mlflow.crt;
    ssl_certificate_key /etc/ssl/private/mlflow.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
NGINX_CONF

# Remove default nginx site and enable mlflow site
rm -f /etc/nginx/sites-enabled/default
ln -s /etc/nginx/sites-available/mlflow /etc/nginx/sites-enabled/

# Test and restart nginx
nginx -t
systemctl restart nginx

echo "=== MLflow Server Setup Complete ==="
