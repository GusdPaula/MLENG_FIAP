#!/bin/bash
# Script to manually configure CloudWatch Logs data source in Grafana
# Run this script by SSHing into the Grafana EC2 instance

GRAFANA_URL="http://localhost:3000"

echo "Waiting for Grafana to be ready..."
for i in {1..30}; do
  if curl -s $GRAFANA_URL/api/health > /dev/null 2>&1; then
    echo "Grafana is ready"
    break
  fi
  echo "Waiting for Grafana... ($i/30)"
  sleep 5
done

# Create API key
echo "Creating API key..."
GRAFANA_API_KEY=$(curl -s -X POST -H "Content-Type: application/json" \
  -d '{"name":"setup-key","role":"Admin"}' \
  $GRAFANA_URL/api/auth/keys | jq -r '.key')

if [ -z "$GRAFANA_API_KEY" ] || [ "$GRAFANA_API_KEY" = "null" ]; then
  echo "Failed to create API key"
  exit 1
fi

echo "API key created successfully"

# Configure CloudWatch Metrics data source
echo "Configuring CloudWatch Metrics data source..."
curl -s -X POST -H "Authorization: Bearer $GRAFANA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name":"CloudWatch",
    "type":"cloudwatch",
    "access":"proxy",
    "jsonData":{
      "authType":"default",
      "defaultRegion":"us-east-1"
    }
  }' \
  $GRAFANA_URL/api/datasources

echo "CloudWatch Metrics data source configured"

# Configure CloudWatch Logs data source
echo "Configuring CloudWatch Logs data source..."
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

echo "CloudWatch Logs data source configured"

echo "Done! You can now import the log dashboards in Grafana"
