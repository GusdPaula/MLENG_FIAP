#!/bin/bash
set -e

# Update system
apt-get update -y

# Install dependencies
apt-get install -y apt-transport-https software-properties-common wget gnupg2

# Add Grafana GPG key and repository
wget -q -O - https://packages.grafana.com/gpg.key | apt-key add -
echo "deb https://packages.grafana.com/oss/deb stable main" | tee -a /etc/apt/sources.list.d/grafana.list

# Install Grafana
apt-get update -y
apt-get install -y grafana

# Generate random admin password
GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)

# Configure Grafana for anonymous access with random admin password
cat > /etc/grafana/grafana.ini << EOF
[server]
http_addr = 0.0.0.0
http_port = 3000

[security]
admin_user = admin
admin_password = ${grafana_admin_password}
allow_embedding = true
cookie_secure = true
cookie_samesite = lax
content_security_policy = true
strict_transport_security = true
x_content_type_options = true
x_xss_protection = true

[auth]
disable_login_form = false

[auth.anonymous]
enabled = true
org_name = Main Org.
org_role = Viewer

[users]
allow_sign_up = false

[log]
mode = console
level = info
EOF

# Install and configure Nginx as reverse proxy with SSL
apt-get install -y nginx certbot python3-certbot-nginx

# Create Grafana dashboards directory
mkdir -p /etc/grafana/provisioning/dashboards
mkdir -p /var/lib/grafana/dashboards

# Copy dashboard configuration
cat > /etc/grafana/provisioning/dashboards/dashboards.yml <<EOF
apiVersion: 1

providers:
  - name: 'Drift Detection Dashboards'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF

# Copy dashboard JSON (will be created via API)
cat > /var/lib/grafana/dashboards/api_metrics_dashboard.json <<EOF
{
  "dashboard": {
    "title": "API Metrics Dashboard",
    "uid": "api-metrics-dashboard",
    "panels": [
      {
        "id": 1,
        "title": "Drift Alerts Total",
        "type": "stat",
        "targets": [
          {
            "expr": "drift_alerts_total",
            "refId": "A",
            "legendFormat": "{{drift_type}} - {{severity}}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "thresholds"
            },
            "thresholds": {
              "steps": [
                {"color": "green", "value": null},
                {"color": "yellow", "value": 1},
                {"color": "red", "value": 5}
              ]
            }
          }
        }
      },
      {
        "id": 2,
        "title": "Drift Score Over Time",
        "type": "timeseries",
        "targets": [
          {
            "expr": "drift_score",
            "refId": "A",
            "legendFormat": "Drift Score"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {"mode": "palette-classic"},
            "custom": {"lineWidth": 2, "fillOpacity": 10}
          }
        }
      },
      {
        "id": 3,
        "title": "API Request Rate",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "refId": "A",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "id": 4,
        "title": "API Error Rate",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "refId": "A",
            "legendFormat": "5xx Errors"
          },
          {
            "expr": "rate(http_requests_total{status=~\"4..\"}[5m])",
            "refId": "B",
            "legendFormat": "4xx Errors"
          }
        ]
      },
      {
        "id": 5,
        "title": "Response Time",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])",
            "refId": "A",
            "legendFormat": "Average Response Time"
          }
        ]
      },
      {
        "id": 6,
        "title": "Active Requests",
        "type": "gauge",
        "targets": [
          {
            "expr": "http_requests_active",
            "refId": "A"
          }
        ]
      },
      {
        "id": 7,
        "title": "Request Count by Endpoint",
        "type": "piechart",
        "targets": [
          {
            "expr": "sum by (endpoint) (http_requests_total)",
            "refId": "A",
            "legendFormat": "{{endpoint}}"
          }
        ]
      },
      {
        "id": 8,
        "title": "Request Count by Method",
        "type": "piechart",
        "targets": [
          {
            "expr": "sum by (method) (http_requests_total)",
            "refId": "A",
            "legendFormat": "{{method}}"
          }
        ]
      },
      {
        "id": 9,
        "title": "System CPU Usage",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(process_cpu_seconds_total[5m])",
            "refId": "A",
            "legendFormat": "CPU Usage"
          }
        ]
      },
      {
        "id": 10,
        "title": "Memory Usage",
        "type": "timeseries",
        "targets": [
          {
            "expr": "process_resident_memory_bytes",
            "refId": "A",
            "legendFormat": "Memory Bytes"
          }
        ]
      }
    ],
    "refresh": "10s",
    "time": {
      "from": "now-1h",
      "to": "now"
    }
  }
}
EOF

# Generate self-signed SSL certificate for internal use
mkdir -p /etc/nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/nginx/ssl/grafana.key \
  -out /etc/nginx/ssl/grafana.crt \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

# Configure Nginx as reverse proxy
cat > /etc/nginx/sites-available/grafana << 'EOF'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Host $host;
    }

    location /api/health {
        proxy_pass http://localhost:3000/api/health;
        access_log off;
    }
}

server {
    listen 443 ssl;
    server_name _;

    ssl_certificate /etc/nginx/ssl/grafana.crt;
    ssl_certificate_key /etc/nginx/ssl/grafana.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Host $host;
    }

    location /api/health {
        proxy_pass http://localhost:3000/api/health;
        access_log off;
    }
}
EOF

# Remove default Nginx site
rm -f /etc/nginx/sites-enabled/default
ln -s /etc/nginx/sites-available/grafana /etc/nginx/sites-enabled/

# Test Nginx configuration
nginx -t

# Start services
systemctl enable grafana-server
systemctl start grafana-server

systemctl enable nginx
systemctl restart nginx

# Wait for Grafana to start and be ready
echo "Waiting for Grafana to be ready..."
for i in {1..30}; do
  if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
    echo "Grafana is ready"
    break
  fi
  echo "Waiting for Grafana... ($i/30)"
  sleep 5
done

# Configure CloudWatch data sources via Grafana API
GRAFANA_URL="http://localhost:3000"
GRAFANA_API_KEY=$(curl -s -X POST -H "Content-Type: application/json" \
  -d '{"name":"terraform-key","role":"Admin"}' \
  $GRAFANA_URL/api/auth/keys | jq -r '.key')

if [ -n "$GRAFANA_API_KEY" ] && [ "$GRAFANA_API_KEY" != "null" ]; then
  echo "API key created successfully"

  # Configure Prometheus data source
  PROMETHEUS_URL="${prometheus_url}"

  # Ensure URL has http:// prefix
  if [[ ! "$PROMETHEUS_URL" =~ ^https?:// ]]; then
    PROMETHEUS_URL="http://$$PROMETHEUS_URL"
  fi

  curl -s -X POST -H "Authorization: Bearer $GRAFANA_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{
      \"name\":\"Prometheus\",
      \"type\":\"prometheus\",
      \"access\":\"proxy\",
      \"url\":\"$PROMETHEUS_URL\",
      \"isDefault\":true,
      \"jsonData\":{
        \"timeInterval\":\"15s\"
      }
    }" \
    $GRAFANA_URL/api/datasources

  echo "Prometheus data source configured at $PROMETHEUS_URL"
else
  echo "Failed to create API key - data sources not configured"
  echo "You will need to configure them manually in Grafana"
fi

echo "Grafana setup complete with anonymous access enabled"
