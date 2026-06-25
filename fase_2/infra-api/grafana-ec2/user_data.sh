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

# Configure Grafana for anonymous access with admin password
cat > /etc/grafana/grafana.ini << 'EOF'
[server]
http_addr = 0.0.0.0
http_port = 3000

[security]
admin_user = admin
admin_password = GrafanaAdmin123!
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

  # Configure CloudWatch Metrics data source
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

  echo "CloudWatch data source configured"

  # Configure CloudWatch Logs data source
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
else
  echo "Failed to create API key - data sources not configured"
  echo "You will need to configure them manually in Grafana"
fi

echo "Grafana setup complete with anonymous access enabled"
