#!/bin/bash
# User data script for Prometheus EC2 instance

# Update system packages
apt-get update -y
apt-get install -y docker.io curl

# Start Docker service
systemctl start docker
systemctl enable docker

# Create Prometheus configuration
cat > /tmp/prometheus.yml <<EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'api'
    static_configs:
      - targets: ['${api_alb_dns}:8000']
    metrics_path: '/metrics/'
    scrape_interval: 10s
EOF

# Run Prometheus container with restart policy
docker run -d \
  --name prometheus \
  --restart unless-stopped \
  -p 9090:9090 \
  -v /tmp/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus:latest

# Create systemd service for Prometheus auto-start on boot
cat > /etc/systemd/system/prometheus.service <<EOF
[Unit]
Description=Prometheus Docker Container
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/docker start prometheus
ExecStop=/usr/bin/docker stop prometheus
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the systemd service
systemctl daemon-reload
systemctl enable prometheus.service

# Wait for Prometheus to start
sleep 10

# Verify Prometheus is running
docker ps | grep prometheus

echo "Prometheus setup complete with auto-start enabled"
